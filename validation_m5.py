"""Validate SKU Sentinel on REAL retail data: the M5 (Walmart) dataset.

The synthetic demo proves detection quality against known ground truth. This
script asks the harder real-world question: what does the monitor report on
real, messy Walmart demand, under three forecaster regimes?

  1 FROZEN     calendar-only model, trained once, never updated.
  2 RETRAINED  same model refit every 28 days.
  3 INFORMED   refit every 28 days AND given the drivers that actually move
               Walmart demand: item price (dips = promotions), SNAP benefit
               days, and holiday events, from calendar.csv + sell_prices.csv.

Reading the results:
  frozen vs retrained   the cost of model staleness
  retrained vs informed the (much larger) cost of missing regressors,
                        the M5 competition's core lesson
  zeros/scale checks    identical across regimes because they test the DATA,
                        not the model: genuine stockouts, discontinuations
                        and level breaks found in real Walmart history.

Setup (one time):
    1. Kaggle account; accept rules at
       https://www.kaggle.com/competitions/m5-forecasting-accuracy
    2. From the Data tab download into ./data/:
         sales_train_evaluation.csv   (required)
         calendar.csv                 (for regime 3)
         sell_prices.csv              (for regime 3)

Run:
    python validation_m5.py [n_skus]
Writes output_m5/report_{frozen,retrained,informed}.html
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from sentinel import forecast as fc
from sentinel import simulate as sim
from sentinel.forecast import _features, fit_and_forecast
from sentinel.monitor import run_monitor
from sentinel.triage import build_queue
from sentinel.report import build_report

BASE = pathlib.Path(__file__).parent
DATA = BASE / "data" / "sales_train_evaluation.csv"
CAL = BASE / "data" / "calendar.csv"
PRICES = BASE / "data" / "sell_prices.csv"
STORE = "CA_1"
RETRAIN_EVERY = 28
PROMO_PRICE_DIP = 0.95  # price below 95% of the item's median = promo proxy


def load_m5_panel(n_skus: int = 300) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if not DATA.exists():
        sys.exit(f"M5 data not found at {DATA}. See setup instructions in the docstring.")
    raw = pd.read_csv(DATA)
    raw = raw[(raw["store_id"] == STORE) & (raw["cat_id"] == "FOODS")]
    day_cols = [c for c in raw.columns if c.startswith("d_")]
    day_cols = day_cols[-730:]  # last 730 days, mirroring the synthetic setup
    totals = raw[day_cols].sum(axis=1)
    raw = raw.loc[totals.sort_values(ascending=False).index[:n_skus]]

    dates = pd.date_range("2014-05-23", periods=len(day_cols), freq="D")
    panel = raw.melt(
        id_vars=["item_id"], value_vars=day_cols, var_name="d", value_name="units"
    )
    panel["date"] = panel["d"].map(dict(zip(day_cols, dates)))
    panel = panel.rename(columns={"item_id": "sku"})
    panel["promo"] = 0
    panel["price"] = 1.0
    panel["revenue"] = panel["units"]
    cols = ["date", "sku", "units", "promo", "price", "revenue", "d"]
    return panel[cols], pd.DataFrame(
        {"sku": panel["sku"].unique(), "price": 1.0}
    ), day_cols


def load_drivers(panel: pd.DataFrame, day_cols: list[str]) -> pd.DataFrame | None:
    """Attach real drivers: SNAP days, holiday events, price, price-dip promos."""
    if not (CAL.exists() and PRICES.exists()):
        return None
    cal = pd.read_csv(CAL)
    cal = cal[cal["d"].isin(day_cols)][
        ["d", "wm_yr_wk", "event_name_1", f"snap_{STORE.split('_')[0]}"]
    ].rename(columns={f"snap_{STORE.split('_')[0]}": "snap"})
    cal["event"] = cal["event_name_1"].notna().astype(int)

    prices = pd.read_csv(PRICES)
    prices = prices[
        (prices["store_id"] == STORE) & (prices["item_id"].isin(panel["sku"].unique()))
    ][["item_id", "wm_yr_wk", "sell_price"]].rename(columns={"item_id": "sku"})

    out = panel.merge(cal[["d", "wm_yr_wk", "snap", "event"]], on="d", how="left")
    out = out.merge(prices, on=["sku", "wm_yr_wk"], how="left")
    out["sell_price"] = out.groupby("sku")["sell_price"].transform(
        lambda s: s.ffill().bfill()
    )
    med = out.groupby("sku")["sell_price"].transform("median")
    out["price_rel"] = (out["sell_price"] / med).fillna(1.0)
    out["promo"] = (out["price_rel"] < PROMO_PRICE_DIP).astype(int)
    out[["snap", "event"]] = out[["snap", "event"]].fillna(0)
    return out


def rolling_forecast(panel: pd.DataFrame, train_end: int, calib_end: int,
                     extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Ridge refit every RETRAIN_EVERY days on all data so far, optionally with
    extra driver features (snap, event, relative price)."""
    out = []
    for sku, g in panel.groupby("sku", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        dates = pd.DatetimeIndex(g["date"])
        X = _features(dates, g["promo"].values)
        if extra_cols:
            X = np.column_stack([X] + [g[c].values.astype(float) for c in extra_cols])
        y = np.log1p(g["units"].values)
        n = len(g)

        preds = np.zeros(n - train_end)
        for block in range(train_end, n, RETRAIN_EVERY):
            hi = min(block + RETRAIN_EVERY, n)
            model = Ridge(alpha=2.0)
            model.fit(X[:block], y[:block])
            preds[block - train_end : hi - train_end] = np.clip(
                np.expm1(model.predict(X[block:hi])), 0, None
            )

        out.append(pd.DataFrame({
            "sku": sku,
            "date": dates[train_end:],
            "forecast": preds,
            "actual": g["units"].values[train_end:],
            "promo": g["promo"].values[train_end:],
        }))
    fcst = pd.concat(out, ignore_index=True)
    split_date = pd.DatetimeIndex(panel["date"].unique()).sort_values()[calib_end]
    fcst["window"] = np.where(fcst["date"] < split_date, "calibration", "monitoring")
    return fcst


def run_regime(name: str, fcst: pd.DataFrame, catalog: pd.DataFrame,
               n_skus: int, out: pathlib.Path) -> int:
    signals, metrics = run_monitor(fcst, catalog)
    queue = build_queue(signals, metrics, fcst, catalog)
    n_exc = int((queue["status"] == "exception").sum()) if len(queue) else 0
    print(f"\n[{name}] exceptions: {n_exc} / {n_skus} SKUs ({n_exc / n_skus:.1%})")
    if len(queue):
        by_check = queue[queue["status"] == "exception"]["primary_check"].value_counts()
        print(f"[{name}] by primary check: {by_check.to_dict()}")
    queue.to_csv(out / f"queue_{name}.csv", index=False)
    build_report(queue, signals, fcst, None, None, n_skus,
                 str(out / f"report_{name}.html"))
    return n_exc


def main(n_skus: int = 300) -> None:
    out = BASE / "output_m5"
    out.mkdir(exist_ok=True)

    panel, catalog, day_cols = load_m5_panel(n_skus)
    n_days = panel["date"].nunique()
    train_end = int(n_days * 0.62)
    calib_end = int(n_days * 0.745)
    fc.TRAIN_END = sim.TRAIN_END = train_end
    fc.CALIB_END = sim.CALIB_END = calib_end

    print(f"{n_skus} real SKUs x {n_days} days")

    print("\nregime 1/3: FROZEN calendar-only model ...")
    n_frozen = run_regime(
        "frozen", fit_and_forecast(panel.drop(columns="d")), catalog, n_skus, out
    )

    print(f"\nregime 2/3: RETRAINED every {RETRAIN_EVERY} days, calendar-only ...")
    n_retrained = run_regime(
        "retrained", rolling_forecast(panel, train_end, calib_end), catalog, n_skus, out
    )

    informed = load_drivers(panel, day_cols)
    n_informed = None
    if informed is None:
        print("\nregime 3/3 skipped: calendar.csv / sell_prices.csv not found in ./data/")
    else:
        print(f"\nregime 3/3: INFORMED (price, SNAP, events) + retrained ...")
        n_informed = run_regime(
            "informed",
            rolling_forecast(informed, train_end, calib_end,
                             extra_cols=["snap", "event", "price_rel"]),
            catalog, n_skus, out,
        )

    print("\n" + "=" * 62)
    print(f"1 frozen, calendar-only:    {n_frozen / n_skus:6.1%} of SKUs in exception")
    print(f"2 retrained, calendar-only: {n_retrained / n_skus:6.1%}")
    if n_informed is not None:
        print(f"3 retrained + real drivers: {n_informed / n_skus:6.1%}")
        print("\n1->2 is the cost of staleness. 2->3 is the (larger) cost of")
        print("missing regressors. What remains in 3 is genuinely anomalous.")
    print(f"reports in {out}/")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
