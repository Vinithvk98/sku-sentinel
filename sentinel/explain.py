"""The Explain layer: WHY does demand move, not just what it will be.

Two capabilities:

1. Promo uplift estimation (the causal question).
   "Do promotions actually lift sales for this product, and by how much?"
   The trap: promo days differ from normal days in other ways too (they start
   on specific weekdays, cluster in seasons). A naive promo-vs-non-promo
   comparison inherits all that confounding. We estimate uplift two ways:

     naive     mean(promo days) / mean(non-promo days)  , confounded
     adjusted  coefficient of the promo flag in a log-space regression that
               ALSO includes weekday/season/trend, confounders controlled

   Because the simulator KNOWS each SKU's true promo lift, we can score both
   estimators against ground truth, measured honesty, again.

2. Driver decomposition (the interpretability question).
   In a log-space linear model, coefficient groups have a clean meaning:
   exp(coef) is a multiplicative effect. We aggregate feature groups into
   plain-language effect sizes: weekday swing, seasonal swing, promo lift,
   annual trend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .forecast import _features
from .simulate import TRAIN_END

# feature layout from forecast._features:
# [0:7]=dow, [7:19]=month, [19:21]=annual sin/cos, [21]=trend, [22]=promo
DOW, MONTH, ANNUAL, TREND, PROMO = slice(0, 7), slice(7, 19), slice(19, 21), 21, 22


def _fit(g: pd.DataFrame) -> tuple[Ridge, np.ndarray, pd.DatetimeIndex]:
    g = g.sort_values("date").reset_index(drop=True).iloc[:TRAIN_END]
    dates = pd.DatetimeIndex(g["date"])
    X = _features(dates, g["promo"].values)
    model = Ridge(alpha=2.0)
    model.fit(X, np.log1p(g["units"].values))
    return model, X, dates


def uplift_for_sku(g: pd.DataFrame) -> dict | None:
    """Naive vs regression-adjusted promo uplift for one SKU (train window only)."""
    tr = g.sort_values("date").reset_index(drop=True).iloc[:TRAIN_END]
    if tr["promo"].sum() < 14:  # not promoted enough to estimate anything
        return None
    naive = float(tr.loc[tr["promo"] == 1, "units"].mean()
                  / max(tr.loc[tr["promo"] == 0, "units"].mean(), 1e-9))
    model, _, _ = _fit(g)
    adjusted = float(np.exp(model.coef_[PROMO]))
    return {"naive_lift": round(naive, 3), "adjusted_lift": round(adjusted, 3)}


def drivers_for_sku(g: pd.DataFrame) -> dict:
    """Plain-language effect sizes from the log-space model's coefficients."""
    model, X, dates = _fit(g)
    c = model.coef_

    dow_swing = float(np.exp(c[DOW].max() - c[DOW].min()) - 1)

    # seasonal component over one year: month dummies + annual sin/cos together
    year = pd.date_range("2025-01-01", periods=365, freq="D")
    Xy = _features(year, np.zeros(365))
    seasonal = Xy[:, MONTH] @ c[MONTH] + Xy[:, ANNUAL] @ c[ANNUAL]
    season_swing = float(np.exp(seasonal.max() - seasonal.min()) - 1)

    trend_yr = float(np.exp(c[TREND]) - 1)  # trend feature is t/365 -> per-year
    promo_lift = float(np.exp(c[PROMO]) - 1) if X[:, PROMO].sum() >= 14 else None

    return {
        "weekday_swing": round(dow_swing, 3),
        "season_swing": round(season_swing, 3),
        "trend_per_year": round(trend_yr, 3),
        "promo_lift": round(promo_lift, 3) if promo_lift is not None else None,
        "best_weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(np.argmax(c[DOW]))],
        "worst_weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(np.argmin(c[DOW]))],
    }


def uplift_fleet(panel: pd.DataFrame, catalog: pd.DataFrame | None = None) -> dict:
    """Uplift estimates for every promoted SKU; scored vs truth when available."""
    rows = []
    truth = (catalog.set_index("sku")["promo_lift"].to_dict()
             if catalog is not None and "promo_lift" in catalog.columns else {})
    names = (catalog.set_index("sku")["name"].to_dict()
             if catalog is not None and "name" in catalog.columns else {})
    for sku, g in panel.groupby("sku", sort=False):
        est = uplift_for_sku(g)
        if est is None:
            continue
        row = {"sku": sku, "product": names.get(sku, sku), **est}
        if sku in truth:
            row["true_lift"] = round(float(truth[sku]), 3)
        rows.append(row)

    out: dict = {"skus": rows}
    scored = [r for r in rows if "true_lift" in r]
    if scored:
        t = np.array([r["true_lift"] for r in scored])
        a = np.array([r["adjusted_lift"] for r in scored])
        n = np.array([r["naive_lift"] for r in scored])
        out["scorecard"] = {
            "n_skus": len(scored),
            "adjusted_mae": round(float(np.abs(a - t).mean()), 3),
            "naive_mae": round(float(np.abs(n - t).mean()), 3),
            "adjusted_corr": round(float(np.corrcoef(a, t)[0, 1]), 3),
            "naive_corr": round(float(np.corrcoef(n, t)[0, 1]), 3),
        }
    return out
