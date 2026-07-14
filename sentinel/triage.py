"""Impact-weighted triage: turn raw signals into a ranked exception queue.

Governance intensity should scale with impact. A drifting SKU worth $40/day
is a footnote; one worth $4,000/day is a P1. The queue is sorted by
revenue-at-risk so a human reviews the 10 exceptions that matter, not 200
rows of noise. Every exception also carries a plain-language explanation a
non-technical planner can act on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG

DATA_CHECKS = {"zeros", "frozen", "scale"}
HORIZON = int(CFG["triage"]["horizon_days"])


def _explain(primary: pd.Series, wape_base: float, wape_now: float, name: str) -> str:
    """One human sentence per exception, what happened and what it means."""
    check, v = primary["check"], primary["value"]
    if check == "scale":
        direction = "jumped roughly 10x overnight" if v > 1 else "collapsed to a tenth of normal"
        return (f"Recorded sales for {name} {direction}. This is almost certainly a unit or "
                f"pipeline bug, not real demand. Don't reorder based on this data.")
    if check == "zeros":
        return (f"{name} suddenly reported zero sales for a week or more, though it "
                f"almost never had zero-sales days before. This looks like a data feed "
                f"outage, not customers disappearing.")
    if check == "frozen":
        return (f"{name} has reported the exact same sales figure day after day. Real "
                f"demand is never that regular. The data extract is likely stuck.")
    if check == "bias":
        side = "under-forecasting (actual sales run higher than predicted)" if v > 0 \
            else "over-forecasting (predicting more than actually sells)"
        return (f"The model is persistently {side} for {name} by about {abs(v):.0%}. "
                f"Something about its demand changed after the model was trained.")
    if check == "accuracy":
        return (f"Forecast error for {name} grew from a normal {wape_base:.0%} to "
                f"{wape_now:.0%} and stayed there. The old sales pattern this model "
                f"learned no longer holds.")
    if check == "page_hinkley":
        return (f"A sequential drift test confirmed the forecast errors for {name} have "
                f"been trending in one direction for weeks: slow, steady change rather "
                f"than a one-off spike.")
    return (f"The pattern of forecast errors for {name} no longer matches its history. "
            f"Worth a look even though headline accuracy hasn't broken tolerance yet.")


def build_queue(
    signals: pd.DataFrame,
    metrics: pd.DataFrame,
    fcst: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()

    cat = catalog.set_index("sku")
    has_names = "name" in cat.columns
    price = cat["price"] if "price" in cat.columns else pd.Series(dtype=float)
    cal = fcst[fcst["window"] == "calibration"]
    avg_units = cal.groupby("sku")["actual"].mean()

    rows = []
    for sku, g in signals.groupby("sku"):
        g = g.sort_values("date")
        max_sev = int(g["severity"].max())
        status = "exception" if max_sev >= 2 else "watch"
        top = g.sort_values(["severity", "date"], ascending=[False, True]).iloc[0]

        m = metrics[metrics["sku"] == sku].sort_values("date")
        wape_now = float(m["wape_28"].iloc[-1]) if len(m) else np.nan
        wape_base = float(m["wape_baseline"].iloc[0]) if len(m) else np.nan
        excess = max(wape_now - wape_base, 0.15 if max_sev == 3 else 0.0)

        daily_rev = float(avg_units.get(sku, 0.0) * price.get(sku, 1.0))
        rev_at_risk = excess * daily_rev * HORIZON

        name = str(cat.loc[sku, "name"]) if has_names and sku in cat.index else sku
        category = str(cat.loc[sku, "category"]) if has_names and sku in cat.index else "-"
        issue = "data quality" if top["check"] in DATA_CHECKS else "model drift"
        rows.append(
            {
                "sku": sku,
                "product": name,
                "category": category,
                "status": status,
                "issue_class": issue,
                "primary_check": top["check"],
                "checks_fired": ", ".join(g["check"].tolist()),
                "first_detected": g["date"].min(),
                "wape_baseline": round(wape_base, 3),
                "wape_now": round(wape_now, 3),
                "avg_daily_revenue": round(daily_rev, 2),
                "revenue_at_risk_28d": round(rev_at_risk, 2),
                "recommended_action": top["action"],
                "explanation": _explain(top, wape_base, wape_now, name),
            }
        )

    queue = pd.DataFrame(rows).sort_values(
        ["status", "revenue_at_risk_28d"], ascending=[True, False]
    )
    return queue.reset_index(drop=True)
