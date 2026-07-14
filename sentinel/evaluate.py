"""Score the monitor against the injected ground truth.

Because drift events were injected synthetically, we can measure the thing
that actually matters for a governance layer: does it catch real problems
(recall), does it cry wolf (precision), and how fast (detection delay)?
"""
from __future__ import annotations

import pandas as pd


def score(queue: pd.DataFrame, ground_truth: pd.DataFrame) -> dict:
    flagged = set(queue.loc[queue["status"] == "exception", "sku"]) if len(queue) else set()
    watch = set(queue.loc[queue["status"] == "watch", "sku"]) if len(queue) else set()
    truth = set(ground_truth["sku"])

    tp = flagged & truth
    fp = flagged - truth
    fn = truth - flagged

    delays, per_type = [], []
    gt_idx = ground_truth.set_index("sku")
    q_idx = queue.set_index("sku") if len(queue) else None
    for sku in sorted(truth):
        etype = gt_idx.loc[sku, "event_type"]
        edate = pd.Timestamp(gt_idx.loc[sku, "event_date"])
        caught = sku in tp
        delay = None
        if caught:
            delay = int((pd.Timestamp(q_idx.loc[sku, "first_detected"]) - edate).days)
            delays.append(max(delay, 0))
        product = (
            str(gt_idx.loc[sku, "name"]) if "name" in ground_truth.columns else sku
        )
        per_type.append(
            {"product": product, "sku": sku, "event_type": etype,
             "event_date": str(edate.date()), "detected": caught,
             "delay_days": delay, "also_watchlisted": sku in watch}
        )

    by_type = (
        pd.DataFrame(per_type)
        .groupby("event_type")
        .agg(events=("sku", "count"), detected=("detected", "sum"))
        .reset_index()
    )
    by_type["recall"] = (by_type["detected"] / by_type["events"]).round(2)

    return {
        "n_events": len(truth),
        "true_positives": len(tp),
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "precision": round(len(tp) / max(len(flagged), 1), 3),
        "recall": round(len(tp) / max(len(truth), 1), 3),
        "median_detection_delay_days": (
            float(pd.Series(delays).median()) if delays else None
        ),
        "missed": sorted(fn),
        "false_positive_skus": sorted(fp),
        "per_event": per_type,
        "by_type": by_type.to_dict(orient="records"),
    }
