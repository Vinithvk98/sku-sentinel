"""End-to-end demo: simulate -> forecast -> monitor -> triage -> score -> report.

Usage:  python run_demo.py [seed]
Writes everything to ./output (CSVs, scorecard.json, report.html).
"""
from __future__ import annotations

import json
import pathlib
import sys

from sentinel.simulate import generate, N_SKUS
from sentinel.forecast import fit_and_forecast
from sentinel.monitor import run_monitor
from sentinel.triage import build_queue
from sentinel.evaluate import score
from sentinel.report import build_report


def main(seed: int = 42) -> None:
    out = pathlib.Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)

    print("1/5 simulating 200-SKU demand panel with injected drift ...")
    panel, catalog, gt = generate(seed)

    print("2/5 fitting per-SKU baseline forecasters (frozen after training) ...")
    fcst = fit_and_forecast(panel)

    print("3/5 running the governance engine ...")
    signals, metrics = run_monitor(fcst, catalog)

    print("4/5 triaging exceptions by revenue at risk ...")
    queue = build_queue(signals, metrics, fcst, catalog)
    sc = score(queue, gt)

    print("5/5 writing report ...")
    panel.to_csv(out / "panel.csv", index=False)
    catalog.to_csv(out / "catalog.csv", index=False)
    gt.drop(columns=["detail"]).to_csv(out / "ground_truth.csv", index=False)
    fcst.to_csv(out / "forecasts.csv", index=False)
    signals.to_csv(out / "signals.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    queue.to_csv(out / "queue.csv", index=False)
    with open(out / "scorecard.json", "w") as f:
        json.dump(sc, f, indent=2, default=str)
    build_report(queue, signals, fcst, sc, gt, N_SKUS, str(out / "report.html"))

    print(
        f"\ndone. recall={sc['recall']:.0%} precision={sc['precision']:.0%} "
        f"median delay={sc['median_detection_delay_days']:.0f}d "
        f"({sc['false_positives']} false alarms across {N_SKUS} SKUs / 6 months)"
    )
    print(f"open {out / 'report.html'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)  # None -> config.yaml seed
