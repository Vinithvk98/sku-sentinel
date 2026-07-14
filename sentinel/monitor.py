"""The governance engine: statistical monitoring of a live forecast fleet.

Design principle (borrowed from how air-traffic control actually works):
high-volume, low-individual-impact decisions like SKU replenishment should be
governed statistically at the SYSTEM level -- tolerance bands, drift tests,
exception queues -- not by per-decision human sign-off.

Checks run at weekly checkpoints over a trailing 28-day window, calibrated
per SKU against a clean out-of-sample calibration period:

  accuracy    rolling WAPE vs. calibrated tolerance band
  bias        sustained signed error (forecast persistently high/low)
  psi         Population Stability Index on relative residuals
  ks          Kolmogorov-Smirnov test on relative residuals
  page_hinkley sequential concept-drift test on the error stream
  frozen      identical values repeating (stale data feed)
  zeros       zero-run where the SKU historically always sold (feed outage)
  scale       order-of-magnitude break vs. forecast (ETL unit bug)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import CFG

_M = CFG["monitor"]
# Operating point chosen by sweeping thresholds across simulation seeds:
# precision 0.80 / recall 0.88 on injected events, with data-quality breaks
# (frozen feeds, outages, scale bugs) detected at the first weekly checkpoint.
# All values are overridable in config.yaml.
WINDOW = int(_M["window_days"])
CHECKPOINT_EVERY = int(_M["checkpoint_every"])
BIAS_FLOOR = float(_M["bias_floor"])      # min relative-bias delta to alarm
WAPE_FACTOR = float(_M["wape_factor"])    # WAPE tolerance = max(factor*base, base+0.10)
PERSIST_MODEL = int(_M["persist_checkpoints"])
PSI_BASE = float(_M["psi_base_threshold"])
KS_P = float(_M["ks_p_value"])

# severity levels
WATCH, ALERT, CRITICAL = 1, 2, 3

ACTIONS = {
    "zeros": "Data feed outage suspected. Check pipeline ingestion before acting on forecasts",
    "frozen": "Stale feed: actuals frozen on a constant. Check the upstream extract",
    "scale": "Unit/scale break. Validate ETL transforms and quarantine model output",
    "bias": "Sustained bias. Retrain the model or add the missing regressor",
    "accuracy": "Accuracy outside tolerance. Investigate driver changes and consider a retrain",
    "page_hinkley": "Concept drift detected in the error stream. Schedule a retrain",
    "psi": "Residual distribution shift. Review segment and seasonality assumptions",
    "ks": "Residual distribution shift. Review segment and seasonality assumptions",
}


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = np.abs(actual).sum()
    return float(np.abs(actual - forecast).sum() / denom) if denom > 0 else np.nan


def psi(baseline: np.ndarray, recent: np.ndarray, bins: int = 5) -> float:
    qs = np.quantile(baseline, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    b = np.clip(np.histogram(baseline, qs)[0] / max(len(baseline), 1), 1e-4, None)
    r = np.clip(np.histogram(recent, qs)[0] / max(len(recent), 1), 1e-4, None)
    return float(np.sum((r - b) * np.log(r / b)))


def psi_null_expectation(n_baseline: int, n_recent: int, bins: int = 5) -> float:
    """Expected PSI under the null hypothesis (no drift) for finite samples.

    PSI is chi-square-like: E[PSI] ~ (bins - 1) * (1/n_recent + 1/n_baseline).
    Ignoring this on small windows is the #1 source of false drift alarms."""
    return (bins - 1) * (1.0 / n_recent + 1.0 / n_baseline)


def page_hinkley(errors: np.ndarray, delta: float, lam: float) -> int | None:
    """Two-sided Page-Hinkley. Returns index of first alarm, or None."""
    mean, mt_pos, mt_neg, M_pos, m_neg = 0.0, 0.0, 0.0, 0.0, 0.0
    for i, x in enumerate(errors):
        mean += (x - mean) / (i + 1)
        mt_pos += x - mean - delta
        mt_neg += x - mean + delta
        M_pos = min(M_pos, mt_pos)
        m_neg = max(m_neg, mt_neg)
        if mt_pos - M_pos > lam or m_neg - mt_neg > lam:
            return i
    return None


def _max_run(values: np.ndarray) -> int:
    if len(values) == 0:
        return 0
    best = run = 1
    for a, b in zip(values[:-1], values[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def _max_zero_run(values: np.ndarray) -> int:
    best = run = 0
    for v in values:
        run = run + 1 if v == 0 else 0
        best = max(best, run)
    return best


def _calibration_baseline(g: pd.DataFrame) -> dict:
    cal = g[g["window"] == "calibration"]
    a, f = cal["actual"].values, cal["forecast"].values
    rel_err = (a - f) / np.maximum(f, 1.0)
    nz = a[a > 0]
    return {
        "wape": wape(a, f),
        "rel_err": rel_err,
        "rel_err_std": float(np.std(rel_err) + 1e-9),
        "bias": float(np.mean(rel_err)),
        "zero_rate": float((a == 0).mean()),
        "max_run": _max_run(a),
        "median_nz": float(np.median(nz)) if len(nz) else 0.0,
        "avg_daily_rev": float(cal["actual"].mean()),
    }


def run_monitor(fcst: pd.DataFrame, catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (signals, metrics_daily)."""
    signals, metrics_rows = [], []
    price = catalog.set_index("sku")["price"].to_dict()

    for sku, g in fcst.groupby("sku", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        base = _calibration_baseline(g)
        mon = g[g["window"] == "monitoring"].reset_index(drop=True)
        a, f = mon["actual"].values, mon["forecast"].values
        rel_err = (a - f) / np.maximum(f, 1.0)

        wape_tol = max(base["wape"] * WAPE_FACTOR, base["wape"] + 0.10)
        # Page-Hinkley runs on STANDARDIZED errors: delta/lambda are in sigma
        # units, so thresholds transfer across SKUs of any volume.
        z_err = (rel_err - base["bias"]) / base["rel_err_std"]
        ph_delta, ph_lam = 0.2, 40.0
        psi_floor = PSI_BASE + psi_null_expectation(len(base["rel_err"]), WINDOW)

        fired: dict[str, dict] = {}
        streak: dict[str, int] = {}  # consecutive checkpoints a check has fired
        PERSIST = {k: PERSIST_MODEL for k in ("accuracy", "bias", "psi", "ks")}

        for cp in range(WINDOW, len(mon), CHECKPOINT_EVERY):
            lo = cp - WINDOW
            aw, fw, rw = a[lo:cp], f[lo:cp], rel_err[lo:cp]
            date = mon["date"].iloc[cp - 1]

            w = wape(aw, fw)
            bias = float(np.mean(rw))
            metrics_rows.append(
                {"sku": sku, "date": date, "wape_28": w, "bias_28": bias,
                 "wape_baseline": base["wape"], "wape_tolerance": wape_tol}
            )

            checks: list[tuple[str, float, float, int]] = []
            # -- data-quality (critical) --------------------------------
            med_nz = np.median(aw[aw > 0]) if (aw > 0).any() else 0.0
            ratio = med_nz / base["median_nz"] if base["median_nz"] > 0 else 1.0
            if ratio > 4.0 or (0 < ratio < 0.25):
                checks.append(("scale", ratio, 4.0, CRITICAL))
            if _max_zero_run(aw) >= 7 and base["zero_rate"] < 0.05:
                checks.append(("zeros", float(_max_zero_run(aw)), 7, CRITICAL))
            if (
                _max_run(aw[aw > 0]) >= 10
                and base["max_run"] <= 5
                and np.std(aw[-14:]) < 1e-9
            ):
                checks.append(("frozen", float(_max_run(aw)), 10, CRITICAL))
            # -- model-health (alert) -----------------------------------
            if w > wape_tol:
                checks.append(("accuracy", w, wape_tol, ALERT))
            # bias is measured RELATIVE to the SKU's own calibration bias --
            # skewed low-volume SKUs carry a benign baseline offset.
            bias_delta = bias - base["bias"]
            bias_tol = max(BIAS_FLOOR, 4 * base["rel_err_std"] / np.sqrt(WINDOW))
            if abs(bias_delta) > bias_tol:
                checks.append(("bias", bias_delta, bias_tol, ALERT))
            p = psi(base["rel_err"], rw)
            if p > psi_floor:
                checks.append(("psi", p, psi_floor, WATCH))
            ks_p = stats.ks_2samp(base["rel_err"], rw).pvalue
            if ks_p < KS_P:
                checks.append(("ks", ks_p, KS_P, WATCH))
            if page_hinkley(z_err[:cp], ph_delta, ph_lam) is not None:
                checks.append(("page_hinkley", 1.0, 1.0, ALERT))

            fired_now = {c[0] for c in checks}
            for name in list(streak):
                if name not in fired_now:
                    streak[name] = 0  # broken streak resets the debounce
            for name, value, threshold, sev in checks:
                streak[name] = streak.get(name, 0) + 1
                if name not in fired and streak[name] >= PERSIST.get(name, 1):
                    fired[name] = {
                        "sku": sku, "check": name, "date": date,
                        "value": value, "threshold": threshold, "severity": sev,
                        "action": ACTIONS[name],
                    }

        signals.extend(fired.values())

    return pd.DataFrame(signals), pd.DataFrame(metrics_rows)
