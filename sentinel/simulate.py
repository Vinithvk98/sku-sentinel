"""Synthetic retail demand generator with ground-truth drift injection.

Generates a daily demand panel for a plausible retailer (real-looking product
names across grocery/household categories). The first ~15 months are clean
(model training + threshold calibration). Drift and data-quality events are
injected only into the final ~6-month monitoring window, with an exact
ground-truth log so detection quality can be scored objectively.

Event taxonomy (mirrors what breaks real SKU forecasting systems):
    level_shift        step change in true demand (competitor exit, delisting)
    gradual_drift      slow erosion/growth over ~90 days (preference shift)
    seasonality_change weekly pattern rotates (channel mix change)
    promo_shift        promo response collapses (promo fatigue)
    frozen_values      data feed stuck on a constant value (stale extract)
    scale_change       unit/currency bug: values x10 or /10 (ETL break)
    missing_data       feed outage recorded as zeros

All knobs live in config.yaml (see sentinel/config.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG

N_SKUS = int(CFG["simulation"]["n_skus"])
DAYS = int(CFG["simulation"]["days"])
TRAIN_END = int(DAYS * 0.623)   # model fit window        [0, TRAIN_END)
CALIB_END = int(DAYS * 0.747)   # threshold calibration   [TRAIN_END, CALIB_END)
START_DATE = "2024-07-09"       # monitoring window ends near "today"

EVENTS_PER_TYPE = int(CFG["simulation"]["events_per_type"])
EVENT_TYPES = [
    "level_shift",
    "gradual_drift",
    "seasonality_change",
    "promo_shift",
    "frozen_values",
    "scale_change",
    "missing_data",
]

# ---------------------------------------------------------------- catalog --
PRODUCTS = {
    "Beverages": (
        ["Arabica Coffee", "Green Tea", "Cola", "Orange Juice", "Energy Drink",
         "Sparkling Water", "Lemon Iced Tea", "Oat Drink", "Hot Chocolate"],
        ["250ml", "330ml", "500ml", "1L", "6-pack"],
    ),
    "Snacks": (
        ["Potato Chips", "Salted Peanuts", "Dark Chocolate Bar", "Granola Bar",
         "Pretzels", "Trail Mix", "Rice Crackers", "Popcorn"],
        ["80g", "150g", "200g", "300g"],
    ),
    "Dairy": (
        ["Whole Milk", "Greek Yogurt", "Cheddar Cheese", "Butter",
         "Cream Cheese", "Mozzarella", "Vanilla Yogurt"],
        ["200g", "250g", "500g", "1L"],
    ),
    "Bakery": (
        ["Sourdough Loaf", "Croissant 4-pack", "Bagels", "Tortilla Wraps",
         "Blueberry Muffins", "Rye Bread"],
        ["300g", "400g", "500g"],
    ),
    "Household": (
        ["Laundry Detergent", "Dish Soap", "Paper Towels", "Trash Bags",
         "All-Purpose Cleaner", "Sponges 5-pack", "Aluminium Foil"],
        ["500ml", "1L", "2L", "12-roll"],
    ),
    "Personal Care": (
        ["Shampoo", "Toothpaste", "Hand Soap", "Body Lotion", "Deodorant",
         "Razor Blades 4-pack", "Cotton Pads"],
        ["75ml", "250ml", "400ml"],
    ),
    "Frozen": (
        ["Margherita Pizza", "Vanilla Ice Cream", "Mixed Vegetables",
         "Fish Fingers", "Berry Mix", "Garlic Bread"],
        ["300g", "450g", "750g", "1kg"],
    ),
    "Pantry": (
        ["Spaghetti", "Basmati Rice", "Tomato Passata", "Olive Oil",
         "Peanut Butter", "Honey", "Canned Chickpeas", "Oat Flakes"],
        ["250g", "400g", "500g", "750ml", "1kg"],
    ),
}


def build_product_names(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """n unique, plausible (category, product name) pairs."""
    combos = [
        (cat, prod if "-pack" in prod else f"{prod} {size}")
        for cat, (prods, sizes) in PRODUCTS.items()
        for prod in prods
        for size in sizes
    ]
    combos = list(dict.fromkeys(combos))  # de-dupe pack items across sizes
    idx = rng.choice(len(combos), size=min(n, len(combos)), replace=False)
    rows = [combos[i] for i in idx]
    while len(rows) < n:  # top up with numbered variants if n is very large
        cat, name = rows[len(rows) % len(idx)]
        rows.append((cat, f"{name} v{len(rows)}"))
    return pd.DataFrame(rows, columns=["category", "name"])


def _weekly_profile(rng: np.random.Generator) -> np.ndarray:
    return rng.dirichlet(np.ones(7) * 8) * 7.0


def build_catalog(rng: np.random.Generator) -> pd.DataFrame:
    names = build_product_names(rng, N_SKUS)
    rows = []
    for i in range(N_SKUS):
        rows.append(
            {
                "sku": f"SKU-{i + 1:04d}",
                "name": names.iloc[i]["name"],
                "category": names.iloc[i]["category"],
                "base": float(np.exp(rng.normal(3.0, 0.9))),
                "price": float(np.round(np.exp(rng.normal(2.5, 0.7)), 2)),
                "promo_lift": float(rng.uniform(1.4, 2.5)),
                "annual_amp": float(rng.uniform(0.05, 0.30)),
                "annual_phase": float(rng.uniform(0, 2 * np.pi)),
                "trend": float(rng.normal(0.00005, 0.0001)),
            }
        )
    return pd.DataFrame(rows)


def build_promo_calendar(rng: np.random.Generator, catalog: pd.DataFrame) -> np.ndarray:
    """Promos are CONFOUNDED with season, like in real retail: merchants promote
    into peak demand. This makes naive promo-vs-normal comparisons overstate
    uplift, the whole reason regression-adjusted estimation exists."""
    promo = np.zeros((N_SKUS, DAYS), dtype=bool)
    t = np.arange(0, DAYS - 7, 7)  # candidate week starts
    for i in range(N_SKUS):
        if rng.random() < 0.6:
            c = catalog.iloc[i]
            seasonal = 1 + c["annual_amp"] * np.sin(2 * np.pi * t / 365.25 + c["annual_phase"])
            w = np.exp(4.0 * (seasonal - seasonal.min()) / (np.ptp(seasonal) + 1e-9))
            w = w / w.sum()
            n = int(rng.integers(6, 14))
            starts = rng.choice(t, size=n, replace=False, p=w)
            for s in starts:
                promo[i, s : s + 7] = True
    return promo


def assign_events(rng: np.random.Generator) -> list[dict]:
    skus = rng.choice(N_SKUS, size=EVENTS_PER_TYPE * len(EVENT_TYPES), replace=False)
    events, k = [], 0
    for etype in EVENT_TYPES:
        for _ in range(EVENTS_PER_TYPE):
            i = int(skus[k])
            k += 1
            t0 = int(rng.integers(CALIB_END + 14, DAYS - 60))
            ev: dict = {"sku_idx": i, "type": etype, "t0": t0}
            if etype == "level_shift":
                ev["magnitude"] = float(rng.choice([-1, 1]) * rng.uniform(0.35, 0.60))
            elif etype == "gradual_drift":
                ev["magnitude"] = float(rng.choice([-1, 1]) * rng.uniform(0.40, 0.65))
                ev["ramp_days"] = 90
            elif etype == "scale_change":
                ev["magnitude"] = float(rng.choice([10.0, 0.1]))
            elif etype == "frozen_values":
                ev["duration"] = int(rng.integers(14, 30))
            elif etype == "missing_data":
                ev["duration"] = int(rng.integers(10, 21))
            events.append(ev)
    return events


def generate(seed: int | None = None):
    """Returns (panel, catalog, ground_truth) dataframes."""
    if seed is None:
        seed = int(CFG["simulation"]["seed"])
    rng = np.random.default_rng(seed)
    catalog = build_catalog(rng)
    weekly = np.stack([_weekly_profile(rng) for _ in range(N_SKUS)])
    promo = build_promo_calendar(rng, catalog)
    events = assign_events(rng)

    # promo_shift needs a real promo cadence in the monitoring window
    for ev in events:
        if ev["type"] == "promo_shift":
            i = ev["sku_idx"]
            promo[i] = False
            for s in range(7, DAYS - 7, 28):
                promo[i, s : s + 7] = True
            catalog.loc[i, "promo_lift"] = float(rng.uniform(1.9, 2.6))

    dates = pd.date_range(START_DATE, periods=DAYS, freq="D")
    dow = dates.dayofweek.values
    t = np.arange(DAYS)

    units = np.zeros((N_SKUS, DAYS))
    for i in range(N_SKUS):
        c = catalog.iloc[i]
        mu = (
            c["base"]
            * weekly[i, dow]
            * (1 + c["annual_amp"] * np.sin(2 * np.pi * t / 365.25 + c["annual_phase"]))
            * np.exp(c["trend"] * t)
        )
        lift = np.where(promo[i], c["promo_lift"], 1.0)

        for ev in events:  # demand-side events
            if ev["sku_idx"] != i:
                continue
            t0 = ev["t0"]
            if ev["type"] == "level_shift":
                mu[t0:] = mu[t0:] * (1 + ev["magnitude"])
            elif ev["type"] == "gradual_drift":
                ramp = np.clip((t - t0) / ev["ramp_days"], 0, 1)
                mu = mu * (1 + ev["magnitude"] * ramp)
            elif ev["type"] == "seasonality_change":
                new_w = rng.dirichlet(np.ones(7) * 2) * 7.0  # sharper new pattern
                mu[t0:] = mu[t0:] / weekly[i, dow[t0:]] * new_w[dow[t0:]]
            elif ev["type"] == "promo_shift":
                lift = lift.copy()
                lift[t0:] = 1.0

        noise = rng.gamma(shape=12.0, scale=1 / 12.0, size=DAYS)
        y = rng.poisson(mu * lift * noise).astype(float)

        for ev in events:  # data-side corruption of recorded actuals
            if ev["sku_idx"] != i:
                continue
            t0 = ev["t0"]
            if ev["type"] == "frozen_values":
                y[t0 : t0 + ev["duration"]] = max(float(np.round(mu[t0])), 2.0)
            elif ev["type"] == "missing_data":
                y[t0 : t0 + ev["duration"]] = 0.0
            elif ev["type"] == "scale_change":
                y[t0:] = np.round(y[t0:] * ev["magnitude"])
        units[i] = y

    panel = pd.DataFrame(
        {
            "date": np.tile(dates, N_SKUS),
            "sku": np.repeat(catalog["sku"].values, DAYS),
            "units": units.reshape(-1),
            "promo": promo.reshape(-1).astype(int),
        }
    )
    panel = panel.merge(catalog[["sku", "price"]], on="sku")
    panel["revenue"] = panel["units"] * panel["price"]

    gt = pd.DataFrame(
        [
            {
                "sku": catalog.iloc[ev["sku_idx"]]["sku"],
                "name": catalog.iloc[ev["sku_idx"]]["name"],
                "event_type": ev["type"],
                "event_date": dates[ev["t0"]],
                "detail": {
                    k: v for k, v in ev.items() if k not in ("sku_idx", "type", "t0")
                },
            }
            for ev in events
        ]
    )
    return panel, catalog, gt


if __name__ == "__main__":
    panel, catalog, gt = generate()
    print(panel.shape, gt.shape)
    print(catalog[["sku", "name", "category"]].head(8).to_string(index=False))
