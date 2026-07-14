"""Per-SKU baseline forecaster.

Deliberately simple (ridge regression on calendar + promo features in log
space): the point of SKU Sentinel is not the forecaster, it is the governance
layer that watches ANY forecaster. The model is fit once on the training
window and then frozen -- exactly how production forecast models drift in the
wild between retrains.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .simulate import TRAIN_END, CALIB_END


def _features(dates: pd.DatetimeIndex, promo: np.ndarray, t0: int = 0) -> np.ndarray:
    t = np.arange(t0, t0 + len(dates))
    dow = pd.get_dummies(dates.dayofweek).reindex(columns=range(7), fill_value=0).values
    month = pd.get_dummies(dates.month).reindex(columns=range(1, 13), fill_value=0).values
    annual = np.column_stack(
        [np.sin(2 * np.pi * t / 365.25), np.cos(2 * np.pi * t / 365.25)]
    )
    return np.column_stack([dow, month, annual, t / 365.0, promo.astype(float)])


def fit_and_forecast(panel: pd.DataFrame) -> pd.DataFrame:
    """Fit on [0, TRAIN_END) per SKU; forecast [TRAIN_END, end] with the frozen model."""
    out = []
    for sku, g in panel.groupby("sku", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        dates = pd.DatetimeIndex(g["date"])
        X = _features(dates, g["promo"].values)
        y = np.log1p(g["units"].values)

        model = Ridge(alpha=2.0)
        model.fit(X[:TRAIN_END], y[:TRAIN_END])
        pred = np.expm1(model.predict(X[TRAIN_END:]))
        pred = np.clip(pred, 0, None)

        out.append(
            pd.DataFrame(
                {
                    "sku": sku,
                    "date": dates[TRAIN_END:],
                    "forecast": pred,
                    "actual": g["units"].values[TRAIN_END:],
                    "promo": g["promo"].values[TRAIN_END:],
                }
            )
        )
    fcst = pd.concat(out, ignore_index=True)
    split_date = pd.DatetimeIndex(panel["date"].unique()).sort_values()[CALIB_END]
    fcst["window"] = np.where(fcst["date"] < split_date, "calibration", "monitoring")
    return fcst
