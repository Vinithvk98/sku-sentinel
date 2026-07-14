"""The model bench: 8 forecasting models across 4 families, one fair fight.

Families:
  baseline     seasonal_naive, the bar every model must clear
  statistical  ridge, holt_winters
  ml_ensemble  random_forest, hist_gb, xgboost, lightgbm
  deep         lstm (tiny PyTorch network)

Rules of the fight (this is what makes it honest):
  - every model trains ONLY on the training window
  - every model is scored on the SAME untouched holdout (the calibration window)
  - same metric (WAPE), same features where applicable
  - fit+predict time is reported, accuracy per second matters in production

Heavy libraries (xgboost, lightgbm, torch) are optional: if not installed,
the model reports itself unavailable instead of crashing the app.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .forecast import _features
from .monitor import wape

# ---------------------------------------------------------- optional deps --
# We record WHY an import failed, not just that it failed, on macOS, for
# example, xgboost/lightgbm install fine but fail to import until
# `brew install libomp` provides the OpenMP runtime.
IMPORT_ERRORS: dict[str, str] = {}

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception as exc:
    HAS_XGB = False
    IMPORT_ERRORS["xgboost"] = str(exc)

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception as exc:
    HAS_LGBM = False
    IMPORT_ERRORS["lightgbm"] = str(exc)

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except Exception as exc:
    HAS_TORCH = False
    IMPORT_ERRORS["lstm"] = str(exc)

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_SM = True
except Exception as exc:
    HAS_SM = False
    IMPORT_ERRORS["holt_winters"] = str(exc)


REGISTRY = {
    "seasonal_naive": {
        "label": "Seasonal Naive", "family": "baseline", "available": True,
        "blurb": "Repeats each weekday's recent average. Free, instant, and the bar every real model must clear.",
    },
    "ridge": {
        "label": "Ridge Regression", "family": "statistical", "available": True,
        "blurb": "Linear model on calendar + promo features in log space. Simple, fast, interpretable.",
    },
    "holt_winters": {
        "label": "Holt-Winters", "family": "statistical", "available": HAS_SM,
        "blurb": "Classic exponential smoothing with weekly seasonality. 1960s math that still wins fights.",
    },
    "random_forest": {
        "label": "Random Forest", "family": "ml_ensemble", "available": True,
        "blurb": "Bagged decision trees on the same features. Robust, but can't extrapolate trend.",
    },
    "hist_gb": {
        "label": "Gradient Boosting", "family": "ml_ensemble", "available": True,
        "blurb": "scikit-learn's histogram gradient boosting. Usually the strongest tabular baseline.",
    },
    "xgboost": {
        "label": "XGBoost", "family": "ml_ensemble", "available": HAS_XGB,
        "blurb": "The Kaggle-famous gradient boosting library.",
    },
    "lightgbm": {
        "label": "LightGBM", "family": "ml_ensemble", "available": HAS_LGBM,
        "blurb": "Microsoft's fast gradient boosting. Leaf-wise growth, tiny memory.",
    },
    "lstm": {
        "label": "LSTM (PyTorch)", "family": "deep", "available": HAS_TORCH,
        "blurb": "A small recurrent neural net reading the last 28 days. Deep learning, often beaten by boosting on clean seasonal demand, which is itself a lesson.",
    },
}


# ------------------------------------------------------------ the models --
def _fit_sklearn(model, X_tr, y_tr, X_ev):
    model.fit(X_tr, np.log1p(y_tr))
    return np.clip(np.expm1(model.predict(X_ev)), 0, None)


def _seasonal_naive(y_tr: np.ndarray, dow_tr: np.ndarray, dow_ev: np.ndarray) -> np.ndarray:
    recent = y_tr[-56:]
    rdow = dow_tr[-56:]
    dow_mean = np.array([
        recent[rdow == d].mean() if (rdow == d).any() else recent.mean()
        for d in range(7)
    ])
    return dow_mean[dow_ev]


def _holt_winters(y_tr: np.ndarray, horizon: int) -> np.ndarray:
    fit = ExponentialSmoothing(
        y_tr + 1e-3, trend="add", seasonal="add", seasonal_periods=7,
        initialization_method="estimated",
    ).fit(optimized=True)
    return np.clip(fit.forecast(horizon), 0, None)


def _lstm(y_tr: np.ndarray, horizon: int, lookback: int = 28,
          epochs: int = 10, hidden: int = 32, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    z = np.log1p(y_tr)
    mu, sd = z.mean(), z.std() + 1e-9
    z = (z - mu) / sd

    X, Y = [], []
    for i in range(len(z) - lookback):
        X.append(z[i : i + lookback])
        Y.append(z[i + lookback])
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1]).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(X), Y)
        loss.backward()
        opt.step()

    # recursive multi-step forecast: feed predictions back in
    window = list(z[-lookback:])
    preds = []
    net.eval()
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(window[-lookback:], dtype=torch.float32).view(1, lookback, 1)
            nxt = float(net(x))
            preds.append(nxt)
            window.append(nxt)
    return np.clip(np.expm1(np.array(preds) * sd + mu), 0, None)


# -------------------------------------------------------------- the bench --
def run_bench(g: pd.DataFrame, model_names: list[str],
              train_end: int, eval_end: int) -> dict:
    """One SKU, many models, one holdout.

    g: one SKU's history (date, units, promo), day-indexed rows 0..n.
    Trains on [0, train_end), scores on [train_end, eval_end), the clean
    calibration window, so injected drift events never contaminate the fight.
    """
    g = g.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(g["date"])
    y = g["units"].values.astype(float)
    promo = g["promo"].values
    X = _features(dates, promo)
    dow = dates.dayofweek.values

    y_tr, X_tr = y[:train_end], X[:train_end]
    y_ev, X_ev = y[train_end:eval_end], X[train_end:eval_end]
    horizon = eval_end - train_end

    results = {}
    for name in model_names:
        spec = REGISTRY.get(name)
        if spec is None:
            continue
        if not spec["available"]:
            results[name] = {**spec, "error": "library not installed"}
            continue
        t0 = time.time()
        try:
            if name == "seasonal_naive":
                pred = _seasonal_naive(y_tr, dow[:train_end], dow[train_end:eval_end])
            elif name == "ridge":
                from sklearn.linear_model import Ridge
                pred = _fit_sklearn(Ridge(alpha=2.0), X_tr, y_tr, X_ev)
            elif name == "holt_winters":
                pred = _holt_winters(y_tr, horizon)
            elif name == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                pred = _fit_sklearn(
                    RandomForestRegressor(n_estimators=150, min_samples_leaf=3,
                                          n_jobs=-1, random_state=0),
                    X_tr, y_tr, X_ev)
            elif name == "hist_gb":
                from sklearn.ensemble import HistGradientBoostingRegressor
                pred = _fit_sklearn(HistGradientBoostingRegressor(random_state=0),
                                    X_tr, y_tr, X_ev)
            elif name == "xgboost":
                pred = _fit_sklearn(
                    XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.9, verbosity=0, random_state=0),
                    X_tr, y_tr, X_ev)
            elif name == "lightgbm":
                pred = _fit_sklearn(
                    LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  subsample=0.9, verbose=-1, random_state=0),
                    X_tr, y_tr, X_ev)
            elif name == "lstm":
                pred = _lstm(y_tr, horizon)
            else:
                continue
        except Exception as exc:
            results[name] = {**spec, "error": str(exc)}
            continue

        err = pred - y_ev
        results[name] = {
            **spec,
            "error": None,
            "wape": round(wape(y_ev, pred), 4),
            "mae": round(float(np.abs(err).mean()), 2),
            "bias": round(float(err.sum() / max(y_ev.sum(), 1e-9)), 4),
            "seconds": round(time.time() - t0, 2),
            "forecast": np.round(pred, 2).tolist(),
        }

    scored = {k: v for k, v in results.items() if v.get("wape") is not None}
    champion = min(scored, key=lambda k: scored[k]["wape"]) if scored else None
    return {
        "dates": dates[train_end:eval_end].strftime("%Y-%m-%d").tolist(),
        "actual": np.round(y_ev, 1).tolist(),
        "models": results,
        "champion": champion,
    }
