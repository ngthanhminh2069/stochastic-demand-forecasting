"""Probabilistic forecasting: XGBoost quantile regression, calibrated with
split conformal prediction.

Point forecasts (a single number) can't drive safety-stock decisions —
supply-chain planning needs a *distribution* (or at least P10/P50/P90) to
answer "how much buffer do I need to hit a 95% service level?". Raw
quantile regression is a good starting point but its quantiles aren't
statistically guaranteed to have correct coverage; split conformal
prediction adds a calibration step on a held-out set to fix that,
regardless of whether the underlying quantile model is well-calibrated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src import config


def fit_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantiles: list[float] = config.QUANTILES,
    n_jobs: int | None = None,
) -> dict[float, xgb.XGBRegressor]:
    """Fit one XGBRegressor per quantile using the pinball (quantile) loss.

    `n_jobs`: threads per XGBoost model. Leave as None to let XGBoost use
    all available cores (fine for sequential/single-leaf runs). Pass 1
    when leaf-level parallelism (multiple leaves training at once via
    `pipeline.run_all_leaves(..., n_jobs=N)`) is already using the CPU
    cores — otherwise every process fights every other process for the
    same cores and things get *slower*, not faster.
    """
    params = dict(config.XGB_QUANTILE_PARAMS)
    if n_jobs is not None:
        params["n_jobs"] = n_jobs

    models = {}
    for q in quantiles:
        model = xgb.XGBRegressor(
            **params,
            objective="reg:quantileerror",
            quantile_alpha=q,
        )
        model.fit(X_train, y_train)
        models[q] = model
    return models


def predict_quantiles(
    models: dict[float, xgb.XGBRegressor], X: pd.DataFrame
) -> pd.DataFrame:
    """Predict all quantiles and return a DataFrame indexed like X, one
    column per quantile, with columns sorted and monotonicity enforced
    (P10 <= P50 <= P90) via a simple sort across the quantile axis."""
    preds = {q: model.predict(X) for q, model in models.items()}
    df = pd.DataFrame(preds, index=X.index)
    # Enforce monotonicity: a low-quantile model can occasionally predict
    # above a higher-quantile model's output on a given row. Sorting each
    # row's predicted values back into quantile order is the standard fix.
    sorted_cols = sorted(df.columns)
    values_sorted = np.sort(df[sorted_cols].values, axis=1)
    return pd.DataFrame(values_sorted, index=df.index, columns=sorted_cols)


def conformal_calibrate(
    models: dict[float, xgb.XGBRegressor],
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    quantiles: list[float] = config.QUANTILES,
) -> dict[float, float]:
    """Split conformal calibration: compute the additive correction for each
    quantile so that, on the calibration set, coverage matches the nominal
    quantile level.

    Returns a dict {quantile: offset} to add to that quantile's raw
    predictions at inference time.
    """
    offsets = {}
    for q in quantiles:
        raw_preds = models[q].predict(X_calib)
        residuals = y_calib.values - raw_preds
        # For the median, center the correction; for tail quantiles, shift
        # to the residual quantile matching q so nominal coverage holds.
        offsets[q] = float(np.quantile(residuals, q))
    return offsets


def apply_conformal_offsets(quantile_preds: pd.DataFrame, offsets: dict[float, float]) -> pd.DataFrame:
    """Apply calibration offsets and re-sort to keep quantiles monotonic."""
    adjusted = quantile_preds.copy()
    for q in adjusted.columns:
        adjusted[q] = adjusted[q] + offsets.get(q, 0.0)
    values_sorted = np.sort(adjusted.values, axis=1)
    return pd.DataFrame(values_sorted, index=adjusted.index, columns=sorted(adjusted.columns))


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, quantile: float) -> float:
    """Pinball (quantile) loss — the correct scoring rule for a single
    quantile forecast, asymmetric by construction."""
    diff = y_true.values - y_pred.values
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))
