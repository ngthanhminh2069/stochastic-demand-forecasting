"""Probabilistic forecasting: XGBoost quantile regression, calibrated with
split conformal prediction.

Point forecasts (a single number) can't drive safety-stock decisions -
supply-chain planning needs a *distribution* to answer "how much buffer do
I need to hit a 95% service level?". Raw quantile regression is a good
starting point but its quantiles aren't statistically guaranteed to have
correct coverage; split conformal prediction adds a calibration step on a
held-out set to fix that, regardless of whether the underlying quantile
model is well-calibrated.

Design notes
------------
* One XGBoost model is fit per leaf and fold, with **all quantiles in a
  single multi-quantile model** (``quantile_alpha`` as an array), and the
  horizon ``h`` is a feature. That is the "direct multi-horizon" setup:
  rows are (origin, h) pairs built by ``features.make_direct_frame``.
* Conformal offsets are estimated **per horizon bucket** (spread grows with
  h) from residuals that the caller may pool across several leaves (the
  pipeline pools across the locations of one SKU).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src import config


def fit_quantile_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    quantiles: list[float] = config.QUANTILES,
    n_jobs: int | None = None,
) -> xgb.XGBRegressor:
    """Fit one multi-quantile XGBRegressor (pinball loss for every level).

    `n_jobs`: threads for XGBoost. Leave as None to let it use all cores
    (fine for sequential runs). Pass 1 when SKU-level process parallelism
    (``pipeline.run_all_leaves(..., n_jobs=N)``) already uses the cores -
    otherwise processes fight each other and things get slower.
    """
    params = dict(config.XGB_QUANTILE_PARAMS)
    if n_jobs is not None:
        params["n_jobs"] = n_jobs

    model = xgb.XGBRegressor(
        **params,
        objective="reg:quantileerror",
        quantile_alpha=np.asarray(sorted(quantiles), dtype=float),
    )
    model.fit(X_train, np.asarray(y_train, dtype=float))
    return model


def predict_quantiles(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    quantiles: list[float] = config.QUANTILES,
) -> pd.DataFrame:
    """Predict all quantiles (one column per level, ascending) with
    monotonicity enforced (P5 <= P10 <= ...) via a row-wise sort, the
    standard fix for quantile crossing."""
    qs = sorted(quantiles)
    preds = np.asarray(model.predict(X), dtype=float).reshape(len(X), len(qs))
    return pd.DataFrame(np.sort(preds, axis=1), index=X.index, columns=qs)


def horizon_bucket(h: np.ndarray | pd.Series) -> np.ndarray:
    """Map horizons (days) to indices of ``config.HORIZON_BUCKETS``."""
    h = np.asarray(h)
    out = np.full(len(h), len(config.HORIZON_BUCKETS) - 1, dtype=int)
    for i, (lo, hi) in reversed(list(enumerate(config.HORIZON_BUCKETS))):
        out[(h >= lo) & (h <= hi)] = i
    return out


def calibration_residuals(
    raw_preds: pd.DataFrame, y: np.ndarray | pd.Series, h: np.ndarray | pd.Series
) -> pd.DataFrame:
    """Residuals ``y - raw_pred_q`` for every quantile column, plus ``h``.
    Kept as a DataFrame so callers can concatenate residuals from several
    leaves before fitting offsets."""
    res = raw_preds.sub(np.asarray(y, dtype=float), axis=0).mul(-1.0)
    res = res.reset_index(drop=True)
    res["h"] = np.asarray(h)
    return res


def conformal_offsets(
    residuals: pd.DataFrame, quantiles: list[float] = config.QUANTILES
) -> dict[int, dict[float, float]]:
    """Split-conformal additive correction, per horizon bucket and quantile.

    For quantile q the offset is the q-th empirical quantile of the
    calibration residuals ``y - pred_q``: if the raw model is already
    calibrated that is ~0; if it under-covers the offset is positive.
    Buckets with no calibration data fall back to the pooled offset.
    """
    bucket = horizon_bucket(residuals["h"].to_numpy())
    pooled = {q: float(np.quantile(residuals[q].to_numpy(), q)) for q in quantiles}
    offsets: dict[int, dict[float, float]] = {}
    for b in range(len(config.HORIZON_BUCKETS)):
        rows = residuals[bucket == b]
        if len(rows) == 0:
            offsets[b] = dict(pooled)
        else:
            offsets[b] = {q: float(np.quantile(rows[q].to_numpy(), q)) for q in quantiles}
    return offsets


def apply_conformal_offsets(
    quantile_preds: pd.DataFrame,
    h: np.ndarray | pd.Series,
    offsets: dict[int, dict[float, float]],
) -> pd.DataFrame:
    """Apply per-bucket calibration offsets, clip at zero (demand can't be
    negative) and re-sort each row to keep quantiles monotonic."""
    adjusted = quantile_preds.copy()
    bucket = horizon_bucket(np.asarray(h))
    for q in adjusted.columns:
        shift = np.array([offsets[b].get(q, 0.0) for b in bucket])
        adjusted[q] = adjusted[q].to_numpy() + shift
    values = np.sort(np.clip(adjusted.to_numpy(), 0.0, None), axis=1)
    return pd.DataFrame(values, index=adjusted.index, columns=sorted(adjusted.columns))


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Pinball (quantile) loss - the correct scoring rule for a single
    quantile forecast, asymmetric by construction."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))
