"""Feature engineering for a single SKU x Location leaf series — *direct
multi-horizon, as-of-origin*.

The earlier version built lag/rolling features once over the full series
and then cut train/validation by date. For a 14-day horizon that leaks:
`lag_7` for day 10 of the horizon is demand from day 3 of the *same*
horizon, which does not exist when the forecast is actually made.

Here every row is a ``(origin, horizon h)`` pair:

* ``origin`` is the last day whose demand is known (the forecast is made
  at the end of that day);
* the target is the demand on ``origin + h`` days, ``h = 1..H``;
* every demand-derived feature is computed strictly from observations
  at or before the origin, so the training rows look exactly like what
  the model sees when it forecasts for real.

Applied per-leaf (not pooled across leaves) so features never mix demand
from a different SKU or location.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src import config


def feature_columns() -> list[str]:
    cols = ["h", "dayofweek", "month", "weekofyear", "last_obs"]
    cols += [f"seasonal_lag_{j}" for j in range(1, config.SEASONAL_LAG_WEEKS + 1)]
    for w in config.ROLLING_WINDOWS:
        cols += [f"rolling_mean_{w}", f"rolling_std_{w}"]
    cols += ["promotion", config.PRICE_COL]
    return cols


def min_origin_pos(horizons: np.ndarray | list[int] | None = None) -> int:
    """Earliest origin position (0-based) that has enough history for every
    feature at every horizon."""
    if horizons is None:
        horizons = range(1, config.HORIZON_DAYS + 1)
    reach = max(
        7 * (math.ceil(h / 7) + config.SEASONAL_LAG_WEEKS - 1) - h for h in horizons
    )
    return max(max(config.ROLLING_WINDOWS) - 1, reach)


def seasonal_naive_forecast(Y: np.ndarray, origin_pos: int, horizons: np.ndarray) -> np.ndarray:
    """Most recent same-weekday observation known at the origin. This is
    both the first seasonal lag feature and the benchmark forecast."""
    h = np.asarray(horizons)
    k = np.ceil(h / 7).astype(int)
    return Y[origin_pos + h - 7 * k]


def make_direct_frame(
    series: pd.Series,
    extra_daily: pd.DataFrame | None,
    origin_pos: np.ndarray | list[int],
    horizons: np.ndarray | list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the (origin, horizon) feature matrix.

    Parameters
    ----------
    series:
        Daily demand for one (sku, location), contiguous daily index.
    extra_daily:
        Daily ``promotion`` and ``price`` columns for the same leaf, indexed
        by date. These are treated as known in advance for the target day.
    origin_pos:
        Integer positions (into ``series``) of forecast origins.
    horizons:
        Horizons in days (1-based) to build for each origin.

    Returns
    -------
    X, meta:
        ``X`` has the model features; ``meta`` has ``origin_pos``,
        ``target_pos``, ``target_date``, ``h`` and ``y`` (actual target
        demand). Rows whose target falls past the end of the series are
        dropped.
    """
    Y = series.to_numpy(dtype=float)
    idx = series.index
    n = len(Y)

    origin_pos = np.asarray(origin_pos, dtype=int)
    horizons = np.asarray(horizons, dtype=int)
    o = np.repeat(origin_pos, len(horizons))
    h = np.tile(horizons, len(origin_pos))
    t = o + h
    keep = (t < n) & (o >= min_origin_pos(horizons))
    o, h, t = o[keep], h[keep], t[keep]

    k = np.ceil(h / 7).astype(int)
    target_dates = idx[t]

    feats: dict[str, np.ndarray] = {
        "h": h,
        "dayofweek": target_dates.dayofweek.to_numpy(),
        "month": target_dates.month.to_numpy(),
        "weekofyear": target_dates.isocalendar().week.to_numpy().astype(int),
        "last_obs": Y[o],
    }
    for j in range(1, config.SEASONAL_LAG_WEEKS + 1):
        feats[f"seasonal_lag_{j}"] = Y[t - 7 * (k + j - 1)]

    s = pd.Series(Y)
    for w in config.ROLLING_WINDOWS:
        # rolling window ends at the origin day *inclusive*: the origin's
        # demand is known when the forecast is made.
        feats[f"rolling_mean_{w}"] = s.rolling(w).mean().to_numpy()[o]
        feats[f"rolling_std_{w}"] = s.rolling(w).std().to_numpy()[o]

    if extra_daily is not None:
        extra = extra_daily.reindex(idx)
        feats["promotion"] = extra["promotion"].to_numpy(dtype=float)[t]
        feats[config.PRICE_COL] = extra[config.PRICE_COL].to_numpy(dtype=float)[t]
    else:
        feats["promotion"] = np.zeros(len(t))
        feats[config.PRICE_COL] = np.zeros(len(t))

    X = pd.DataFrame(feats)[feature_columns()]
    meta = pd.DataFrame(
        {"origin_pos": o, "target_pos": t, "target_date": target_dates, "h": h, "y": Y[t]}
    )
    return X.reset_index(drop=True), meta.reset_index(drop=True)
