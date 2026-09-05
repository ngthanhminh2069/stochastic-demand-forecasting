"""Feature engineering for a single SKU x Location leaf series.

Applied per-leaf (not pooled across leaves) so lag/rolling features never
leak demand from a different SKU or location into a leaf's own history.
"""
from __future__ import annotations

import pandas as pd

from src import config


def build_features(series: pd.Series, extra_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build lag/rolling/calendar features for one (sku, location) series.

    Parameters
    ----------
    series:
        Daily demand for a single (sku, location) pair, indexed by date.
    extra_daily:
        Optional extra daily columns (e.g. promotion, price) for this same
        (sku, location), indexed by date, to join in as-is.
    """
    df = pd.DataFrame({config.TARGET_COL: series})
    if extra_daily is not None:
        df = df.join(extra_daily)

    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["weekofyear"] = df.index.isocalendar().week.astype(int)

    for lag in config.LAG_DAYS:
        df[f"lag_{lag}"] = df[config.TARGET_COL].shift(lag)

    for window in config.ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = df[config.TARGET_COL].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df[config.TARGET_COL].shift(1).rolling(window).std()

    return df.dropna()


def split_features_target(df: pd.DataFrame, split_point):
    """Same semantics as proj1: split_point is an int (trailing rows) or a
    date-like cutoff."""
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]

    if isinstance(split_point, int):
        X_train, X_val = X.iloc[:-split_point], X.iloc[-split_point:]
        y_train, y_val = y.iloc[:-split_point], y.iloc[-split_point:]
    else:
        X_train, X_val = X[X.index < split_point], X[X.index >= split_point]
        y_train, y_val = y[y.index < split_point], y[y.index >= split_point]

    return X_train, X_val, y_train, y_val
