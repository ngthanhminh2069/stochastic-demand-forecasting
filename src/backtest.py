"""Walk-forward (expanding-window) backtest — same design as the sister
`tabpfn-forecast-benchmark` project's `backtest.py`, duplicated here so
this project stays standalone. Applied per (SKU, Location) leaf.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config


@dataclass
class Fold:
    fold_id: int
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def generate_folds(
    series: pd.Series,
    horizon_days: int = config.HORIZON_DAYS,
    step_days: int = config.STEP_DAYS,
    min_train_days: int = config.MIN_TRAIN_DAYS,
) -> list[Fold]:
    start_date = series.index.min()
    end_date = series.index.max()

    folds: list[Fold] = []
    train_end = start_date + pd.Timedelta(days=min_train_days)
    fold_id = 0

    while train_end + pd.Timedelta(days=horizon_days) <= end_date + pd.Timedelta(days=1):
        val_start = train_end
        val_end = train_end + pd.Timedelta(days=horizon_days)
        folds.append(Fold(fold_id=fold_id, train_end=train_end, val_start=val_start, val_end=val_end))
        fold_id += 1
        train_end = train_end + pd.Timedelta(days=step_days)

    return folds


def split_fold(series: pd.Series, fold: Fold) -> tuple[pd.Series, pd.Series]:
    train = series[series.index < fold.train_end]
    val = series[(series.index >= fold.val_start) & (series.index < fold.val_end)]
    return train, val
