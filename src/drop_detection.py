"""CUSUM (cumulative sum) control-chart based demand-drop detection.

The previous version of this project flagged a "drop day" with a fixed
`demand < 70% of mean` threshold — arbitrary, and blind to gradual decline
(a slow erosion that never crosses one day's threshold but adds up to a
real problem). CUSUM instead accumulates *small* negative deviations from
a reference mean and raises an alarm once the cumulative deviation exceeds
a statistically motivated threshold, which is the standard tool for
detecting persistent shifts in a process — including gradual ones — not
just single-point outliers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


@dataclass
class CusumResult:
    cusum_series: pd.Series          # cumulative negative-deviation statistic
    alarm_days: pd.DatetimeIndex     # dates where the alarm threshold was breached
    reference_mean: float
    reference_std: float
    threshold: float


def detect_drops(
    series: pd.Series,
    reference_window: int = 60,
    threshold_std: float = config.CUSUM_THRESHOLD_STD,
    slack_std: float = config.CUSUM_SLACK_STD,
) -> CusumResult:
    """Run a one-sided (downward) CUSUM to flag sustained demand drops.

    Parameters
    ----------
    series:
        Demand series to monitor, indexed by date.
    reference_window:
        Number of leading observations used to establish the reference
        mean/std (the "in-control" baseline) before monitoring begins.
    threshold_std:
        Alarm threshold, in standard deviations of the reference period.
    slack_std:
        Slack (k) parameter, in standard deviations — deviations smaller
        than this are not accumulated, which filters out routine noise so
        CUSUM responds to sustained shifts rather than single-day dips.
    """
    if len(series) <= reference_window:
        raise ValueError(
            f"Series has {len(series)} points; need more than reference_window={reference_window}."
        )

    reference = series.iloc[:reference_window]
    ref_mean = reference.mean()
    ref_std = reference.std() if reference.std() > 0 else 1.0
    threshold = threshold_std * ref_std
    slack = slack_std * ref_std

    monitored = series.iloc[reference_window:]
    deviations = monitored - ref_mean

    cusum_values = np.zeros(len(monitored))
    running = 0.0
    for i, dev in enumerate(deviations.values):
        # One-sided lower CUSUM: accumulates negative deviations beyond slack.
        running = min(0.0, running + dev + slack)
        cusum_values[i] = running
        if running <= -threshold:
            # Reset after an alarm so CUSUM detects distinct drop *events*
            # rather than continuing to alarm on every subsequent day.
            running = 0.0

    cusum_series = pd.Series(cusum_values, index=monitored.index)
    alarm_days = cusum_series[cusum_series <= -threshold].index

    return CusumResult(
        cusum_series=cusum_series,
        alarm_days=alarm_days,
        reference_mean=ref_mean,
        reference_std=ref_std,
        threshold=threshold,
    )


def summarize_alarms(result: CusumResult) -> pd.DataFrame:
    """One row per alarm day, with how far below the reference mean it was."""
    if len(result.alarm_days) == 0:
        return pd.DataFrame(columns=["date", "cusum_value", "std_below_reference"])
    rows = [
        {
            "date": d,
            "cusum_value": result.cusum_series.loc[d],
            "std_below_reference": abs(result.cusum_series.loc[d]) / result.reference_std,
        }
        for d in result.alarm_days
    ]
    return pd.DataFrame(rows)
