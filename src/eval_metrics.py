"""Point-forecast metrics (MAE/RMSE/MAPE plus the planner-facing WAPE, MASE
and bias), re-exported alongside the pinball loss in `probabilistic.py` and
the cost-weighted metrics in `inventory.py` — kept separate because they
answer different questions (point accuracy vs. distributional calibration
vs. business cost).

* **WAPE** = sum|e| / sum(actual): volume-weighted, robust to zero-demand
  days (unlike MAPE), and additive across leaves.
* **MASE** = MAE / MAE of the in-sample seasonal-naive forecast: < 1 means
  better than the benchmark, scale-free across SKUs.
* **Bias** = sum(forecast - actual) / sum(actual): > 0 over-forecasts
  (excess stock), < 0 under-forecasts (stockout risk).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ForecastMetrics:
    mae: float
    rmse: float
    mape: float
    wape: float = float("nan")
    bias: float = float("nan")
    mase: float = float("nan")

    def as_dict(self) -> dict:
        return {
            "MAE": self.mae, "RMSE": self.rmse, "MAPE (%)": self.mape,
            "WAPE": self.wape, "Bias": self.bias, "MASE": self.mase,
        }


def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else float("nan")


def bias(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    return float((y_pred - y_true).sum() / denom) if denom > 0 else float("nan")


def mase(y_true, y_pred, scale: float) -> float:
    """MAE divided by the in-sample seasonal-naive MAE (``scale``)."""
    if not scale or np.isnan(scale) or scale <= 0:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))) / scale)


def seasonal_naive_scale(train_values, season: int = 7) -> float:
    """Mean absolute seasonal difference of the training series."""
    v = np.asarray(train_values, dtype=float)
    if len(v) <= season:
        return float("nan")
    return float(np.mean(np.abs(v[season:] - v[:-season])))


def compute_metrics(y_true, y_pred, mase_scale: float | None = None) -> ForecastMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # avoid divide-by-zero on days with zero actual demand
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else float("nan")
    return ForecastMetrics(
        mae=mae, rmse=rmse, mape=mape,
        wape=wape(y_true, y_pred), bias=bias(y_true, y_pred),
        mase=mase(y_true, y_pred, mase_scale) if mase_scale is not None else float("nan"),
    )


def metrics_table(results: dict[str, ForecastMetrics]) -> pd.DataFrame:
    return pd.DataFrame({name: m.as_dict() for name, m in results.items()}).T
