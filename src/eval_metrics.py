"""Point-forecast metrics (MAE/RMSE/MAPE), re-exported alongside the
pinball loss in `probabilistic.py` and the cost-weighted metrics in
`inventory.py` — kept separate because they answer different questions
(point accuracy vs. distributional calibration vs. business cost)."""
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

    def as_dict(self) -> dict:
        return {"MAE": self.mae, "RMSE": self.rmse, "MAPE (%)": self.mape}


def compute_metrics(y_true, y_pred) -> ForecastMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # avoid divide-by-zero on days with zero actual demand
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else float("nan")
    return ForecastMetrics(mae=mae, rmse=rmse, mape=mape)


def metrics_table(results: dict[str, ForecastMetrics]) -> pd.DataFrame:
    return pd.DataFrame({name: m.as_dict() for name, m in results.items()}).T
