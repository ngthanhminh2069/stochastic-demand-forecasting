"""Turn probabilistic forecasts into inventory decisions, and evaluate
forecasts by their actual business cost rather than symmetric error alone.

Two things a plain MAE/RMSE-optimized point forecast can't do:

1. Size a safety stock buffer (needs a *distribution*, not a point).
2. Tell you whether a forecast's errors are actually expensive — a model
   that's slightly worse on MAE but never under-forecasts a stockout-prone
   SKU can be the better real-world choice.

**On cost inputs:** this dataset has no direct "$ lost per stockout" or
"$ cost of holding a unit" column, so those costs are *derived* from each
SKU's `price` using standard inventory-management rules of thumb — see
`derive_costs_from_price`. Treat the resulting $ figures as directionally
useful (comparing models/SKUs to each other), not as precise business
numbers, until real margin/holding-cost data is available.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config


def estimate_daily_sigma_from_quantiles(p10: float, p90: float) -> float:
    """Back out an implied daily demand std-dev from P10/P90 forecasts,
    assuming approximate normality of forecast error. This lets us size
    safety stock at *any* service level, not just the quantiles the model
    was trained on.
    """
    z10, z90 = norm.ppf(0.10), norm.ppf(0.90)
    spread_in_z = z90 - z10
    if spread_in_z <= 0:
        return 0.0
    return max(0.0, (p90 - p10) / spread_in_z)


def derive_costs_from_price(
    avg_unit_price: float | None,
    stockout_fraction: float = config.STOCKOUT_COST_AS_FRACTION_OF_PRICE,
    annual_holding_rate: float = config.ANNUAL_HOLDING_COST_RATE,
) -> tuple[float, float]:
    """Derive a per-SKU (stockout_cost, holding_cost_per_day) pair from its
    average unit price, using standard inventory-management rules of
    thumb. Falls back to flat illustrative constants if no price is
    available for the SKU.

    - Stockout cost: `stockout_fraction * price` — defaults to the full
      sale price (conservative: assumes the entire lost sale's revenue,
      plus goodwill, is forfeited; a true margin figure would be smaller).
    - Holding cost: `annual_holding_rate * price / 365` — the standard
      "cost of capital + storage + obsolescence risk" rule of thumb,
      commonly 20-30% of item value per year, prorated to a daily rate.
    """
    if avg_unit_price is None or avg_unit_price <= 0 or pd.isna(avg_unit_price):
        return config.FALLBACK_UNIT_STOCKOUT_COST, config.FALLBACK_UNIT_HOLDING_COST

    stockout_cost = stockout_fraction * avg_unit_price
    holding_cost_per_day = annual_holding_rate * avg_unit_price / 365
    return stockout_cost, holding_cost_per_day


@dataclass
class SafetyStockPlan:
    median_demand_over_lead_time: float
    safety_stock: float
    reorder_point: float
    service_level: float
    lead_time_days: int


def compute_safety_stock(
    median_daily_demand: float,
    daily_sigma: float,
    lead_time_days: int = config.DEFAULT_LEAD_TIME_DAYS,
    service_level: float = config.DEFAULT_SERVICE_LEVEL,
) -> SafetyStockPlan:
    """Classic safety-stock formula: SS = z * sigma_LT, where sigma_LT
    assumes i.i.d. daily demand so variance scales with lead time.
    """
    z = norm.ppf(service_level)
    sigma_lead_time = daily_sigma * np.sqrt(lead_time_days)
    safety_stock = z * sigma_lead_time
    median_over_lt = median_daily_demand * lead_time_days
    reorder_point = median_over_lt + safety_stock

    return SafetyStockPlan(
        median_demand_over_lead_time=median_over_lt,
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        service_level=service_level,
        lead_time_days=lead_time_days,
    )


def newsvendor_cost(
    y_true,
    y_pred,
    unit_stockout_cost: float = config.FALLBACK_UNIT_STOCKOUT_COST,
    unit_holding_cost: float = config.FALLBACK_UNIT_HOLDING_COST,
) -> float:
    """Asymmetric cost-weighted evaluation: under-forecasting (stockout)
    and over-forecasting (excess holding) are penalized at their real,
    different costs instead of symmetric MAE/RMSE.

    Pass `unit_stockout_cost`/`unit_holding_cost` explicitly (e.g. from
    `derive_costs_from_price`) — the defaults here are flat fallbacks,
    not meaningful business numbers on their own.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    shortfall = np.maximum(y_true - y_pred, 0)  # under-forecast -> stockout
    excess = np.maximum(y_pred - y_true, 0)  # over-forecast -> holding cost

    total_cost = unit_stockout_cost * shortfall.sum() + unit_holding_cost * excess.sum()
    return float(total_cost)


def newsvendor_cost_per_unit_time(
    y_true,
    y_pred,
    unit_stockout_cost: float = config.FALLBACK_UNIT_STOCKOUT_COST,
    unit_holding_cost: float = config.FALLBACK_UNIT_HOLDING_COST,
) -> float:
    """Same as `newsvendor_cost`, averaged per time period — comparable
    across forecasts of different lengths."""
    y_true = np.asarray(y_true, dtype=float)
    return newsvendor_cost(y_true, y_pred, unit_stockout_cost, unit_holding_cost) / max(1, len(y_true))
