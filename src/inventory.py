"""Turn probabilistic forecasts into inventory decisions, and evaluate
forecasts by their actual business cost rather than symmetric error alone.

Two things a plain MAE/RMSE-optimized point forecast can't do:

1. Size a safety stock buffer (needs a *distribution*, not a point).
2. Tell you whether a forecast's errors are actually expensive - a model
   that's slightly worse on MAE but never under-forecasts a stockout-prone
   SKU can be the better real-world choice.

**Newsvendor logic.** With per-unit underage cost Cu (a lost sale) and
overage cost Co (an unsold unit held for one ordering cycle) the optimal
stocking level is the *critical-ratio quantile* ``Cu / (Cu + Co)`` of the
demand distribution - not the median. Forecasts are therefore scored at
that quantile (`select_service_quantile`), not at P50.

**On cost inputs:** this dataset has no direct "$ lost per stockout" or
"$ cost of holding a unit" column, so those costs are *derived* from each
SKU's `price` - see `derive_costs_from_price` and README "Assumptions".
Treat the resulting $ figures as directionally useful, not as precise
business numbers, until real margin/holding-cost data is available.

**Safety stock.** `safety_stock_from_lead_time_samples` reads the reorder
point straight off the simulated distribution of demand over the lead time
(optionally with random lead times). The textbook ``z * sigma * sqrt(L)``
formula (`compute_safety_stock`) is kept as a reference for comparison; it
assumes i.i.d. normal daily demand and a fixed lead time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def derive_costs_from_price(
    avg_unit_price: float | None,
    gross_margin: float = config.GROSS_MARGIN,
    annual_holding_rate: float = config.ANNUAL_HOLDING_COST_RATE,
    review_period_days: int = config.REVIEW_PERIOD_DAYS,
) -> tuple[float, float]:
    """Derive per-SKU ``(stockout_cost_per_unit, holding_cost_per_unit_per_cycle)``
    from its average unit price. Falls back to flat illustrative constants
    if no price is available.

    - Stockout (underage) cost: ``gross_margin * price`` - a lost sale
      forfeits the *margin*, not the whole revenue (the cost of goods is
      not spent if nothing is sold). Goodwill/backorder effects are
      ignored.
    - Holding (overage) cost: ``annual_holding_rate * price *
      review_period_days / 365`` - the standard "cost of capital + storage
      + obsolescence" rule of thumb (20-30%/year), charged for one
      ordering cycle, which is how long a surplus unit is held before the
      next order can correct it.
    """
    if avg_unit_price is None or avg_unit_price <= 0 or pd.isna(avg_unit_price):
        return config.FALLBACK_UNIT_STOCKOUT_COST, config.FALLBACK_UNIT_HOLDING_COST

    stockout_cost = gross_margin * avg_unit_price
    holding_cost_per_cycle = annual_holding_rate * avg_unit_price * review_period_days / 365
    return stockout_cost, holding_cost_per_cycle


def critical_ratio(
    unit_stockout_cost: float,
    unit_holding_cost: float,
    max_ratio: float = config.MAX_CRITICAL_RATIO,
) -> float:
    """Newsvendor critical ratio ``Cu / (Cu + Co)``, capped at ``max_ratio``
    (the highest quantile the model is trained/calibrated for)."""
    total = unit_stockout_cost + unit_holding_cost
    if total <= 0:
        return 0.5
    return float(min(max_ratio, unit_stockout_cost / total))


def select_service_quantile(ratio: float, levels: list[float] = config.QUANTILES) -> float:
    """Trained quantile level closest to the critical ratio."""
    lv = np.asarray(sorted(levels))
    return float(lv[np.argmin(np.abs(lv - ratio))])


def newsvendor_cost(
    y_true,
    y_pred,
    unit_stockout_cost: float = config.FALLBACK_UNIT_STOCKOUT_COST,
    unit_holding_cost: float = config.FALLBACK_UNIT_HOLDING_COST,
) -> float:
    """Asymmetric cost-weighted evaluation: under-forecasting (stockout)
    and over-forecasting (excess holding) are penalized at their different
    costs instead of symmetric MAE/RMSE.

    Pass `unit_stockout_cost`/`unit_holding_cost` explicitly (e.g. from
    `derive_costs_from_price`) - the defaults here are flat fallbacks,
    not meaningful business numbers on their own.

    Note that ``newsvendor_cost`` of a forecast at the critical-ratio
    quantile equals ``(Cu + Co) * pinball_loss`` summed - i.e. the
    critical-ratio quantile is exactly the cost-minimising forecast.
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
    """Same as `newsvendor_cost`, averaged per time period - comparable
    across forecasts of different lengths."""
    y_true = np.asarray(y_true, dtype=float)
    return newsvendor_cost(y_true, y_pred, unit_stockout_cost, unit_holding_cost) / max(1, len(y_true))


# ---------------------------------------------------------------------------
# Safety stock
# ---------------------------------------------------------------------------
@dataclass
class LeadTimePlan:
    expected_demand_over_lead_time: float
    safety_stock: float
    reorder_point: float
    service_level: float
    lead_time_days: float
    lead_time_std_days: float


def safety_stock_from_lead_time_samples(
    lead_time_demand_samples: np.ndarray,
    service_level: float = config.DEFAULT_SERVICE_LEVEL,
    lead_time_days: float = config.DEFAULT_LEAD_TIME_DAYS,
    lead_time_std_days: float = config.DEFAULT_LEAD_TIME_STD_DAYS,
) -> LeadTimePlan:
    """Reorder point = ``service_level`` quantile of simulated lead-time
    demand (cycle service level); safety stock = reorder point minus the
    expected lead-time demand."""
    s = np.asarray(lead_time_demand_samples, dtype=float)
    expected = float(s.mean())
    rop = float(np.quantile(s, service_level))
    return LeadTimePlan(
        expected_demand_over_lead_time=expected,
        safety_stock=max(0.0, rop - expected),
        reorder_point=rop,
        service_level=service_level,
        lead_time_days=lead_time_days,
        lead_time_std_days=lead_time_std_days,
    )


# --- classical reference formula (kept for comparison) ---------------------
def estimate_daily_sigma_from_quantiles(p10: float, p90: float) -> float:
    """Back out an implied daily demand std-dev from P10/P90 forecasts,
    assuming approximate normality of forecast error. Reference only: the
    pipeline now sizes safety stock from simulated lead-time demand."""
    z10, z90 = norm.ppf(0.10), norm.ppf(0.90)
    spread_in_z = z90 - z10
    if spread_in_z <= 0:
        return 0.0
    return max(0.0, (p90 - p10) / spread_in_z)


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
    assumes i.i.d. daily demand so variance scales with lead time. Kept as
    a textbook reference to compare against the simulation-based plan."""
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
