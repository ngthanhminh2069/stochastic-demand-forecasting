"""Central configuration for the SKU x Location supply-chain forecasting project."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = Path(os.getenv("SC_DATA_PATH", DATA_DIR / "raw" / "sku_location_demand.csv"))
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

DATE_COL = "date"
SKU_COL = "sku"
LOCATION_COL = "location"
TARGET_COL = "demand"
PRICE_COL = "price"

# This project uses only the real dataset provided (adapted via
# src/prepare_real_data.py) - there is no synthetic-data generator or
# fallback anywhere in this pipeline.

# ---------------------------------------------------------------------------
# Feature engineering (direct multi-horizon, as-of-origin)
# ---------------------------------------------------------------------------
# Every training/validation row is one (forecast origin, horizon h) pair.
# Features may only use demand observed on or before the origin day.
# `price` and `promotion` of the *target* day are treated as known in
# advance (planned promotions / price lists) - see README "Assumptions".
SEASONAL_LAG_WEEKS = 3          # same-weekday demand 1..3 weeks back, relative to the most recent known occurrence
ROLLING_WINDOWS = [7, 28]       # rolling mean/std of demand ending at the origin day (inclusive)

# ---------------------------------------------------------------------------
# Walk-forward backtest (see also proj1's src/backtest.py - same idea,
# duplicated here so the two projects stay independent / standalone)
# ---------------------------------------------------------------------------
HORIZON_DAYS = 14
STEP_DAYS = 28
MIN_TRAIN_DAYS = 180

# ---------------------------------------------------------------------------
# Probabilistic forecasting
# ---------------------------------------------------------------------------
# Wide enough that a 95% / 97.5% cycle service level is a *trained*
# quantile, not an extrapolation from P10/P90 under a normality assumption.
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
MEDIAN_Q = 0.5
BAND_LOW_Q = 0.1    # P10-P90 band used for plots / coverage summaries
BAND_HIGH_Q = 0.9

CONFORMAL_CALIB_DAYS = 60   # trailing training days (as forecast *targets*) carved out for conformal calibration
# Calibration offsets are estimated separately per horizon bucket because
# forecast spread grows with horizon. Residuals are pooled across all
# locations of the same SKU so each offset rests on hundreds of points
# instead of a handful.
HORIZON_BUCKETS = [(1, 7), (8, 14)]
XGB_QUANTILE_PARAMS = dict(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

# ---------------------------------------------------------------------------
# Scenario generation (sample-based rollup + lead-time demand)
# ---------------------------------------------------------------------------
N_SCENARIO_SAMPLES = 2000
SCENARIO_SEED = 42
COPULA_LEAF_SHRINKAGE = 0.2     # shrink the estimated cross-leaf correlation matrix toward identity
COPULA_MAX_HORIZON_RHO = 0.95   # cap on the AR(1) correlation between adjacent horizons

# ---------------------------------------------------------------------------
# Inventory / newsvendor cost assumptions
# ---------------------------------------------------------------------------
# The dataset has no direct "cost of a lost sale" or "cost of holding a
# unit" column - those are derived from `price` using standard inventory-
# management rules of thumb, NOT observed business data. See
# src/inventory.py:derive_costs_from_price for the formula and README
# "Assumptions" for the consequences (notably: because both costs scale
# with price, the critical ratio is identical for every SKU).
GROSS_MARGIN = 0.30                # a stockout forfeits the *margin* on the lost sale, not the whole price
ANNUAL_HOLDING_COST_RATE = 0.25    # common rule of thumb: holding 1 unit costs ~20-30% of its price per year
REVIEW_PERIOD_DAYS = 7             # ordering cycle: the holding cost of an unsold unit is charged per cycle
MAX_CRITICAL_RATIO = 0.975         # cap = highest trained quantile; the raw ratio is typically ~0.98
DEFAULT_LEAD_TIME_DAYS = 7
DEFAULT_LEAD_TIME_STD_DAYS = 0.0   # std-dev of lead time in days; 0 = deterministic lead time
DEFAULT_SERVICE_LEVEL = 0.95       # target cycle service level for safety stock

# Flat fallback costs (per unit; holding is per ordering cycle), used only
# if no price data is available for a leaf.
FALLBACK_UNIT_STOCKOUT_COST = 8.0
FALLBACK_UNIT_HOLDING_COST = 1.5

# ---------------------------------------------------------------------------
# Drop detection (CUSUM control chart)
# ---------------------------------------------------------------------------
CUSUM_THRESHOLD_STD = 4.0   # alarm when cumulative deviation exceeds this many std devs
CUSUM_SLACK_STD = 0.5       # slack (k) in std devs, filters out small noise
