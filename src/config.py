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
# src/prepare_real_data.py) — there is no synthetic-data generator or
# fallback anywhere in this pipeline.

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
LAG_DAYS = [7, 14, 28]
ROLLING_WINDOWS = [7, 28]

# ---------------------------------------------------------------------------
# Walk-forward backtest (see also proj1's src/backtest.py — same idea,
# duplicated here so the two projects stay independent / standalone)
# ---------------------------------------------------------------------------
HORIZON_DAYS = 14
STEP_DAYS = 28
MIN_TRAIN_DAYS = 180

# ---------------------------------------------------------------------------
# Probabilistic forecasting
# ---------------------------------------------------------------------------
QUANTILES = [0.1, 0.5, 0.9]  # P10 / P50 / P90
CONFORMAL_CALIB_DAYS = 30  # trailing training days carved out for conformal calibration
XGB_QUANTILE_PARAMS = dict(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

# ---------------------------------------------------------------------------
# Inventory / newsvendor cost assumptions
# ---------------------------------------------------------------------------
# The dataset has no direct "cost of a lost sale" or "cost of holding a
# unit" column — those are derived from `price` using standard inventory-
# management rules of thumb, NOT observed business data. Treat the
# resulting $ figures as directionally useful (comparing models/SKUs to
# each other) rather than precise, until real margin/holding-cost data is
# available. See src/inventory.py:derive_costs_from_price for the formula.
STOCKOUT_COST_AS_FRACTION_OF_PRICE = 1.0   # assume the full sale price is lost on a stockout (conservative)
ANNUAL_HOLDING_COST_RATE = 0.25            # common rule of thumb: holding 1 unit costs ~20-30% of its price per year
DEFAULT_LEAD_TIME_DAYS = 7
DEFAULT_SERVICE_LEVEL = 0.95               # target cycle service level for safety stock

# Flat fallback costs, used only if no price data is available for a leaf.
FALLBACK_UNIT_STOCKOUT_COST = 8.0
FALLBACK_UNIT_HOLDING_COST = 1.5

# ---------------------------------------------------------------------------
# Drop detection (CUSUM control chart)
# ---------------------------------------------------------------------------
CUSUM_THRESHOLD_STD = 4.0   # alarm when cumulative deviation exceeds this many std devs
CUSUM_SLACK_STD = 0.5       # slack (k) in std devs, filters out small noise
