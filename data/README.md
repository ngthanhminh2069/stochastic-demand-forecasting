# Data

**Real dataset included with this delivery.** The original file
(`Date, Store ID, Product ID, Category, Region, Demand, Promotion,
Price, ...`) is adapted into this project's schema via:

```bash
python -m src.prepare_real_data --input /path/to/demand_forecasting.csv
```

This writes `data/raw/sku_location_demand.csv` (already generated for
this delivery: 20 SKUs × 5 locations × 760 days = 76,000 rows) with the
columns the pipeline expects:

| Column      | Type    | Description                        |
|-------------|---------|--------------------------------------|
| `date`      | date    | Observation date                     |
| `sku`       | string  | SKU identifier (from `Product ID`)   |
| `location`  | string  | Store identifier (from `Store ID`)   |
| `demand`    | numeric | Units demanded that day               |
| `promotion` | 0/1     | Exogenous regressor                   |
| `price`     | numeric | Exogenous regressor                   |

The adapter validates there's exactly one row per (date, sku, location) —
a ragged panel would silently break the per-leaf feature engineering.

## Data quality notes (from the real dataset)

- **`Store ID` → `Region` is a fixed mapping** (each store belongs to
  exactly one region) — a natural 4th hierarchy level (Total → Region →
  Store → SKU) if you want to extend `src/hierarchy.py`.
- **`Product ID` → `Category` is *not* consistent** — the same product
  appears under different categories across rows. Category is therefore
  not used as a hierarchy dimension in this project.
- **`Demand` vs. `Units Sold` disagree ~28% of the time**, and
  `Units Sold` is always ≤ `Inventory Level` — consistent with `Units
  Sold` being censored (stockout-limited) realized sales, while `Demand`
  looks like an underlying demand signal. This project forecasts
  `Demand` as given and does not model censoring — worth flagging as a
  known simplification if you extend this further.

## Only real data — no synthetic fallback

This project deliberately does not include a synthetic-data generator.
If `data/raw/sku_location_demand.csv` is missing, `run_pipeline.py` will
raise a clear error pointing back to the `prepare_real_data.py` command
above rather than silently substituting fake data.
