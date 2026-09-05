"""One-time adapter: convert the raw retail CSV (Date, Store ID, Product ID,
Demand, Promotion, Price, ...) into the long-format schema this project's
pipeline expects (date, sku, location, demand, promotion, price).

Usage:
    python -m src.prepare_real_data --input /path/to/demand_forecasting.csv

Run once; the output is written to data/raw/sku_location_demand.csv, which
`src/data_loader.py` reads by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import config

COLUMN_MAP = {
    "Date": config.DATE_COL,
    "Product ID": config.SKU_COL,
    "Store ID": config.LOCATION_COL,
    "Demand": config.TARGET_COL,
    "Promotion": "promotion",
    "Price": "price",
}


def prepare(input_path: Path, output_path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing expected columns: {missing}")

    out = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)
    out[config.DATE_COL] = pd.to_datetime(out[config.DATE_COL])

    # Sanity check: exactly one row per (date, sku, location) — a ragged
    # panel would silently break the per-leaf feature engineering later.
    dupes = out.duplicated(subset=[config.DATE_COL, config.SKU_COL, config.LOCATION_COL]).sum()
    if dupes:
        raise ValueError(
            f"{dupes} duplicate (date, sku, location) rows found — resolve before proceeding."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, required=True, help="Path to the raw retail CSV.")
    parser.add_argument("--output", type=str, default=str(config.RAW_DATA_PATH))
    args = parser.parse_args()

    result = prepare(Path(args.input), Path(args.output))
    print(f"Wrote {len(result):,} rows to {args.output}")
    print(f"{result[config.SKU_COL].nunique()} SKUs x {result[config.LOCATION_COL].nunique()} locations "
          f"x {result[config.DATE_COL].nunique()} days")
