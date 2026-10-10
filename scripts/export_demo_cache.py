#!/usr/bin/env python
"""Pre-compute and export all backtest results, diagnostics, and scenario
samples to data/cache/demo_cache.pkl so the Streamlit demo app launches in < 1 second.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from src import config, pipeline, scenarios, hierarchy, inventory

logger = logging.getLogger("export_cache")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export demo cache for Streamlit dashboard")
    parser.add_argument("--n-jobs", type=int, default=12, help="Number of worker processes")
    parser.add_argument("--out", type=str, default=str(ROOT / "data" / "cache" / "demo_cache.pkl.gz"),
                        help="Path to output compressed cache file (defaults to demo_cache.pkl.gz)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== [1/4] Loading dataset ===")
    df = pipeline.load_data()
    print(f"Loaded {len(df):,} rows across {df[config.SKU_COL].nunique()} SKUs x {df[config.LOCATION_COL].nunique()} Locations.")

    print(f"\n=== [2/4] Running walk-forward backtest (n_jobs={args.n_jobs}) ===")
    leaf_results = pipeline.run_all_leaves(df, n_jobs=args.n_jobs)
    print(f"Finished backtesting {len(leaf_results)} leaves.")

    print("\n=== [3/4] Computing diagnostics ===")
    diagnostics = pipeline.aggregate_diagnostics(leaf_results)

    print("\n=== [4/4] Generating joint scenario samples & Copula dependence ===")
    scenarios_data = pipeline.build_latest_scenarios(leaf_results)

    # Clean up leaf_results for compact serialization
    # Keep essential objects per leaf:
    # sku, location, avg_price, unit_stockout_cost, unit_holding_cost, critical_ratio, service_quantile,
    # mean_mae, mean_wape, mean_bias, mean_mase, baseline_mean_mae, baseline_mean_wape, baseline_mean_mase,
    # mean_newsvendor_cost_per_day, mean_newsvendor_cost_per_day_median,
    # last_val_forecast, last_val_actual, cusum_result, raw_series, residuals, val_records
    compact_leaves = {}
    for key, r in leaf_results.items():
        compact_leaves[key] = {
            "sku": r["sku"],
            "location": r["location"],
            "avg_price": r["avg_price"],
            "unit_stockout_cost": r["unit_stockout_cost"],
            "unit_holding_cost": r["unit_holding_cost"],
            "critical_ratio": r["critical_ratio"],
            "service_quantile": r["service_quantile"],
            "mean_mae": r["mean_mae"],
            "mean_wape": r["mean_wape"],
            "mean_bias": r["mean_bias"],
            "mean_mase": r["mean_mase"],
            "baseline_mean_mae": r["baseline_mean_mae"],
            "baseline_mean_wape": r["baseline_mean_wape"],
            "baseline_mean_mase": r["baseline_mean_mase"],
            "mean_newsvendor_cost_per_day": r["mean_newsvendor_cost_per_day"],
            "mean_newsvendor_cost_per_day_median": r["mean_newsvendor_cost_per_day_median"],
            "mean_pinball_by_quantile": r["mean_pinball_by_quantile"],
            "last_val_forecast": r["last_val_forecast"],
            "last_val_actual": r["last_val_actual"],
            "cusum_result": r["cusum_result"],
            "raw_series": r["raw_series"],
            "residuals": r["residuals"],
            "val_records": r["val_records"],
        }

    # Optimize array precision for compact storage and lightning-fast loading
    import numpy as np
    import gzip
    if "samples" in scenarios_data and hasattr(scenarios_data["samples"], "astype"):
        scenarios_data["samples"] = scenarios_data["samples"].astype(np.float32)
    if "aggregates" in scenarios_data:
        for k, arr in scenarios_data["aggregates"].items():
            if hasattr(arr, "astype"):
                scenarios_data["aggregates"][k] = arr.astype(np.float32)

    cache = {
        "version": "2.0",
        "raw_df": df,
        "leaf_results": compact_leaves,
        "diagnostics": diagnostics,
        "scenarios_data": scenarios_data,
    }

    if str(out_path).endswith(".gz"):
        print(f"\nSaving compressed deployment cache to {out_path}...")
        with gzip.open(out_path, "wb", compresslevel=6) as f_gz:
            pickle.dump(cache, f_gz, protocol=pickle.HIGHEST_PROTOCOL)
        gz_size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"Compressed cache saved: {gz_size_mb:.2f} MB (Optimized for Streamlit Cloud deployment)")
    else:
        print(f"\nSaving uncompressed cache to {out_path}...")
        with open(out_path, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"Uncompressed cache saved: {size_mb:.2f} MB")
        
        gz_path = out_path.with_suffix(".pkl.gz")
        print(f"Saving compressed deployment cache to {gz_path}...")
        with gzip.open(gz_path, "wb", compresslevel=6) as f_gz:
            pickle.dump(cache, f_gz, protocol=pickle.HIGHEST_PROTOCOL)
        gz_size_mb = gz_path.stat().st_size / (1024 * 1024)
        print(f"Compressed cache saved: {gz_size_mb:.2f} MB (Optimized for Streamlit Cloud deployment)")


if __name__ == "__main__":
    main()
