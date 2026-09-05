#!/usr/bin/env python
"""CLI entry point: run the full SKU x Location supply-chain pipeline on
the real dataset (no synthetic data anywhere in this project).

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --max-leaves 10   # quick smoke test
    python scripts/run_pipeline.py --n-jobs 4         # parallelize across 4 CPU cores
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config, pipeline, plotting  # noqa: E402


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-leaves", type=int, default=None,
        help="Only backtest the first N (sku, location) leaves — useful for a quick smoke test.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Number of leaves to backtest in parallel via CPU multiprocessing "
             "(not GPU/CUDA). 1 = sequential (default). Try os.cpu_count() for "
             "a full run.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    df = pipeline.load_data()
    print(f"Loaded {len(df):,} rows across "
          f"{df[config.SKU_COL].nunique()} SKUs x {df[config.LOCATION_COL].nunique()} locations.")

    print("\n=== Generating data-characteristics figures ===")
    plotting.plot_demand_distribution_and_seasonality(df)
    plotting.plot_demand_heatmap(df)

    results = pipeline.run_all_leaves(df, max_leaves=args.max_leaves, n_jobs=args.n_jobs)

    print("\n=== Per-leaf backtest summary (mean pinball loss by quantile, mean daily newsvendor cost) ===")
    rows = []
    for (sku, location), r in results.items():
        row = {"sku": sku, "location": location, "n_folds": r["n_folds"],
               "avg_price": round(r["avg_price"], 2),
               "newsvendor_$/day": round(r["mean_newsvendor_cost_per_day"], 2)}
        for q, val in r["mean_pinball_by_quantile"].items():
            row[f"pinball_P{int(q*100)}"] = round(val, 2)
        n_drops = len(r["cusum_result"].alarm_days) if r["cusum_result"] else 0
        row["cusum_alarm_days"] = n_drops
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Bottom-up reconciled forecasts (last fold, median) ===")
    aggregates = pipeline.reconcile_latest_forecasts(results)
    if aggregates.get("total") is not None:
        print(f"Total forecast (last fold horizon): {aggregates['total'].sum():.1f} units")

    print("\n=== Example safety stock plan (first leaf) ===")
    first_leaf = next(iter(results.values()))
    plan = pipeline.build_safety_stock_plan(first_leaf)
    if plan:
        print(f"{first_leaf['sku']} x {first_leaf['location']} "
              f"(stockout cost=${first_leaf['unit_stockout_cost']:.2f}/unit, "
              f"holding cost=${first_leaf['unit_holding_cost']:.3f}/unit/day, "
              f"derived from avg price ${first_leaf['avg_price']:.2f}): "
              f"reorder point = {plan.reorder_point:.1f} units "
              f"(median demand over {plan.lead_time_days}d lead time = "
              f"{plan.median_demand_over_lead_time:.1f}, "
              f"safety stock = {plan.safety_stock:.1f}, "
              f"service level = {plan.service_level:.0%})")

    print("\n=== Model fit & calibration diagnostics ===")
    diagnostics = pipeline.aggregate_diagnostics(results)
    print("Reliability table (nominal vs. empirical quantile coverage):")
    print(diagnostics["reliability_table"].round(3).to_string(index=False))
    print(f"Pooled residual mean: {diagnostics['all_residuals'].mean():.2f} "
          f"(should be close to 0 if the median forecast is unbiased)")

    print("\n=== Generating model-fit and accuracy figures ===")
    if first_leaf["last_val_forecast"] is not None:
        plotting.plot_quantile_fan(
            first_leaf["last_val_actual"], first_leaf["last_val_forecast"],
            title=f"Probabilistic Forecast: {first_leaf['sku']} x {first_leaf['location']}",
            name="quantile_fan_example",
        )
    if first_leaf["cusum_result"] is not None:
        plotting.plot_cusum(
            first_leaf["raw_series"], first_leaf["cusum_result"],
            title=f"Demand & CUSUM Drop Detection: {first_leaf['sku']} x {first_leaf['location']}",
            name="cusum_example",
        )
    plotting.plot_hierarchy_bars(aggregates)
    if len(diagnostics["all_residuals"]):
        plotting.plot_residual_diagnostics(diagnostics["all_residuals"])
    plotting.plot_reliability_diagram(diagnostics["reliability_table"])
    if len(diagnostics["pinball_by_leaf_and_quantile"]):
        plotting.plot_pinball_by_quantile(diagnostics["pinball_by_leaf_and_quantile"])
    if len(diagnostics["leaf_summary"]):
        plotting.plot_leaf_accuracy_distribution(diagnostics["leaf_summary"])
    print(f"Figures saved to: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
