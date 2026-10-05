#!/usr/bin/env python
"""CLI entry point: run the full SKU x Location supply-chain pipeline on
the real dataset (no synthetic data anywhere in this project).

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --max-leaves 10   # quick smoke test
    python scripts/run_pipeline.py --n-jobs 4         # parallelize across 4 CPU cores (per SKU)
    python scripts/run_pipeline.py --lead-time 10 --lead-time-std 2 --service-level 0.97
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
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
        help="Number of SKUs to backtest in parallel via CPU multiprocessing "
             "(not GPU/CUDA). 1 = sequential (default). Try os.cpu_count() for "
             "a full run.",
    )
    parser.add_argument("--lead-time", type=float, default=config.DEFAULT_LEAD_TIME_DAYS,
                        help="Mean replenishment lead time in days (max %d)." % config.HORIZON_DAYS)
    parser.add_argument("--lead-time-std", type=float, default=config.DEFAULT_LEAD_TIME_STD_DAYS,
                        help="Std-dev of the lead time in days (0 = deterministic).")
    parser.add_argument("--service-level", type=float, default=config.DEFAULT_SERVICE_LEVEL,
                        help="Target cycle service level for safety stock.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pd.set_option("display.width", 200)

    df = pipeline.load_data()
    print(f"Loaded {len(df):,} rows across "
          f"{df[config.SKU_COL].nunique()} SKUs x {df[config.LOCATION_COL].nunique()} locations.")

    print("\n=== Generating data-characteristics figures ===")
    plotting.plot_demand_distribution_and_seasonality(df)
    plotting.plot_demand_heatmap(df)

    results = pipeline.run_all_leaves(df, max_leaves=args.max_leaves, n_jobs=args.n_jobs)
    diagnostics = pipeline.aggregate_diagnostics(results)

    print("\n=== Per-leaf backtest summary ===")
    print("(as-of-origin, direct multi-horizon: every forecast uses only data known at its origin)")
    rows = []
    for (sku, location), r in results.items():
        row = {"sku": sku, "location": location, "n_folds": r["n_folds"],
               "avg_price": round(r["avg_price"], 2),
               "WAPE": round(r["mean_wape"], 3), "WAPE_naive": round(r["baseline_mean_wape"], 3),
               "MASE": round(r["mean_mase"], 3), "bias": round(r["mean_bias"], 3),
               "crit_ratio": round(r["critical_ratio"], 3),
               "cost_$/day@CR": round(r["mean_newsvendor_cost_per_day"], 2),
               "cost_$/day@P50": round(r["mean_newsvendor_cost_per_day_median"], 2)}
        n_drops = len(r["cusum_result"].alarm_days) if r["cusum_result"] else 0
        row["cusum_alarm_days"] = n_drops
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== Portfolio accuracy: model vs. seasonal-naive benchmark ===")
    print(diagnostics["portfolio_accuracy"].round(3).to_string(index=False))
    print(f"Share of leaves where the model beats seasonal naive on MAE: "
          f"{diagnostics['share_leaves_beating_baseline_mae']:.0%}")

    print("\n=== Model fit & calibration diagnostics ===")
    print("Reliability table (nominal vs. empirical quantile coverage):")
    print(diagnostics["reliability_table"].round(3).to_string(index=False))
    print(f"Pooled residual mean: {diagnostics['all_residuals'].mean():.2f} "
          f"(should be close to 0 if the median forecast is unbiased)")
    if len(diagnostics["coverage_by_horizon"]):
        print("\nP10-P90 coverage by horizon (nominal 80%):")
        print(diagnostics["coverage_by_horizon"].set_index("h")["p10_p90_coverage"].round(2).to_frame().T.to_string())

    print("\n=== Latest forecast: joint scenarios, aggregates and safety stock ===")
    scen = pipeline.build_latest_scenarios(
        results, lead_time_days=args.lead_time, lead_time_std_days=args.lead_time_std,
        service_level=args.service_level,
    )
    dep = scen["dependence"]
    print(f"Dependence from backtest PITs: mean cross-leaf corr = {dep.mean_leaf_corr:.3f}, "
          f"adjacent-horizon rho = {dep.rho_h:.2f}")
    summary = scen["aggregate_summary"]
    show = summary[summary["aggregate"].isin(["total"]) | summary["aggregate"].str.startswith("location:")]
    print(show.round(2).to_string(index=False))
    total = summary.set_index("aggregate").loc["total"]
    print(f"Risk pooling: Total safety stock {total['pooled_safety_stock']:.0f} units vs. "
          f"{total['sum_leaf_safety_stock']:.0f} if every leaf is buffered separately "
          f"({total['risk_pooling_saving']:.0%} lower).")

    first_leaf = next(iter(results.values()))
    leaf_key = (first_leaf["sku"], first_leaf["location"])
    plan = scen["plans"].get(leaf_key)
    if plan:
        print(f"\nExample plan {first_leaf['sku']} x {first_leaf['location']} "
              f"(stockout cost=${first_leaf['unit_stockout_cost']:.2f}/unit, "
              f"holding cost=${first_leaf['unit_holding_cost']:.3f}/unit/cycle, "
              f"derived from avg price ${first_leaf['avg_price']:.2f} and "
              f"{config.GROSS_MARGIN:.0%} margin; critical ratio {first_leaf['critical_ratio']:.3f}): "
              f"reorder point = {plan.reorder_point:.1f} units "
              f"(expected demand over {plan.lead_time_days:g}d lead time = "
              f"{plan.expected_demand_over_lead_time:.1f}, "
              f"safety stock = {plan.safety_stock:.1f}, "
              f"service level = {plan.service_level:.0%})")

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
    plotting.plot_hierarchy_bars(summary)
    if len(diagnostics["all_residuals"]):
        plotting.plot_residual_diagnostics(diagnostics["all_residuals"])
    plotting.plot_reliability_diagram(diagnostics["reliability_table"])
    if len(diagnostics["coverage_by_horizon"]):
        plotting.plot_coverage_by_horizon(diagnostics["coverage_by_horizon"])
    if len(diagnostics["pinball_by_leaf_and_quantile"]):
        plotting.plot_pinball_by_quantile(diagnostics["pinball_by_leaf_and_quantile"])
    if len(diagnostics["leaf_summary"]):
        plotting.plot_leaf_accuracy_distribution(diagnostics["leaf_summary"])
    print(f"Figures saved to: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
