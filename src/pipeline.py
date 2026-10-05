"""End-to-end SKU x Location pipeline (real dataset only — no synthetic
data anywhere in this project):

1. Load the real SKU x Location demand panel (prepared via
   src/prepare_real_data.py).
2. Walk-forward backtest, **as-of-origin and direct multi-horizon**: at
   each fold's origin (the last known day) the model forecasts days
   1..14 ahead using only information available at that origin. A
   multi-quantile XGBoost per leaf is conformal-calibrated per horizon
   bucket, with calibration residuals pooled across the locations of each
   SKU. Scored with pinball loss (distributional), MAE/WAPE/MASE/bias
   (point, via the median, against a seasonal-naive benchmark) and
   newsvendor cost (business — margin-based costs, scored at the
   critical-ratio quantile).
3. Estimate cross-leaf / cross-horizon dependence from backtest PITs and
   draw **joint sample paths** for the latest fold's forecast; aggregate
   SKU / Location / Total distributions and lead-time demand from the
   *paths* (quantiles are not additive).
4. Run CUSUM drop detection per leaf.
5. Size safety stock / reorder points from simulated lead-time demand.
6. Accumulate residuals and coverage across every fold of every leaf, so
   the pipeline can report whether the model is actually well-calibrated
   (not just "accurate on average") — see `aggregate_diagnostics` and
   `src/plotting.py`.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import (
    backtest, config, data_loader, drop_detection, eval_metrics, features, hierarchy,
    inventory, probabilistic, scenarios,
)

logger = logging.getLogger(__name__)

_H = np.arange(1, config.HORIZON_DAYS + 1)


def load_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the real SKU x Location dataset. Raises FileNotFoundError with
    setup instructions if it hasn't been prepared yet — this project does
    not fall back to synthetic data."""
    return data_loader.load_raw_data(path)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def _prepare_leaf(df: pd.DataFrame, sku: str, location: str) -> dict:
    series = data_loader.get_series(df, sku, location)
    idx = series.index
    if (idx[-1] - idx[0]).days + 1 != len(idx) or not idx.is_unique:
        raise ValueError(f"{sku} x {location}: demand series is not a contiguous daily series.")

    leaf_rows = df[(df[config.SKU_COL] == sku) & (df[config.LOCATION_COL] == location)]
    extra = leaf_rows.set_index(config.DATE_COL)[["promotion", config.PRICE_COL]].sort_index()

    avg_price = leaf_rows[config.PRICE_COL].mean()
    unit_stockout_cost, unit_holding_cost = inventory.derive_costs_from_price(avg_price)
    ratio = inventory.critical_ratio(unit_stockout_cost, unit_holding_cost)

    origins = np.arange(features.min_origin_pos(_H), len(series) - 1)
    X_all, meta_all = features.make_direct_frame(series, extra, origins, _H)
    return {
        "sku": sku, "location": location, "series": series, "avg_price": avg_price,
        "unit_stockout_cost": unit_stockout_cost, "unit_holding_cost": unit_holding_cost,
        "critical_ratio": ratio, "service_quantile": inventory.select_service_quantile(ratio),
        "X_all": X_all, "meta_all": meta_all,
        "records": [], "fold_pinball": {q: [] for q in config.QUANTILES},
        "fold_point": [], "fold_baseline": [], "fold_cost_cr": [], "fold_cost_median": [],
    }


def run_sku_backtest(
    df: pd.DataFrame, sku: str, locations: list[str] | None = None, xgb_n_jobs: int | None = None
) -> dict[tuple[str, str], dict]:
    """Walk-forward backtest for every location of one SKU.

    The leaves of a SKU are processed together because conformal
    calibration residuals are pooled across its locations (each offset
    then rests on hundreds of points instead of a handful).

    `xgb_n_jobs`: forwarded to `probabilistic.fit_quantile_model`. Leave
    as None for a normal sequential run; `run_all_leaves(..., n_jobs=N)`
    sets this to 1 when N > 1.
    """
    if locations is None:
        locations = sorted(df.loc[df[config.SKU_COL] == sku, config.LOCATION_COL].unique())
    leaves = {loc: _prepare_leaf(df, sku, loc) for loc in locations}

    ref_series = next(iter(leaves.values()))["series"]
    folds = backtest.generate_folds(ref_series)
    qs = sorted(config.QUANTILES)
    median_q = config.MEDIAN_Q
    calib_days = config.CONFORMAL_CALIB_DAYS
    min_train_rows = len(_H) * 30

    for fold in folds:
        te = ref_series.index.get_loc(fold.train_end)   # position of the first validation day
        cs = te - calib_days                             # first calibration *target* day
        per_leaf = {}
        residual_frames = []

        for loc, leaf in leaves.items():
            meta, X = leaf["meta_all"], leaf["X_all"]
            tp, op = meta["target_pos"].to_numpy(), meta["origin_pos"].to_numpy()
            train, calib, val = tp < cs, (tp >= cs) & (tp < te), op == te - 1
            if train.sum() < min_train_rows or calib.sum() == 0 or val.sum() < len(_H):
                continue

            model = probabilistic.fit_quantile_model(X[train], meta.loc[train, "y"], n_jobs=xgb_n_jobs)
            raw_calib = probabilistic.predict_quantiles(model, X[calib])
            raw_val = probabilistic.predict_quantiles(model, X[val])
            residual_frames.append(
                probabilistic.calibration_residuals(raw_calib, meta.loc[calib, "y"], meta.loc[calib, "h"])
            )
            per_leaf[loc] = (raw_val, meta[val])

        if not per_leaf:
            continue
        offsets = probabilistic.conformal_offsets(pd.concat(residual_frames, ignore_index=True))

        for loc, (raw_val, vmeta) in per_leaf.items():
            leaf = leaves[loc]
            Y = leaf["series"].to_numpy(dtype=float)
            h = vmeta["h"].to_numpy()
            cal = probabilistic.apply_conformal_offsets(raw_val, h, offsets)
            y = vmeta["y"].to_numpy()
            baseline = features.seasonal_naive_forecast(Y, te - 1, h)
            scale = eval_metrics.seasonal_naive_scale(Y[:te])

            for q in qs:
                leaf["fold_pinball"][q].append(probabilistic.pinball_loss(y, cal[q].to_numpy(), q))
            point = cal[median_q].to_numpy()
            leaf["fold_point"].append(eval_metrics.compute_metrics(y, point, scale))
            leaf["fold_baseline"].append(eval_metrics.compute_metrics(y, baseline, scale))
            leaf["fold_cost_cr"].append(inventory.newsvendor_cost_per_unit_time(
                y, cal[leaf["service_quantile"]].to_numpy(), leaf["unit_stockout_cost"], leaf["unit_holding_cost"]))
            leaf["fold_cost_median"].append(inventory.newsvendor_cost_per_unit_time(
                y, point, leaf["unit_stockout_cost"], leaf["unit_holding_cost"]))

            rec = pd.DataFrame({
                "fold_id": fold.fold_id, "origin_date": leaf["series"].index[te - 1],
                "date": vmeta["target_date"].to_numpy(), "h": h, "y": y,
                "baseline": baseline, "mase_scale": scale,
            })
            for q in qs:
                rec[q] = cal[q].to_numpy()
            leaf["records"].append(rec)

    return {(sku, loc): _finalize_leaf(leaf) for loc, leaf in leaves.items()}


def _finalize_leaf(leaf: dict) -> dict:
    series = leaf["series"]
    records = pd.concat(leaf["records"], ignore_index=True) if leaf["records"] else pd.DataFrame()
    pm, bm = leaf["fold_point"], leaf["fold_baseline"]

    def mean(vals):
        return float(np.nanmean(vals)) if len(vals) else float("nan")

    last_forecast = last_actual = None
    if len(records):
        last = records[records["fold_id"] == records["fold_id"].max()].sort_values("h")
        last_forecast = last.set_index("date")[sorted(config.QUANTILES)]
        last_actual = last.set_index("date")["y"]

    cusum_result = drop_detection.detect_drops(series) if len(series) > 60 else None
    residuals = (
        (records["y"] - records[config.MEDIAN_Q]).set_axis(pd.DatetimeIndex(records["date"]))
        if len(records) else pd.Series(dtype=float)
    )
    return {
        "sku": leaf["sku"], "location": leaf["location"],
        "n_folds": len(pm),
        "mean_pinball_by_quantile": {q: (sum(v) / len(v) if v else float("nan")) for q, v in leaf["fold_pinball"].items()},
        "point_metrics": pm,
        "baseline_metrics": bm,
        "mean_mae": mean([m.mae for m in pm]),
        "mean_wape": mean([m.wape for m in pm]),
        "mean_bias": mean([m.bias for m in pm]),
        "mean_mase": mean([m.mase for m in pm]),
        "baseline_mean_mae": mean([m.mae for m in bm]),
        "baseline_mean_wape": mean([m.wape for m in bm]),
        "baseline_mean_bias": mean([m.bias for m in bm]),
        "baseline_mean_mase": mean([m.mase for m in bm]),
        # cost of stocking at the critical-ratio quantile (the cost-optimal policy) ...
        "mean_newsvendor_cost_per_day": mean(leaf["fold_cost_cr"]),
        # ... vs. naively stocking at the median forecast
        "mean_newsvendor_cost_per_day_median": mean(leaf["fold_cost_median"]),
        "unit_stockout_cost": leaf["unit_stockout_cost"],
        "unit_holding_cost": leaf["unit_holding_cost"],
        "critical_ratio": leaf["critical_ratio"],
        "service_quantile": leaf["service_quantile"],
        "avg_price": leaf["avg_price"],
        "val_records": records,
        "last_val_forecast": last_forecast,
        "last_val_actual": last_actual,
        "cusum_result": cusum_result,
        "raw_series": series,
        "residuals": residuals,
    }


def run_leaf_backtest(df: pd.DataFrame, sku: str, location: str, xgb_n_jobs: int | None = None) -> dict:
    """Walk-forward backtest for a single (sku, location) leaf. Calibration
    then rests only on that leaf's own residuals; prefer `run_all_leaves`
    / `run_sku_backtest` which pool residuals across a SKU's locations."""
    return run_sku_backtest(df, sku, [location], xgb_n_jobs)[(sku, location)]


def run_all_leaves(
    df: pd.DataFrame, max_leaves: int | None = None, n_jobs: int = 1
) -> dict[tuple[str, str], dict]:
    """Backtest every (sku, location) leaf.

    Work is split **per SKU** (its locations share conformal calibration
    residuals). `n_jobs`: 1 (default) runs sequentially (XGBoost uses all
    cores per model); n_jobs > 1 processes several SKUs at once via
    `multiprocessing` with each XGBoost model pinned to 1 thread so the
    workers don't oversubscribe the cores.
    """
    pairs = data_loader.list_sku_location_pairs(df)
    if max_leaves is not None:
        pairs = pairs[:max_leaves]

    by_sku: dict[str, list[str]] = {}
    for sku, loc in pairs:
        by_sku.setdefault(sku, []).append(loc)
    tasks = [(sku, sorted(locs)) for sku, locs in by_sku.items()]

    if n_jobs <= 1:
        outputs = []
        for sku, locs in tasks:
            logger.info("Backtesting %s (%d locations)...", sku, len(locs))
            outputs.append(run_sku_backtest(df, sku, locs))
    else:
        import multiprocessing as mp

        n_jobs = min(n_jobs, mp.cpu_count(), len(tasks))
        logger.info("Backtesting %d SKUs across %d processes...", len(tasks), n_jobs)
        with mp.Pool(processes=n_jobs, initializer=_init_worker, initargs=(df,)) as pool:
            outputs = pool.map(_worker, tasks)

    merged: dict[tuple[str, str], dict] = {}
    for out in outputs:
        merged.update(out)
    return {pair: merged[pair] for pair in pairs}


# Module-level globals + helpers for multiprocessing.Pool — the pool's
# initializer sets `_WORKER_DF` once per worker process (not once per
# task), so the dataframe isn't re-serialized for every SKU.
_WORKER_DF: pd.DataFrame | None = None


def _init_worker(df: pd.DataFrame) -> None:
    global _WORKER_DF
    _WORKER_DF = df


def _worker(task: tuple[str, list[str]]) -> dict:
    sku, locations = task
    return run_sku_backtest(_WORKER_DF, sku, locations, xgb_n_jobs=1)


# ---------------------------------------------------------------------------
# Latest forecast -> joint scenarios -> aggregates and inventory plans
# ---------------------------------------------------------------------------
def _forecast_matrix(leaf_result: dict) -> np.ndarray | None:
    f = leaf_result["last_val_forecast"]
    if f is None or len(f) < config.HORIZON_DAYS:
        return None
    return f[sorted(config.QUANTILES)].to_numpy()


def build_latest_scenarios(
    leaf_results: dict[tuple[str, str], dict],
    n_samples: int = config.N_SCENARIO_SAMPLES,
    seed: int = config.SCENARIO_SEED,
    lead_time_days: float = config.DEFAULT_LEAD_TIME_DAYS,
    lead_time_std_days: float = config.DEFAULT_LEAD_TIME_STD_DAYS,
    service_level: float = config.DEFAULT_SERVICE_LEVEL,
) -> dict:
    """Joint sample paths for the latest fold's forecast, with SKU /
    Location / Total aggregates and safety-stock plans.

    The dependence structure is estimated from all backtest folds *except*
    the latest one, so it isn't fitted on the data it is applied to.

    Returns a dict with:
      leaves, samples (n_samples, n_leaves, H), dependence,
      aggregates ({key: (n_samples, H)}),
      aggregate_summary (DataFrame: P10/P50/P90 of horizon-total demand per
        aggregate, vs. the *wrong* sum-of-leaf-quantiles for comparison),
      plans ({leaf or aggregate key: LeadTimePlan}),
      risk_pooling (dict comparing summed leaf safety stock to the pooled
        safety stock at Total / SKU / Location level).
    """
    levels = sorted(config.QUANTILES)
    usable = {leaf: r for leaf, r in leaf_results.items() if _forecast_matrix(r) is not None}
    if not usable:
        raise ValueError("No leaf has a complete latest forecast; cannot build scenarios.")

    last_fold = max(int(r["val_records"]["fold_id"].max()) for r in usable.values())
    dependence = scenarios.estimate_dependence(
        {leaf: r["val_records"] for leaf, r in usable.items()}, levels, exclude_fold=last_fold,
    )
    forecasts = {leaf: _forecast_matrix(usable[leaf]) for leaf in dependence.leaves}
    samples = scenarios.sample_joint_paths(forecasts, dependence, levels, n_samples, seed)
    aggregates = hierarchy.aggregate_sample_paths(samples, dependence.leaves)

    lead_times = scenarios.draw_lead_times(n_samples, lead_time_days, lead_time_std_days, config.HORIZON_DAYS, seed)

    def plan(paths: np.ndarray) -> inventory.LeadTimePlan:
        ltd = scenarios.lead_time_demand(paths, lead_times)
        return inventory.safety_stock_from_lead_time_samples(ltd, service_level, lead_time_days, lead_time_std_days)

    plans: dict = {leaf: plan(samples[:, i, :]) for i, leaf in enumerate(dependence.leaves)}
    plans.update({key: plan(paths) for key, paths in aggregates.items()})

    band = (config.BAND_LOW_Q, config.MEDIAN_Q, config.BAND_HIGH_Q)
    rows = []
    leaf_totals = samples.sum(axis=2)  # (n_samples, n_leaves) horizon-total per leaf
    for key, paths in aggregates.items():
        if key == "total":
            members = list(range(len(dependence.leaves)))
        elif key.startswith("sku:"):
            members = [i for i, (s, _) in enumerate(dependence.leaves) if s == key[4:]]
        else:
            members = [i for i, (_, l) in enumerate(dependence.leaves) if l == key[9:]]
        totals = paths.sum(axis=1)
        row = {"aggregate": key, "n_leaves": len(members)}
        for q in band:
            row[f"P{q * 100:g}"] = float(np.quantile(totals, q))
        # What you'd get by (incorrectly) adding leaf quantiles:
        row[f"naive_sum_of_leaf_P{band[2] * 100:g}"] = float(sum(np.quantile(leaf_totals[:, i], band[2]) for i in members))
        row["pooled_safety_stock"] = plans[key].safety_stock
        row["sum_leaf_safety_stock"] = float(sum(plans[dependence.leaves[i]].safety_stock for i in members))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["risk_pooling_saving"] = 1 - summary["pooled_safety_stock"] / summary["sum_leaf_safety_stock"].replace(0, np.nan)

    return {
        "leaves": dependence.leaves, "samples": samples, "dependence": dependence,
        "aggregates": aggregates, "aggregate_summary": summary, "plans": plans,
        "risk_pooling": summary.set_index("aggregate")["risk_pooling_saving"].to_dict(),
        "lead_times": lead_times,
    }


def reconcile_latest_forecasts(leaf_results: dict[tuple[str, str], dict]) -> dict[str, pd.Series]:
    """Bottom-up aggregate each leaf's last-fold *median* forecast (a
    coherent point forecast; aggregate quantiles come from
    `build_latest_scenarios`, not from this)."""
    leaf_medians = {}
    for (sku, location), result in leaf_results.items():
        forecast = result["last_val_forecast"]
        if forecast is None:
            continue
        leaf_medians[(sku, location)] = forecast[config.MEDIAN_Q]
    return hierarchy.aggregate_leaf_forecasts(leaf_medians)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def aggregate_diagnostics(leaf_results: dict[tuple[str, str], dict]) -> dict:
    """Pool residuals, quantile calibration, and per-leaf metrics across
    the whole portfolio — the basis for the calibration/residual/
    robustness figures and the model-vs-baseline table.
    """
    recs = [r["val_records"].assign(sku=r["sku"], location=r["location"])
            for r in leaf_results.values() if len(r["val_records"])]
    all_records = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()

    all_residuals = pd.concat(
        [r["residuals"] for r in leaf_results.values() if len(r["residuals"])], axis=0
    ) if leaf_results else pd.Series(dtype=float)

    # Reliability table: for each nominal quantile q, the empirical
    # fraction of (actual <= predicted_q) across every fold of every leaf.
    # A well-calibrated model has empirical ~= nominal for every q.
    reliability_rows = []
    for q in sorted(config.QUANTILES):
        flags = (all_records["y"] <= all_records[q]) if len(all_records) else pd.Series(dtype=bool)
        reliability_rows.append({
            "nominal_quantile": q,
            "empirical_fraction_below": float(flags.mean()) if len(flags) else float("nan"),
            "n_obs": int(len(flags)),
        })
    reliability_table = pd.DataFrame(reliability_rows)

    # Calibration by horizon bucket and coverage of the P10-P90 band by horizon.
    by_bucket_rows, coverage_rows = [], []
    if len(all_records):
        bucket = probabilistic.horizon_bucket(all_records["h"].to_numpy())
        for b, (lo, hi) in enumerate(config.HORIZON_BUCKETS):
            sub = all_records[bucket == b]
            for q in sorted(config.QUANTILES):
                by_bucket_rows.append({
                    "horizon_bucket": f"h{lo}-{hi}", "nominal_quantile": q,
                    "empirical_fraction_below": float((sub["y"] <= sub[q]).mean()), "n_obs": len(sub),
                })
        for h, sub in all_records.groupby("h"):
            inside = (sub["y"] >= sub[config.BAND_LOW_Q]) & (sub["y"] <= sub[config.BAND_HIGH_Q])
            coverage_rows.append({
                "h": int(h), "p10_p90_coverage": float(inside.mean()),
                "nominal_coverage": config.BAND_HIGH_Q - config.BAND_LOW_Q, "n_obs": len(sub),
            })

    leaf_summary_rows, pinball_rows = [], []
    for (sku, location), r in leaf_results.items():
        leaf_summary_rows.append({
            "sku": sku, "location": location, "n_folds": r["n_folds"],
            "mean_mae": r["mean_mae"], "mean_wape": r["mean_wape"], "mean_bias": r["mean_bias"],
            "mean_mase": r["mean_mase"],
            "baseline_mean_mae": r["baseline_mean_mae"], "baseline_mean_wape": r["baseline_mean_wape"],
            "baseline_mean_mase": r["baseline_mean_mase"],
            "mean_cost_per_day": r["mean_newsvendor_cost_per_day"],
            "mean_cost_per_day_median": r["mean_newsvendor_cost_per_day_median"],
            "critical_ratio": r["critical_ratio"], "service_quantile": r["service_quantile"],
            "avg_price": r["avg_price"],
        })
        for q, val in r["mean_pinball_by_quantile"].items():
            pinball_rows.append({"sku": sku, "location": location, "quantile": q, "pinball_loss": val})
    leaf_summary = pd.DataFrame(leaf_summary_rows)

    # Portfolio-level model vs. seasonal-naive benchmark (volume-weighted).
    accuracy_rows = []
    if len(all_records):
        for name, pred in (("Quantile XGBoost (P50, calibrated)", all_records[config.MEDIAN_Q]),
                           ("Seasonal naive (same weekday)", all_records["baseline"])):
            y = all_records["y"].to_numpy()
            mase_vals = (np.abs(y - pred.to_numpy()) / all_records["mase_scale"].replace(0, np.nan).to_numpy())
            accuracy_rows.append({
                "forecast": name,
                "WAPE": eval_metrics.wape(y, pred), "Bias": eval_metrics.bias(y, pred),
                "MASE": float(np.nanmean(mase_vals)), "n_obs": len(y),
            })
    portfolio_accuracy = pd.DataFrame(accuracy_rows)
    beats = float((leaf_summary["mean_mae"] < leaf_summary["baseline_mean_mae"]).mean()) if len(leaf_summary) else float("nan")

    return {
        "all_records": all_records,
        "all_residuals": all_residuals,
        "reliability_table": reliability_table,
        "reliability_by_horizon_bucket": pd.DataFrame(by_bucket_rows),
        "coverage_by_horizon": pd.DataFrame(coverage_rows),
        "leaf_summary": leaf_summary,
        "pinball_by_leaf_and_quantile": pd.DataFrame(pinball_rows),
        "portfolio_accuracy": portfolio_accuracy,
        "share_leaves_beating_baseline_mae": beats,
    }
