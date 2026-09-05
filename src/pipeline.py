"""End-to-end SKU x Location pipeline (real dataset only — no synthetic
data anywhere in this project):

1. Load the real SKU x Location demand panel (prepared via
   src/prepare_real_data.py).
2. For each leaf, run a walk-forward backtest of conformal-calibrated
   quantile XGBoost, scored with pinball loss (distributional), MAE/RMSE
   (point, via the median), and newsvendor cost (business — using
   per-SKU costs derived from that SKU's average price).
3. Reconcile leaf forecasts bottom-up into SKU-level, Location-level and
   Total forecasts.
4. Run CUSUM drop detection per leaf.
5. Size a safety stock / reorder point from the most recent quantile
   forecast.
6. Accumulate residuals and P10-P90 coverage across every fold of every
   leaf, so the pipeline can report whether the model is actually
   well-calibrated (not just "accurate on average") — see
   `aggregate_diagnostics` and `src/plotting.py`.
"""
from __future__ import annotations

import logging

import pandas as pd

from src import backtest, config, data_loader, drop_detection, eval_metrics, features, hierarchy, inventory, probabilistic

logger = logging.getLogger(__name__)


def load_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the real SKU x Location dataset. Raises FileNotFoundError with
    setup instructions if it hasn't been prepared yet — this project does
    not fall back to synthetic data."""
    return data_loader.load_raw_data(path)


def run_leaf_backtest(df: pd.DataFrame, sku: str, location: str, xgb_n_jobs: int | None = None) -> dict:
    """Walk-forward backtest for a single (sku, location) leaf.

    `xgb_n_jobs`: forwarded to `probabilistic.fit_quantile_models`. Leave
    as None for a normal sequential run (XGBoost uses all cores per
    model). `run_all_leaves(..., n_jobs=N)` sets this to 1 automatically
    when N > 1, so leaf-level process parallelism and XGBoost's own
    thread parallelism don't fight over the same CPU cores.
    """
    series = data_loader.get_series(df, sku, location)
    leaf_rows = df[(df[config.SKU_COL] == sku) & (df[config.LOCATION_COL] == location)]
    extra = leaf_rows.set_index(config.DATE_COL)[["promotion", config.PRICE_COL]].sort_index()

    avg_price = leaf_rows[config.PRICE_COL].mean()
    unit_stockout_cost, unit_holding_cost = inventory.derive_costs_from_price(avg_price)

    full_features = features.build_features(series, extra)
    folds = backtest.generate_folds(series)

    fold_pinball: dict[float, list[float]] = {q: [] for q in config.QUANTILES}
    fold_point_metrics: list[eval_metrics.ForecastMetrics] = []
    fold_cost: list[float] = []
    last_val_forecast: pd.DataFrame | None = None
    last_val_actual: pd.Series | None = None

    residual_chunks: list[pd.Series] = []  # actual - median_pred, per fold, concatenated
    # For each quantile q, track whether actual <= predicted q on each val
    # point. If the model is well-calibrated, the empirical fraction where
    # this holds should be close to q itself for every q — this is what a
    # reliability diagram checks (a much stronger test than "coverage
    # looks about right on average").
    below_flags: dict[float, list[bool]] = {q: [] for q in config.QUANTILES}

    median_q = sorted(config.QUANTILES)[len(config.QUANTILES) // 2]

    for fold in folds:
        X_train_full, X_val, y_train_full, y_val = features.split_features_target(
            full_features, fold.train_end
        )
        X_val = X_val[X_val.index < fold.val_end]
        y_val = y_val.loc[X_val.index]
        if len(X_val) == 0 or len(X_train_full) < config.CONFORMAL_CALIB_DAYS + 30:
            continue

        calib_start = fold.train_end - pd.Timedelta(days=config.CONFORMAL_CALIB_DAYS)
        X_calib = X_train_full[X_train_full.index >= calib_start]
        y_calib = y_train_full.loc[X_calib.index]
        X_train = X_train_full[X_train_full.index < calib_start]
        y_train = y_train_full.loc[X_train.index]

        models = probabilistic.fit_quantile_models(X_train, y_train, n_jobs=xgb_n_jobs)
        offsets = probabilistic.conformal_calibrate(models, X_calib, y_calib)

        raw_val_preds = probabilistic.predict_quantiles(models, X_val)
        calibrated_val_preds = probabilistic.apply_conformal_offsets(raw_val_preds, offsets)

        for q in config.QUANTILES:
            fold_pinball[q].append(probabilistic.pinball_loss(y_val, calibrated_val_preds[q], q))

        point_pred = calibrated_val_preds[median_q]
        fold_point_metrics.append(eval_metrics.compute_metrics(y_val, point_pred))
        fold_cost.append(
            inventory.newsvendor_cost_per_unit_time(
                y_val, point_pred, unit_stockout_cost, unit_holding_cost
            )
        )

        residual_chunks.append(y_val - point_pred)
        for q in config.QUANTILES:
            below_flags[q].extend((y_val.values <= calibrated_val_preds[q].values).tolist())

        last_val_forecast = calibrated_val_preds
        last_val_actual = y_val

    cusum_result = None
    if len(series) > 60:
        cusum_result = drop_detection.detect_drops(series)

    return {
        "sku": sku,
        "location": location,
        "n_folds": len(fold_point_metrics),
        "mean_pinball_by_quantile": {q: (sum(v) / len(v) if v else float("nan")) for q, v in fold_pinball.items()},
        "point_metrics": fold_point_metrics,
        "mean_mae": (sum(m.mae for m in fold_point_metrics) / len(fold_point_metrics)) if fold_point_metrics else float("nan"),
        "mean_newsvendor_cost_per_day": (sum(fold_cost) / len(fold_cost)) if fold_cost else float("nan"),
        "unit_stockout_cost": unit_stockout_cost,
        "unit_holding_cost": unit_holding_cost,
        "avg_price": avg_price,
        "last_val_forecast": last_val_forecast,
        "last_val_actual": last_val_actual,
        "cusum_result": cusum_result,
        "raw_series": series,
        "residuals": pd.concat(residual_chunks) if residual_chunks else pd.Series(dtype=float),
        "below_flags": below_flags,
    }


def build_safety_stock_plan(leaf_result: dict) -> inventory.SafetyStockPlan | None:
    forecast = leaf_result["last_val_forecast"]
    if forecast is None or 0.1 not in forecast.columns or 0.9 not in forecast.columns:
        return None
    latest = forecast.iloc[-1]
    sigma = inventory.estimate_daily_sigma_from_quantiles(latest[0.1], latest[0.9])
    median_col = 0.5 if 0.5 in forecast.columns else sorted(forecast.columns)[len(forecast.columns) // 2]
    return inventory.compute_safety_stock(median_daily_demand=latest[median_col], daily_sigma=sigma)


def run_all_leaves(
    df: pd.DataFrame, max_leaves: int | None = None, n_jobs: int = 1
) -> dict[tuple[str, str], dict]:
    """Backtest every (sku, location) leaf.

    `n_jobs`: 1 (default) runs sequentially — safest, simplest, and
    XGBoost itself uses all CPU cores per model. Pass n_jobs > 1 to
    process multiple leaves at once via `multiprocessing` (CPU cores,
    not GPU/CUDA — each leaf's data is small and the 100 leaves are
    fully independent of each other, so process-level CPU parallelism is
    the right lever here). When n_jobs > 1, each XGBoost model is pinned
    to 1 thread (see `run_leaf_backtest`'s `xgb_n_jobs`) so the worker
    processes don't oversubscribe the same cores XGBoost would otherwise
    try to use internally.
    """
    pairs = data_loader.list_sku_location_pairs(df)
    if max_leaves is not None:
        pairs = pairs[:max_leaves]

    if n_jobs <= 1:
        results = {}
        for sku, location in pairs:
            logger.info("Backtesting %s x %s...", sku, location)
            results[(sku, location)] = run_leaf_backtest(df, sku, location)
        return results

    import multiprocessing as mp

    n_jobs = min(n_jobs, mp.cpu_count(), len(pairs))
    logger.info("Backtesting %d leaves across %d processes...", len(pairs), n_jobs)

    with mp.Pool(processes=n_jobs, initializer=_init_worker, initargs=(df,)) as pool:
        outputs = pool.map(_worker, pairs)

    return dict(zip(pairs, outputs))


# Module-level globals + helpers for multiprocessing.Pool — the pool's
# initializer sets `_WORKER_DF` once per worker process (not once per
# leaf/task), so the ~6MB dataframe isn't re-serialized 100 times.
_WORKER_DF: pd.DataFrame | None = None


def _init_worker(df: pd.DataFrame) -> None:
    global _WORKER_DF
    _WORKER_DF = df


def _worker(pair: tuple[str, str]) -> dict:
    sku, location = pair
    return run_leaf_backtest(_WORKER_DF, sku, location, xgb_n_jobs=1)


def reconcile_latest_forecasts(leaf_results: dict[tuple[str, str], dict]) -> dict[str, pd.Series]:
    """Bottom-up aggregate each leaf's last-fold median forecast."""
    leaf_medians = {}
    for (sku, location), result in leaf_results.items():
        forecast = result["last_val_forecast"]
        if forecast is None:
            continue
        median_col = 0.5 if 0.5 in forecast.columns else sorted(forecast.columns)[len(forecast.columns) // 2]
        leaf_medians[(sku, location)] = forecast[median_col]
    return hierarchy.aggregate_leaf_forecasts(leaf_medians)


def aggregate_diagnostics(leaf_results: dict[tuple[str, str], dict]) -> dict:
    """Pool residuals, quantile calibration, and per-leaf metrics across
    the whole portfolio — the basis for the calibration/residual/
    robustness figures.
    """
    all_residuals = pd.concat(
        [r["residuals"] for r in leaf_results.values() if len(r["residuals"])], axis=0
    ) if leaf_results else pd.Series(dtype=float)

    # Reliability table: for each nominal quantile q, the empirical
    # fraction of (actual <= predicted_q) across every fold of every leaf.
    # A well-calibrated model has empirical ≈ nominal for every q.
    reliability_rows = []
    for q in config.QUANTILES:
        flags: list[bool] = []
        for r in leaf_results.values():
            flags.extend(r["below_flags"].get(q, []))
        empirical = (sum(flags) / len(flags)) if flags else float("nan")
        reliability_rows.append({"nominal_quantile": q, "empirical_fraction_below": empirical, "n_obs": len(flags)})
    reliability_table = pd.DataFrame(reliability_rows)

    leaf_summary_rows = []
    pinball_rows = []
    for (sku, location), r in leaf_results.items():
        leaf_summary_rows.append({
            "sku": sku, "location": location, "n_folds": r["n_folds"],
            "mean_mae": r["mean_mae"], "mean_cost_per_day": r["mean_newsvendor_cost_per_day"],
            "avg_price": r["avg_price"],
        })
        for q, val in r["mean_pinball_by_quantile"].items():
            pinball_rows.append({"sku": sku, "location": location, "quantile": q, "pinball_loss": val})

    return {
        "all_residuals": all_residuals,
        "reliability_table": reliability_table,
        "leaf_summary": pd.DataFrame(leaf_summary_rows),
        "pinball_by_leaf_and_quantile": pd.DataFrame(pinball_rows),
    }
