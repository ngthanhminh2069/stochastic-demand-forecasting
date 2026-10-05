"""Tests for the leakage-free features, conformal calibration, scenario
generation / risk pooling, and the margin-based cost model."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src import config, features, hierarchy, inventory, probabilistic, scenarios

LEVELS = sorted(config.QUANTILES)


def _toy_leaf(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    y = pd.Series(rng.poisson(50, n).astype(float), index=idx)
    extra = pd.DataFrame({"promotion": rng.integers(0, 2, n), config.PRICE_COL: 10.0}, index=idx)
    return y, extra


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------
def test_features_ignore_demand_after_the_origin():
    """Corrupting every demand value after the origin must not change a
    single feature: the model may only see what is known at the origin."""
    y, extra = _toy_leaf()
    origin = 120
    horizons = np.arange(1, config.HORIZON_DAYS + 1)

    X1, m1 = features.make_direct_frame(y, extra, [origin], horizons)

    y_corrupt = y.copy()
    y_corrupt.iloc[origin + 1:] = 1e9
    X2, m2 = features.make_direct_frame(y_corrupt, extra, [origin], horizons)

    pd.testing.assert_frame_equal(X1, X2)
    assert (m1["origin_pos"] == origin).all()
    assert len(X1) == config.HORIZON_DAYS


def test_seasonal_lag_is_most_recent_known_same_weekday():
    y, extra = _toy_leaf()
    origin = 100
    horizons = np.arange(1, config.HORIZON_DAYS + 1)
    X, meta = features.make_direct_frame(y, extra, [origin], horizons)
    naive = features.seasonal_naive_forecast(y.to_numpy(), origin, horizons)
    np.testing.assert_array_equal(X["seasonal_lag_1"].to_numpy(), naive)
    # same weekday as the target, and never later than the origin
    target_dates = y.index[meta["target_pos"].to_numpy()]
    lag_dates = y.index[meta["target_pos"].to_numpy() - 7 * np.ceil(horizons / 7).astype(int)]
    assert (target_dates.dayofweek == lag_dates.dayofweek).all()
    assert (lag_dates <= y.index[origin]).all()


# ---------------------------------------------------------------------------
# Conformal calibration
# ---------------------------------------------------------------------------
def test_conformal_offsets_repair_a_biased_model():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.normal(100, 10, n)
    h = rng.integers(1, config.HORIZON_DAYS + 1, n)
    true_q = {q: 100 + 10 * norm.ppf(q) for q in LEVELS}
    biased = pd.DataFrame({q: np.full(n, true_q[q] - 5.0) for q in LEVELS})  # every quantile 5 too low

    offsets = probabilistic.conformal_offsets(probabilistic.calibration_residuals(biased, y, h))
    fixed = probabilistic.apply_conformal_offsets(biased, h, offsets)

    for q in LEVELS:
        assert abs(np.mean(y <= fixed[q].to_numpy()) - q) < 0.03
        assert abs(offsets[0][q] - 5.0) < 1.0


def test_calibrated_quantiles_stay_monotone_and_non_negative():
    preds = pd.DataFrame({q: np.array([2.0, 1.0]) + i for i, q in enumerate(LEVELS)})
    out = probabilistic.apply_conformal_offsets(preds, np.array([1, 14]), {0: {q: -10.0 for q in LEVELS}, 1: {q: 0.0 for q in LEVELS}})
    assert (out.to_numpy() >= 0).all()
    assert (np.diff(out.to_numpy(), axis=1) >= 0).all()


# ---------------------------------------------------------------------------
# Scenarios and risk pooling
# ---------------------------------------------------------------------------
def _identity_dependence(leaves, H=config.HORIZON_DAYS, rho=0.0):
    steps = np.arange(H)
    corr_h = rho ** np.abs(steps[:, None] - steps[None, :]) if rho else np.eye(H)
    return scenarios.Dependence(
        leaves=leaves, chol_leaf=np.eye(len(leaves)), chol_h=np.linalg.cholesky(corr_h),
        rho_h=rho, mean_leaf_corr=0.0,
    )


def _normal_forecast(mu, sd, H=config.HORIZON_DAYS):
    row = np.array([mu + sd * norm.ppf(q) for q in LEVELS])
    return np.tile(row, (H, 1))


def test_quantile_curve_reproduces_forecast_quantiles():
    q = np.array([mu for mu in _normal_forecast(100, 10)[0]])
    gl, gv = scenarios.quantile_curve(q, LEVELS)
    for lv, val in zip(LEVELS, q):
        assert np.interp(lv, gl, gv) == pytest.approx(val, abs=1e-6)
    assert (np.diff(gv) > 0).all()


def test_sum_of_leaf_quantiles_overstates_aggregate_uncertainty():
    leaves = [("A", "L1"), ("A", "L2"), ("B", "L1"), ("B", "L2")]
    forecasts = {leaf: _normal_forecast(100, 20) for leaf in leaves}
    dep = _identity_dependence(leaves)
    samples = scenarios.sample_joint_paths(forecasts, dep, LEVELS, n_samples=4000, seed=1)
    aggs = hierarchy.aggregate_sample_paths(samples, leaves)

    leaf_p90_sum = sum(np.quantile(samples[:, i, 0], 0.9) for i in range(len(leaves)))
    total_p90 = np.quantile(aggs["total"][:, 0], 0.9)
    assert total_p90 < leaf_p90_sum - 10      # independent leaves: aggregate band is much narrower
    # ... and matches the analytic answer for a sum of independent normals
    expected = 4 * 100 + norm.ppf(0.9) * 20 * np.sqrt(4)
    assert total_p90 == pytest.approx(expected, rel=0.02)


def test_perfectly_correlated_leaves_recover_the_naive_sum():
    leaves = [("A", "L1"), ("A", "L2")]
    forecasts = {leaf: _normal_forecast(100, 20) for leaf in leaves}
    n = len(leaves)
    dep = scenarios.Dependence(
        leaves=leaves, chol_leaf=np.linalg.cholesky(np.full((n, n), 0.999999) + np.eye(n) * 1e-6),
        chol_h=np.eye(config.HORIZON_DAYS), rho_h=0.0, mean_leaf_corr=1.0,
    )
    samples = scenarios.sample_joint_paths(forecasts, dep, LEVELS, n_samples=4000, seed=2)
    total_p90 = np.quantile(samples.sum(axis=1)[:, 0], 0.9)
    assert total_p90 == pytest.approx(2 * (100 + norm.ppf(0.9) * 20), rel=0.02)


def test_aggregates_are_coherent_for_every_sample():
    leaves = [("A", "L1"), ("A", "L2"), ("B", "L1")]
    forecasts = {leaf: _normal_forecast(50 + 10 * i, 5) for i, leaf in enumerate(leaves)}
    samples = scenarios.sample_joint_paths(forecasts, _identity_dependence(leaves), LEVELS, 500, seed=3)
    aggs = hierarchy.aggregate_sample_paths(samples, leaves)
    np.testing.assert_allclose(aggs["total"], aggs["sku:A"] + aggs["sku:B"])
    np.testing.assert_allclose(aggs["total"], aggs["location:L1"] + aggs["location:L2"])


def test_estimate_dependence_detects_common_shocks():
    """Leaves that share a demand shock must come out positively correlated."""
    rng = np.random.default_rng(0)
    folds, hs = 40, np.arange(1, config.HORIZON_DAYS + 1)
    common = rng.standard_normal((folds, len(hs)))
    recs = {}
    for k, leaf in enumerate([("A", "L1"), ("A", "L2"), ("B", "L1")]):
        z = 0.8 * common + 0.6 * rng.standard_normal((folds, len(hs)))
        rows = []
        for f in range(folds):
            for j, h in enumerate(hs):
                qv = [100 + 10 * norm.ppf(q) for q in LEVELS]
                rows.append({"fold_id": f, "h": h, "y": 100 + 10 * z[f, j], **dict(zip(LEVELS, qv))})
        recs[leaf] = pd.DataFrame(rows)
    dep = scenarios.estimate_dependence(recs, LEVELS)
    assert dep.mean_leaf_corr > 0.3


def test_lead_time_demand_and_safety_stock_match_normal_theory():
    H, L, sigma, mu = config.HORIZON_DAYS, 7, 20.0, 100.0
    forecasts = {("A", "L1"): _normal_forecast(mu, sigma)}
    dep = _identity_dependence(list(forecasts))
    samples = scenarios.sample_joint_paths(forecasts, dep, LEVELS, n_samples=20000, seed=4)[:, 0, :]
    lead_times = scenarios.draw_lead_times(20000, L, 0.0, H)
    ltd = scenarios.lead_time_demand(samples, lead_times)
    plan = inventory.safety_stock_from_lead_time_samples(ltd, 0.95, L, 0.0)
    expected_ss = norm.ppf(0.95) * sigma * np.sqrt(L)
    assert plan.expected_demand_over_lead_time == pytest.approx(mu * L, rel=0.02)
    assert plan.safety_stock == pytest.approx(expected_ss, rel=0.1)
    assert plan.reorder_point == pytest.approx(plan.expected_demand_over_lead_time + plan.safety_stock)


def test_random_lead_time_increases_safety_stock():
    H = config.HORIZON_DAYS
    forecasts = {("A", "L1"): _normal_forecast(100, 20)}
    samples = scenarios.sample_joint_paths(forecasts, _identity_dependence(list(forecasts)), LEVELS, 20000, seed=5)[:, 0, :]
    fixed = inventory.safety_stock_from_lead_time_samples(
        scenarios.lead_time_demand(samples, scenarios.draw_lead_times(20000, 7, 0.0, H)), 0.95)
    random = inventory.safety_stock_from_lead_time_samples(
        scenarios.lead_time_demand(samples, scenarios.draw_lead_times(20000, 7, 2.0, H)), 0.95)
    assert random.safety_stock > fixed.safety_stock


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def test_stockout_cost_is_margin_not_full_price():
    cu, co = inventory.derive_costs_from_price(100.0, gross_margin=0.3, annual_holding_rate=0.25, review_period_days=7)
    assert cu == pytest.approx(30.0)
    assert co == pytest.approx(0.25 * 100 * 7 / 365)


def test_critical_ratio_is_capped_and_independent_of_price():
    r_cheap = inventory.critical_ratio(*inventory.derive_costs_from_price(10.0))
    r_dear = inventory.critical_ratio(*inventory.derive_costs_from_price(120.0))
    assert r_cheap == pytest.approx(r_dear)                     # both costs scale with price
    assert r_cheap <= config.MAX_CRITICAL_RATIO
    assert inventory.critical_ratio(100.0, 1.0, max_ratio=0.9) == 0.9
    assert inventory.critical_ratio(10.0, 10.0) == pytest.approx(0.5)
    assert inventory.select_service_quantile(0.98) == max(config.QUANTILES)


def test_critical_ratio_quantile_minimises_newsvendor_cost():
    rng = np.random.default_rng(0)
    y = rng.normal(100, 20, 50000).clip(0)
    cu, co = 30.0, 3.0
    ratio = inventory.critical_ratio(cu, co)
    best = np.quantile(y, ratio)
    cost_best = inventory.newsvendor_cost(y, np.full_like(y, best), cu, co)
    cost_median = inventory.newsvendor_cost(y, np.full_like(y, np.median(y)), cu, co)
    assert cost_best < cost_median
