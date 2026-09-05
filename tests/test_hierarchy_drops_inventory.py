import numpy as np
import pandas as pd

from src import drop_detection, hierarchy, inventory


def test_bottom_up_reconciliation_is_coherent():
    idx = pd.date_range("2024-01-01", periods=14, freq="D")
    leaf_forecasts = {
        ("SKU-A", "LOC-1"): pd.Series(np.arange(14, dtype=float), index=idx),
        ("SKU-A", "LOC-2"): pd.Series(np.arange(14, dtype=float) * 2, index=idx),
        ("SKU-B", "LOC-1"): pd.Series(np.arange(14, dtype=float) * 0.5, index=idx),
    }
    aggregates = hierarchy.aggregate_leaf_forecasts(leaf_forecasts)

    sku_sum = aggregates["sku:SKU-A"] + aggregates["sku:SKU-B"]
    location_sum = aggregates["location:LOC-1"] + aggregates["location:LOC-2"]

    pd.testing.assert_series_equal(aggregates["total"], sku_sum, check_names=False)
    pd.testing.assert_series_equal(aggregates["total"], location_sum, check_names=False)


def test_cusum_flags_injected_drop():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=150, freq="D")
    values = rng.normal(100, 5, size=150)
    # Inject a sustained drop from day 80 to day 95.
    values[80:95] -= 40
    series = pd.Series(values, index=idx)

    result = drop_detection.detect_drops(series, reference_window=60)

    assert len(result.alarm_days) > 0
    # At least one alarm should fall within (or shortly after) the injected drop window.
    assert any(idx[80] <= d <= idx[100] for d in result.alarm_days)


def test_cusum_no_alarms_on_stable_series():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=150, freq="D")
    series = pd.Series(rng.normal(100, 5, size=150), index=idx)

    result = drop_detection.detect_drops(series, reference_window=60)
    assert len(result.alarm_days) == 0


def test_safety_stock_increases_with_service_level():
    plan_low = inventory.compute_safety_stock(median_daily_demand=100, daily_sigma=10, lead_time_days=7, service_level=0.80)
    plan_high = inventory.compute_safety_stock(median_daily_demand=100, daily_sigma=10, lead_time_days=7, service_level=0.99)
    assert plan_high.safety_stock > plan_low.safety_stock


def test_newsvendor_cost_penalizes_stockout_more_when_configured():
    y_true = [100]
    y_pred = [80]  # under-forecast by 20 -> stockout
    cost = inventory.newsvendor_cost(y_true, y_pred, unit_stockout_cost=10, unit_holding_cost=1)
    assert cost == 200  # 20 units short * $10
