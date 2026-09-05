"""Hierarchical reconciliation for SKU x Location forecasts.

The hierarchy has three levels:

    Total
    ├── SKU (summed across locations)
    ├── Location (summed across SKUs)
    └── SKU x Location (leaf/base level)

We reconcile with **bottom-up aggregation**: forecast each leaf
(SKU x Location) independently, then sum leaves to get SKU-level,
Location-level, and Total forecasts. This guarantees coherence (children
always sum exactly to their parent) by construction, with no extra
estimation step.

This is a deliberate simplification vs. top-down or MinT (trace minimization)
reconciliation, which can produce more statistically efficient forecasts by
using information from every level — but require either historical
proportions (top-down) or a full forecast-error covariance matrix (MinT).
Bottom-up is the right default here because: (1) leaf-level forecasts are
what actually drive SKU x Location inventory decisions, so there's no
accuracy to "borrow" from higher levels that matters more than getting the
leaves right, and (2) it stays fully transparent — every aggregate number
is traceable to a leaf forecast, which matters for stakeholder trust in a
supply-chain context.
"""
from __future__ import annotations

import pandas as pd

from src import config


def aggregate_leaf_forecasts(
    leaf_forecasts: dict[tuple[str, str], pd.Series],
) -> dict[str, pd.Series]:
    """Bottom-up aggregate a dict of {(sku, location): forecast_series} into
    per-SKU, per-Location, and Total forecast series — all guaranteed
    coherent with the leaves by construction.
    """
    sku_level: dict[str, pd.Series] = {}
    location_level: dict[str, pd.Series] = {}
    total_level: pd.Series | None = None

    for (sku, location), series in leaf_forecasts.items():
        sku_level[sku] = series if sku not in sku_level else sku_level[sku].add(series, fill_value=0)
        location_level[location] = (
            series if location not in location_level else location_level[location].add(series, fill_value=0)
        )
        total_level = series if total_level is None else total_level.add(series, fill_value=0)

    return {
        "total": total_level,
        **{f"sku:{k}": v for k, v in sku_level.items()},
        **{f"location:{k}": v for k, v in location_level.items()},
    }


def build_hierarchy_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small reference table of every (sku, location) leaf and its
    parent SKU/Location aggregates — useful for sanity-checking coherence."""
    return df[[config.SKU_COL, config.LOCATION_COL]].drop_duplicates().reset_index(drop=True)
