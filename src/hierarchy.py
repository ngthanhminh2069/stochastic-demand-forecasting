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

**Point forecasts vs. distributions.** Summing is exact for *means/sums of
samples*, but **not for quantiles**: the sum of leaf P90s is not the P90 of
the total. Aggregate quantiles are therefore computed from **joint sample
paths** (see `src/scenarios.py`), summed leaf-by-leaf, and only then
summarised into quantiles (`aggregate_sample_paths`, `summarize_paths`).
Because the paths carry the estimated cross-leaf dependence, the resulting
aggregate bands are narrower than summed leaf quantiles (risk pooling).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def aggregate_leaf_forecasts(
    leaf_forecasts: dict[tuple[str, str], pd.Series],
) -> dict[str, pd.Series]:
    """Bottom-up aggregate a dict of {(sku, location): forecast_series} into
    per-SKU, per-Location, and Total forecast series — all guaranteed
    coherent with the leaves by construction. Use for additive quantities
    (point forecasts, means); use `aggregate_sample_paths` for quantiles.
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


def aggregate_sample_paths(
    samples: np.ndarray, leaves: list[tuple[str, str]]
) -> dict[str, np.ndarray]:
    """Sum joint leaf sample paths into aggregates.

    ``samples``: (n_samples, n_leaves, H) ordered as ``leaves``. Returns
    ``{"total" | "sku:<id>" | "location:<id>": (n_samples, H)}`` - the same
    keys as `aggregate_leaf_forecasts`.
    """
    out: dict[str, np.ndarray] = {"total": samples.sum(axis=1)}
    skus = sorted({s for s, _ in leaves})
    locs = sorted({l for _, l in leaves})
    for sku in skus:
        idx = [i for i, (s, _) in enumerate(leaves) if s == sku]
        out[f"sku:{sku}"] = samples[:, idx, :].sum(axis=1)
    for loc in locs:
        idx = [i for i, (_, l) in enumerate(leaves) if l == loc]
        out[f"location:{loc}"] = samples[:, idx, :].sum(axis=1)
    return out


def summarize_paths(
    paths: np.ndarray, quantiles: list[float] = (config.BAND_LOW_Q, config.MEDIAN_Q, config.BAND_HIGH_Q)
) -> pd.DataFrame:
    """Quantiles of the *horizon-total* demand of (n_samples, H) paths."""
    totals = paths.sum(axis=1)
    return pd.DataFrame({"quantile": list(quantiles), "horizon_total": np.quantile(totals, list(quantiles))})


def build_hierarchy_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small reference table of every (sku, location) leaf and its
    parent SKU/Location aggregates — useful for sanity-checking coherence."""
    return df[[config.SKU_COL, config.LOCATION_COL]].drop_duplicates().reset_index(drop=True)
