"""Scenario (sample-path) generation from quantile forecasts.

Why this exists
---------------
Quantiles are **not additive**: the sum of leaf P90s is not the P90 of the
summed demand. Adding them silently assumes every leaf lands in its tail
on the same day (correlation = 1), which overstates the uncertainty of any
aggregate (SKU / Location / Total) and throws away the risk-pooling
benefit that makes aggregate safety stocks smaller than the sum of leaf
safety stocks.

Instead we:

1. turn each leaf/horizon quantile forecast into a marginal distribution
   (piecewise-linear inverse CDF through the forecast quantiles, with
   linear tails clipped at zero);
2. couple the marginals with a **Gaussian copula** whose dependence is
   estimated from backtest PIT residuals - a Kronecker structure with a
   cross-leaf correlation matrix (shrunk toward identity) and an AR(1)
   correlation across horizons;
3. draw joint sample paths, and sum them to get *any* aggregate quantile
   or any lead-time demand distribution.

Assumptions are listed in README ("Assumptions & limitations").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config

_TAIL_LO = 0.001
_TAIL_HI = 0.999


# ---------------------------------------------------------------------------
# Marginals from quantiles
# ---------------------------------------------------------------------------
def quantile_curve(qvals: np.ndarray, levels: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (grid_levels, grid_values) of a monotone piecewise-linear
    inverse CDF through the given quantiles, extended to levels 0.001 and
    0.999 along the end slopes (lower tail floored at 0)."""
    lv = np.asarray(levels, dtype=float)
    v = np.maximum.accumulate(np.asarray(qvals, dtype=float))
    s_lo = (v[1] - v[0]) / (lv[1] - lv[0])
    s_hi = (v[-1] - v[-2]) / (lv[-1] - lv[-2])
    v_lo = max(0.0, v[0] - s_lo * (lv[0] - _TAIL_LO))
    v_hi = v[-1] + s_hi * (_TAIL_HI - lv[-1])
    grid_l = np.concatenate([[_TAIL_LO], lv, [_TAIL_HI]])
    grid_v = np.concatenate([[v_lo], v, [v_hi]])
    grid_v = np.maximum.accumulate(grid_v) + np.arange(len(grid_v)) * 1e-9  # strictly increasing for interp
    return grid_l, grid_v


def pit_values(records: pd.DataFrame, levels: list[float]) -> np.ndarray:
    """Probability integral transform of each actual under its forecast."""
    Q = records[levels].to_numpy()
    y = records["y"].to_numpy()
    out = np.empty(len(y))
    for i in range(len(y)):
        gl, gv = quantile_curve(Q[i], levels)
        out[i] = np.interp(y[i], gv, gl)
    return np.clip(out, _TAIL_LO, _TAIL_HI)


# ---------------------------------------------------------------------------
# Dependence estimation
# ---------------------------------------------------------------------------
@dataclass
class Dependence:
    leaves: list[tuple[str, str]]
    chol_leaf: np.ndarray    # (n_leaves, n_leaves) Cholesky factor of the cross-leaf correlation
    chol_h: np.ndarray       # (H, H) Cholesky factor of the AR(1) horizon correlation
    rho_h: float             # adjacent-horizon correlation
    mean_leaf_corr: float    # average off-diagonal cross-leaf correlation (diagnostic)


def _nearest_corr(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((C + C.T) / 2)
    w = np.clip(w, 1e-6, None)
    C = (V * w) @ V.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def estimate_dependence(
    leaf_records: dict[tuple[str, str], pd.DataFrame],
    levels: list[float] = config.QUANTILES,
    horizon: int = config.HORIZON_DAYS,
    exclude_fold: int | None = None,
) -> Dependence:
    """Estimate copula dependence from backtest records (columns: fold_id,
    h, y, <quantile levels>). ``exclude_fold`` leaves out a fold - the
    pipeline excludes the one used for the "latest forecast" so the
    dependence structure isn't fitted on the data it is then applied to."""
    leaves = list(leaf_records)
    z_cols = {}
    z_by_leaf = {}
    for leaf, rec in leaf_records.items():
        if exclude_fold is not None:
            rec = rec[rec["fold_id"] != exclude_fold]
        z = norm.ppf(pit_values(rec, levels))
        frame = pd.DataFrame({"fold_id": rec["fold_id"].to_numpy(), "h": rec["h"].to_numpy(), "z": z})
        z_by_leaf[leaf] = frame
        z_cols[leaf] = frame.set_index(["fold_id", "h"])["z"]

    wide = pd.DataFrame(z_cols)
    corr = wide.corr(min_periods=10).to_numpy()
    corr = np.where(np.isnan(corr), 0.0, corr)
    np.fill_diagonal(corr, 1.0)
    lam = config.COPULA_LEAF_SHRINKAGE
    corr = _nearest_corr((1 - lam) * corr + lam * np.eye(len(leaves)))
    off = corr[~np.eye(len(leaves), dtype=bool)]
    mean_off = float(off.mean()) if off.size else 0.0

    # AR(1) correlation between adjacent horizons: pooled over leaves and folds.
    num = den_a = den_b = 0.0
    for frame in z_by_leaf.values():
        piv = frame.pivot_table(index="fold_id", columns="h", values="z")
        for h in range(1, horizon):
            if h in piv.columns and (h + 1) in piv.columns:
                a, b = piv[h].to_numpy(), piv[h + 1].to_numpy()
                m = ~(np.isnan(a) | np.isnan(b))
                num += float(np.sum(a[m] * b[m]))
                den_a += float(np.sum(a[m] ** 2))
                den_b += float(np.sum(b[m] ** 2))
    rho = num / np.sqrt(den_a * den_b) if den_a > 0 and den_b > 0 else 0.0
    rho = float(np.clip(rho, 0.0, config.COPULA_MAX_HORIZON_RHO))

    steps = np.arange(horizon)
    corr_h = rho ** np.abs(steps[:, None] - steps[None, :])
    return Dependence(
        leaves=leaves,
        chol_leaf=np.linalg.cholesky(corr),
        chol_h=np.linalg.cholesky(corr_h),
        rho_h=rho,
        mean_leaf_corr=mean_off,
    )


# ---------------------------------------------------------------------------
# Joint sampling
# ---------------------------------------------------------------------------
def sample_joint_paths(
    forecasts: dict[tuple[str, str], np.ndarray],
    dependence: Dependence,
    levels: list[float] = config.QUANTILES,
    n_samples: int = config.N_SCENARIO_SAMPLES,
    seed: int = config.SCENARIO_SEED,
) -> np.ndarray:
    """Draw joint demand sample paths.

    ``forecasts[leaf]`` is an (H, n_levels) array of calibrated quantile
    forecasts for the next H days. Returns an array of shape
    ``(n_samples, n_leaves, H)`` ordered as ``dependence.leaves``.
    """
    leaves = dependence.leaves
    H = dependence.chol_h.shape[0]
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((n_samples, len(leaves), H))
    Z = np.einsum("ij,sjh->sih", dependence.chol_leaf, E)   # cross-leaf dependence
    Z = np.einsum("sih,gh->sig", Z, dependence.chol_h)       # cross-horizon dependence
    U = norm.cdf(Z)

    samples = np.empty_like(U)
    for i, leaf in enumerate(leaves):
        Q = forecasts[leaf]
        for h in range(H):
            gl, gv = quantile_curve(Q[h], levels)
            samples[:, i, h] = np.interp(U[:, i, h], gl, gv)
    return samples


def draw_lead_times(
    n_samples: int, mean_days: float, std_days: float, max_days: int, seed: int = config.SCENARIO_SEED
) -> np.ndarray:
    """Integer lead times (days), 1..max_days. Deterministic if std = 0."""
    if std_days <= 0:
        return np.full(n_samples, int(np.clip(round(mean_days), 1, max_days)))
    rng = np.random.default_rng(seed + 1)
    return np.clip(np.rint(rng.normal(mean_days, std_days, n_samples)), 1, max_days).astype(int)


def lead_time_demand(daily_samples: np.ndarray, lead_times: np.ndarray) -> np.ndarray:
    """Cumulative demand over each sample's own lead time.

    ``daily_samples``: (n_samples, H) daily demand paths of one series (a
    leaf or any aggregate). Returns (n_samples,).
    """
    cum = np.cumsum(daily_samples, axis=1)
    return cum[np.arange(len(lead_times)), lead_times - 1]
