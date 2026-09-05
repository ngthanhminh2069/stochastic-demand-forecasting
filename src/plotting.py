"""Plotting helpers for the SKU x Location pipeline. All figures are saved
under reports/figures/ as PNGs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src import config


def _save(fig, name: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_quantile_fan(
    y_actual: pd.Series,
    quantile_forecast: pd.DataFrame,
    title: str,
    name: str,
) -> Path:
    """Fan chart: actual demand vs. the P10/P50/P90 forecast band. This is
    the visual that makes "probabilistic forecast" concrete — a shaded
    range you can size safety stock from, not just one line.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_actual.index, y_actual.values, color="black", linewidth=2, marker="o",
             markersize=4, label="Actual")

    cols = sorted(quantile_forecast.columns)
    low, mid, high = cols[0], cols[len(cols) // 2], cols[-1]

    ax.plot(quantile_forecast.index, quantile_forecast[mid], color="#1E40AF",
             linewidth=1.8, linestyle="--", label=f"Median (P{int(mid*100)})")
    ax.fill_between(
        quantile_forecast.index, quantile_forecast[low], quantile_forecast[high],
        color="#1E40AF", alpha=0.18, label=f"P{int(low*100)}\u2013P{int(high*100)} band"
    )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return _save(fig, name)


def plot_cusum(series: pd.Series, cusum_result, title: str, name: str) -> Path:
    """Two-panel chart: raw demand on top, CUSUM statistic with alarm
    threshold below, alarm days marked on both."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(series.index, series.values, color="#1E40AF", linewidth=1.2)
    for d in cusum_result.alarm_days:
        axes[0].axvline(d, color="#991B1B", alpha=0.3, linewidth=1)
    axes[0].set_title(title, fontsize=13)
    axes[0].set_ylabel("Demand")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(cusum_result.cusum_series.index, cusum_result.cusum_series.values,
                  color="#991B1B", linewidth=1.3)
    axes[1].axhline(-cusum_result.threshold, color="black", linestyle="--",
                      linewidth=1, label="Alarm threshold")
    axes[1].scatter(cusum_result.alarm_days,
                      cusum_result.cusum_series.loc[cusum_result.alarm_days],
                      color="#991B1B", zorder=5, s=25)
    axes[1].set_title("CUSUM statistic (resets after each alarm)", fontsize=11)
    axes[1].set_xlabel("Date")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return _save(fig, name)


def plot_hierarchy_bars(aggregates: dict[str, pd.Series], name: str = "hierarchy_reconciliation") -> Path:
    """Bar chart comparing the reconciled Total to the sum of SKU-level and
    Location-level forecasts — the visual proof that bottom-up
    reconciliation is coherent."""
    total = aggregates["total"].sum()
    sku_keys = sorted(k for k in aggregates if k.startswith("sku:"))
    location_keys = sorted(k for k in aggregates if k.startswith("location:"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sku_values = [aggregates[k].sum() for k in sku_keys]
    axes[0].bar([k.replace("sku:", "") for k in sku_keys], sku_values, color="#1E40AF")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title(f"By SKU (sum = {sum(sku_values):,.0f}, Total = {total:,.0f})", fontsize=11)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Forecast demand (horizon sum)")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    loc_values = [aggregates[k].sum() for k in location_keys]
    axes[1].bar([k.replace("location:", "") for k in location_keys], loc_values, color="#166534")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"By Location (sum = {sum(loc_values):,.0f}, Total = {total:,.0f})", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Bottom-Up Reconciliation: SKU and Location views both sum to Total", fontsize=13)
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Data characteristics — establish what kind of data this is before
# trusting any model built on it.
# ---------------------------------------------------------------------------

def plot_demand_distribution_and_seasonality(df: pd.DataFrame, name: str = "demand_distribution_seasonality") -> Path:
    """Two panels: (1) histogram of daily per-leaf demand — shows the
    overall shape (skew, spread) of what's being forecast; (2) boxplot of
    demand by day-of-week, pooled across every leaf — shows whether weekly
    seasonality is actually present in the raw data (motivating the
    lag-7/rolling-7 features in `src/features.py`).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df[config.TARGET_COL], bins=60, color="#1E40AF", alpha=0.8)
    axes[0].set_title("Distribution of Daily Demand (pooled, all leaves)", fontsize=12)
    axes[0].set_xlabel("Demand")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    dow = df.copy()
    dow["dayofweek"] = dow[config.DATE_COL].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    data_by_day = [dow.loc[dow["dayofweek"] == d, config.TARGET_COL].values for d in order]
    axes[1].boxplot(data_by_day, labels=[d[:3] for d in order], showfliers=False)
    axes[1].set_title("Demand by Day of Week (pooled, all leaves)", fontsize=12)
    axes[1].set_ylabel("Demand")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    return _save(fig, name)


def plot_demand_heatmap(df: pd.DataFrame, name: str = "demand_heatmap_sku_location") -> Path:
    """Heatmap of average daily demand across every SKU x Location leaf —
    shows at a glance which leaves are high/low volume, and whether
    demand varies more by SKU or by location (informs whether pooling or
    per-leaf modeling makes more sense)."""
    pivot = df.pivot_table(
        index=config.SKU_COL, columns=config.LOCATION_COL, values=config.TARGET_COL, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(1.3 * len(pivot.columns) + 3, 0.35 * len(pivot.index) + 2))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Average Daily Demand: SKU x Location", fontsize=13)
    fig.colorbar(im, ax=ax, label="Mean demand")
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Model fit diagnostics — is the model's error behaving the way a good
# forecast's error should (centered at zero, no obvious pattern left)?
# ---------------------------------------------------------------------------

def plot_residual_diagnostics(residuals: pd.Series, name: str = "residual_diagnostics") -> Path:
    """Three panels: residuals over time (checks for drift/pattern the
    model missed), histogram of residuals (checks centering/skew), and a
    QQ-plot against a normal distribution (justifies — or flags — the
    normal-approximation used to size safety stock in `src/inventory.py`).
    """
    import numpy as np
    from scipy import stats

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].scatter(residuals.index, residuals.values, s=6, alpha=0.35, color="#1E40AF")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Residuals over Time (all leaves pooled)", fontsize=11)
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Actual − Predicted (median)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].hist(residuals.values, bins=60, color="#166534", alpha=0.8)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].axvline(residuals.mean(), color="#991B1B", linestyle="--", linewidth=1.3,
                     label=f"Mean = {residuals.mean():.1f}")
    axes[1].set_title("Residual Distribution", fontsize=11)
    axes[1].set_xlabel("Residual")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.5)

    stats.probplot(residuals.values, dist="norm", plot=axes[2])
    axes[2].set_title("QQ-Plot vs. Normal", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Residual Diagnostics (median forecast, all walk-forward folds, all leaves)", fontsize=13)
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Stochastic / probabilistic accuracy — does the P10/P50/P90 forecast mean
# what it claims to mean, not just "is the median close on average"?
# ---------------------------------------------------------------------------

def plot_reliability_diagram(reliability_table: pd.DataFrame, name: str = "reliability_diagram") -> Path:
    """The core calibration check for a probabilistic forecast: for each
    nominal quantile q (e.g. P90), what fraction of actual observations
    fell at or below the predicted q, across every fold of every leaf? A
    perfectly calibrated model sits exactly on the diagonal. This is a
    stronger, more honest test than reporting a single "coverage %" number
    — it's what actually validates the conformal calibration step in
    `src/probabilistic.py`.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="Perfect calibration")
    ax.scatter(
        reliability_table["nominal_quantile"], reliability_table["empirical_fraction_below"],
        color="#991B1B", s=80, zorder=5, label="Observed"
    )
    for _, row in reliability_table.iterrows():
        ax.annotate(
            f"P{int(row['nominal_quantile']*100)}\n(n={int(row['n_obs'])})",
            (row["nominal_quantile"], row["empirical_fraction_below"]),
            textcoords="offset points", xytext=(10, -4), fontsize=8,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Empirical fraction of actuals at/below prediction")
    ax.set_title("Reliability Diagram — Quantile Calibration Check", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return _save(fig, name)


def plot_pinball_by_quantile(pinball_df: pd.DataFrame, name: str = "pinball_loss_by_quantile") -> Path:
    """Boxplot of each leaf's mean pinball loss, grouped by quantile —
    shows both typical performance and how much it varies across the
    100-leaf portfolio (a model that's great on 90 leaves and terrible on
    10 looks very different from one that's consistently mediocre, even
    with the same average)."""
    quantiles = sorted(pinball_df["quantile"].unique())
    data = [pinball_df.loc[pinball_df["quantile"] == q, "pinball_loss"].dropna().values for q in quantiles]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=[f"P{int(q*100)}" for q in quantiles])
    ax.set_title("Pinball Loss by Quantile, Across All SKU x Location Leaves", fontsize=12)
    ax.set_ylabel("Mean pinball loss (per leaf)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return _save(fig, name)


def plot_leaf_accuracy_distribution(leaf_summary: pd.DataFrame, name: str = "leaf_accuracy_distribution") -> Path:
    """Two panels: (1) histogram of mean MAE across all leaves — shows
    whether accuracy is consistent across the portfolio or a few leaves
    drag the average down; (2) scatter of mean daily newsvendor cost vs.
    average price — sanity-checks that higher-value SKUs really do carry
    more forecast-error cost, as the price-derived cost model intends."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(leaf_summary["mean_mae"].dropna(), bins=25, color="#1E40AF", alpha=0.8)
    axes[0].axvline(leaf_summary["mean_mae"].median(), color="#991B1B", linestyle="--",
                     label=f"Median = {leaf_summary['mean_mae'].median():.1f}")
    axes[0].set_title("Mean MAE Across All Leaves", fontsize=12)
    axes[0].set_xlabel("Mean MAE (per leaf, across folds)")
    axes[0].set_ylabel("Number of leaves")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.5)

    axes[1].scatter(leaf_summary["avg_price"], leaf_summary["mean_cost_per_day"], color="#166534", alpha=0.7)
    axes[1].set_title("Newsvendor Cost vs. SKU Price", fontsize=12)
    axes[1].set_xlabel("Average unit price ($)")
    axes[1].set_ylabel("Mean newsvendor cost ($/day)")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return _save(fig, name)
