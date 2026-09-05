# Supply Chain Demand Forecasting: SKU x Location, Probabilistic, Decision-Linked

A demand forecasting pipeline built around what actually drives supply-chain
decisions — not just point-forecast accuracy. Forecasts are produced **per
SKU x Location**, reconciled bottom-up into SKU/Location/Total views,
expressed as **probability distributions** (not single numbers) so they can
directly size safety stock, evaluated by **business cost** (stockout vs.
holding, derived per-SKU from real price data), and monitored with a proper
**statistical drop-detection** method.

**This project uses only the real dataset provided — there is no synthetic
data anywhere in this pipeline.**

This is the decision-oriented sister project to
[`tabpfn-forecast-benchmark`](../tabpfn-forecast-benchmark), which focuses
purely on model accuracy comparison. This project assumes "which model" is
a solved-enough question and focuses on turning a forecast into a plan.

## What's different from a typical forecasting demo

| Typical demo | This project |
|---|---|
| One aggregate series | Per SKU x Location, reconciled bottom-up |
| Point forecast only | P10/P50/P90 quantile forecast, conformal-calibrated |
| Single train/test split | Walk-forward backtest, multiple folds |
| MAE/RMSE only | + pinball loss (distributional) + newsvendor cost (business) |
| "Trust me, it's calibrated" | Reliability diagram: nominal vs. empirical quantile coverage, measured |
| Ad-hoc "demand < 70% of mean" drop flag | CUSUM control chart (detects sustained shifts, not just single-day dips) |
| Flat $/unit cost assumption | Cost derived per-SKU from that SKU's real price |
| Forecast stops at accuracy | Forecast → safety stock / reorder point |

## Why bottom-up reconciliation (not top-down or MinT)

Leaf-level (SKU x Location) forecasts are what actually drive inventory
decisions, so there's no accuracy worth "borrowing" from higher levels that
matters more than getting the leaves right — and bottom-up guarantees every
aggregate is coherent with (traceable to) its leaves by construction, no
extra estimation step required. See `src/hierarchy.py` for the full
rationale and where MinT/top-down would be worth the added complexity.

## Why conformal-calibrated quantile regression (not just quantile XGBoost)

Raw quantile regression's quantiles aren't statistically guaranteed to have
correct coverage — an XGBoost model trained with `quantile_alpha=0.9` won't
necessarily contain the true value 90% of the time. Split conformal
prediction adds a calibration step on held-out data to fix that — and
`reports/figures/reliability_diagram.png` measures whether it worked,
rather than assuming it did. See `src/probabilistic.py`.

## Why CUSUM instead of a fixed threshold

A fixed "demand < 70% of mean" rule only catches single-day cliffs and is
blind to a slow, sustained erosion that never crosses the threshold on any
one day but adds up to a real problem. CUSUM accumulates small deviations
over time and alarms once the *cumulative* shift is statistically
significant — the standard tool for this in process-control / supply-chain
monitoring. See `src/drop_detection.py`.

## Project structure

```
.
├── data/
│   ├── raw/sku_location_demand.csv   # real data (76K rows), git-ignored
│   └── README.md
├── notebooks/
│   └── 01_hierarchy_eda.ipynb
├── src/
│   ├── config.py                 # hierarchy, backtest, cost-derivation, CUSUM parameters
│   ├── prepare_real_data.py      # one-time adapter: original CSV -> this schema
│   ├── data_loader.py
│   ├── features.py               # per-leaf lag/rolling/calendar features
│   ├── backtest.py               # walk-forward fold generation
│   ├── probabilistic.py          # quantile XGBoost + conformal calibration
│   ├── hierarchy.py              # bottom-up reconciliation
│   ├── drop_detection.py         # CUSUM control chart
│   ├── inventory.py              # safety stock + price-derived newsvendor cost
│   ├── eval_metrics.py           # MAE / RMSE / MAPE
│   ├── plotting.py               # all figures (data, fit, calibration diagnostics)
│   └── pipeline.py               # orchestrates everything, per leaf + aggregated
├── scripts/run_pipeline.py
├── tests/
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No `.env` needed — no external API dependency (TabPFN was left in the
sister project). The real dataset (76,000 rows: 20 SKUs × 5 locations ×
760 days) is included with this delivery, already adapted to
`data/raw/sku_location_demand.csv`. If you need to regenerate it from the
original file:

```bash
python -m src.prepare_real_data --input /path/to/demand_forecasting.csv
```

See `data/README.md` for the schema and data-quality notes.

## Usage

```bash
python scripts/run_pipeline.py
# Quick smoke test on a handful of leaves instead of all 100:
python scripts/run_pipeline.py --max-leaves 10
# Parallelize across CPU cores (not GPU/CUDA — see note below):
python scripts/run_pipeline.py --n-jobs 4
```

Prints: per-SKU-x-Location backtest summary (pinball loss, newsvendor
cost, CUSUM alarms), a bottom-up reconciled total, an example safety-stock
plan, and a portfolio-wide calibration table — then saves 9 figures to
`reports/figures/`.

**Runtime:** ~3–4 seconds/leaf sequentially, so the full 100-leaf run
takes roughly 5–6 minutes on one CPU core. Use `--max-leaves` while
iterating, or `--n-jobs N` to backtest N leaves in parallel across CPU
cores (`multiprocessing`, not GPU/CUDA — each leaf's dataset is small
and the 100 leaves are fully independent, so process-level CPU
parallelism is the right lever, not a GPU). When `--n-jobs > 1`, each
XGBoost model is pinned to 1 thread internally so the worker processes
don't fight XGBoost's own multi-threading for the same cores — see
`src/pipeline.py:run_all_leaves`.

Explore interactively:

```bash
jupyter lab notebooks/01_hierarchy_eda.ipynb
```

## The figures, and what each one proves

Every figure below was generated on the real dataset (a 20-leaf, 4-SKU x
5-store slice for speed — re-run `run_pipeline.py` on all 100 for the full
portfolio version).

**Data characteristics** — establishing what kind of data this is, before
trusting any model built on it:
- `demand_distribution_seasonality.png` — pooled demand is **right-skewed**
  (long tail of high-demand days), and, contrary to the usual retail
  assumption, **day-of-week has almost no effect** on this dataset — the
  boxplots for Mon–Sun are nearly identical. This is worth knowing: the
  `lag_7`/`rolling_mean_7` features in `src/features.py` are there for
  robustness, not because this specific dataset has strong weekly
  seasonality.
- `demand_heatmap_sku_location.png` — average demand varies far more by
  SKU than by store, i.e. per-leaf (not pooled) modeling is justified.

**Model fit diagnostics** — is the model's error well-behaved?
- `residual_diagnostics.png` — pooled residuals (actual − median
  forecast) across every fold of every leaf: (1) scattered around zero
  over time with no obvious drift, (2) centered near zero (mean ≈ 1.2 on
  a demand scale of ~100 — negligible bias), (3) **QQ-plot shows fatter
  tails than a normal distribution**, especially on the upside. This
  directly matters for `src/inventory.py`'s safety-stock formula, which
  assumes normal residuals — the fat tails mean that formula is likely
  **understating** the buffer needed against rare extreme-demand days.

**Stochastic / probabilistic accuracy** — does "P90" actually mean P90?
- `reliability_diagram.png` — the core calibration check: for each
  nominal quantile (P10/P50/P90), what fraction of real observations
  actually fell at or below the prediction, across ~5,900 fold-leaf
  observations per quantile? On the included run: **P10→12.9%,
  P50→50.6%, P90→87.3%** — close to the diagonal, meaning the conformal
  calibration step is doing its job, with a slight over-coverage at the
  tails (a conservative, safe direction of error for inventory decisions).
- `pinball_loss_by_quantile.png` — boxplot of each leaf's mean pinball
  loss, by quantile, across the leaf portfolio. Note: pinball loss
  magnitudes aren't meant to be compared *across* quantiles (P50's loss
  formula weights differently than P10's/P90's) — use this to compare
  *consistency* across leaves at a given quantile, not to conclude "P50
  is worse than P10."
- `leaf_accuracy_distribution.png` — (1) MAE spread across the leaf
  portfolio (is accuracy consistent, or do a few leaves drag the average
  down?); (2) newsvendor cost vs. SKU price — confirms the price-derived
  cost model scales sensibly (higher-value SKUs carry more $ risk per
  unit of forecast error, as intended).

**Decision outputs:**
- `quantile_fan_example.png` — one leaf's actual demand vs. its P10–P90
  forecast band, the shape a safety-stock decision is actually made from.
- `cusum_example.png` — raw demand + CUSUM statistic with alarm threshold
  and flagged drop events.
- `hierarchy_reconciliation.png` — proof that SKU-level and Location-level
  views both sum exactly to the same Total.

## Known simplifications (documented, not hidden)

- **Category is unreliable in the source data** (same product tagged with
  different categories across rows), so it isn't used as a hierarchy
  dimension. `Region` (fixed per Store) would be a clean 4th level to add.
- **Demand vs. Units Sold censoring is not modeled.** The real dataset's
  `Demand` and `Units Sold` disagree ~28% of the time in a pattern
  consistent with stockout-censored sales — this project forecasts
  `Demand` as given, without a censored-demand correction. Documented in
  `data/README.md`.
- **Bottom-up reconciliation only** — no top-down or MinT (trace
  minimization). Documented trade-off in `src/hierarchy.py`.
- **Normal-approximation safety stock**, despite the residual diagnostics
  above showing fat tails — a first-pass simplification. A full
  simulation-based or empirical-quantile approach would be the next step,
  especially for highly-intermittent or lumpy-demand SKUs.
- **Costs are derived from price via rules of thumb**
  (`STOCKOUT_COST_AS_FRACTION_OF_PRICE`, `ANNUAL_HOLDING_COST_RATE` in
  `src/config.py`), not observed margin/holding-cost data — the dataset
  has no such columns. Treat newsvendor $ figures as directionally useful
  for comparing SKUs/models, not as precise business numbers, until real
  cost data is available. See `src/inventory.py:derive_costs_from_price`.
- **Weekly-seasonality features exist but the data doesn't show much
  weekly seasonality** (see `demand_distribution_seasonality.png`) — the
  lag-7 features aren't hurting, but don't expect them to be doing much
  work on this particular dataset.

## License

MIT — see [LICENSE](LICENSE).
