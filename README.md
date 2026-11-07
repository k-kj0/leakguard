# Leakguard: does your order-book model survive honest validation?

Leakguard tests whether a short-horizon order-book machine-learning result is real or an artifact of leakage. It audits the data, checks features for look-ahead, validates with purged walk-forward splits, and reports uncertainty and calibration.

## What it found

**1. Shuffled cross-validation inflates the score.** Same data, same models; only the validation method changes (synthetic data, 30,000 rows, 50-row horizon):

| Model | Naive K-fold AUC | Purged walk-forward AUC | Inflation |
|---|---:|---:|---:|
| Logistic regression | 0.617 | 0.577 | +0.040 |
| Gradient boosting | 0.622 | 0.559 | +0.062 |

An earlier, over-fitted version of the gradient-boosting model showed a far bigger gap (AUC 0.725 naive vs 0.555 purged). A flexible model can memorise neighbouring rows, and shuffled splits reward that.

**2. A good AUC can hide bad probabilities.** That over-fitted model scored a Brier of 0.280, worse than the 0.25 of a constant 50/50 forecast. Smaller trees, larger leaves and a stronger L2 penalty brought it to 0.2475. Every report now prints calibration error next to Brier.

**3. Contemporaneous is not predictive.** The repo reproduces the method of Cont, Kukanov and Stoikov (2014), who found that order flow imbalance explains same-interval price changes well (average R² about 0.65). A forecaster needs it to predict the *next* interval. On synthetic data the code gives R² 0.68 for the same bin and 0.002 for the next bin. Synthetic prices are generated tick by tick, so this only checks the code. The real-data result is below.

## Real-data result

> FILL IN after the real-data run. Link the folder (for example `reports/real_capture_run_1/`), and paste the headline table from `experiment.md` and the three rows from `ofi_reproduction.md`. Report the numbers as they came out, including "no edge".

## Quick start

Requires Python 3.11 or newer.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m leakguard.experiment --n 30000 --horizon 50
python -m leakguard.ofi_reproduction --n 30000
```

Reports are written to `results/` (ignored by git). Curated real-data reports are committed to `reports/`.

## How it works

```text
 Public / synthetic market data
              |
   Snapshot + update handling
              |
   Order-book reconstruction
              |
     Data-quality auditing
              |
   Causal feature engineering
              |
      Leakage diagnostics
              |
 Purged walk-forward validation  <--  naive K-fold shown as the negative control
              |
   Models: logistic regression, gradient boosting
              |
   Regimes | Confidence intervals | Fold stability | Calibration
              |
 Null + selection-bias experiments, Red-Team challenges
              |
     Reproducible reports
```

### Leakage-aware validation

Random cross-validation misleads on time series when label windows overlap. Leakguard provides chronological walk-forward folds, purging of training labels that overlap the test window, a configurable embargo, label-overlap diagnostics, and a naive K-fold comparison so the inflation is measured, not assumed.

### Causal features

Spread, level-one and multi-level imbalance, rolling imbalance, microprice, short returns, rolling volatility and spread in basis points. A truncation test checks that each feature at row *t* does not change when the future is removed. A deliberately leaky feature proves the test can catch a problem. This is a practical diagnostic, not a proof that no leakage exists.

### Label and horizon choices

- **Ties are dropped by default.** If the mid-price did not move over the horizon, the row is removed so the label means "up vs down". Use `--keep-ties` for the old behaviour. Each report prints the share of tied rows.
- **Time-based horizon.** `--horizon-ms 5000` predicts five seconds ahead instead of 50 rows ahead. Purging automatically covers the longest label window.
- **Both models balance classes**, so the 0.5 threshold means the same thing for each.
- **The cost diagnostic uses non-overlapping positions**, one per horizon, so returns are not counted many times over.

### Statistical reliability

- Volatility regimes (low, normal, high) with thresholds taken from the training folds only.
- Circular moving-block bootstrap intervals for AUC, accuracy and Brier.
- Fold-by-fold stability: mean, standard deviation, minimum, maximum.
- Calibration error and Brier, with the 0.25 constant-forecast baseline stated in the report.

These are approximate diagnostics, not guarantees of independence or generalisation.

### Null experiments, selection bias and Red-Team

```bash
python -m leakguard.null_experiments --observations 1000 --repetitions 500 --seed 7
python -m leakguard.selection_bias --n-observations 300 --n-trials 20 --repetitions 30 --seed 7
python -m leakguard.redteam --n 1200 --seed 7
```

- **Null experiment:** random labels and scores give the chance-level AUC distribution.
- **Selection bias:** picking the best of many random predictors produces an impressive-looking AUC with no signal.
- **Red-Team:** injected faults (duplicate or out-of-order timestamps, missing values, crossed books, negative sizes, future-dependent features) plus clean controls. The pass rate covers only the implemented cases, not arbitrary real-world corruption.

### Reproduction: order flow imbalance

`leakguard/ofi_reproduction.py` computes level-1 order flow imbalance, regresses mid-price change on it inside fixed time bins, and also regresses the *next* bin's change on it.

```bash
python -m leakguard.ofi_reproduction --data results/coinbase_normalized.csv --bin-ms 10000
```

Differences from the paper are listed inside the generated report. The main ones: the paper uses event-by-event NYSE data for many stocks, while this repo uses fixed-grid snapshots of one crypto product.

### Benchmarks

```bash
python -m leakguard.benchmark
```

Measures synthetic generation and feature construction time and Python-tracked memory. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for methodology and limits.

## Market data

Public Coinbase adapters need no credentials.

```bash
# REST Level 2 snapshot
python -m leakguard.market_data --product BTC-USD --depth 10 --out data/coinbase_btc_usd_snapshot.csv

# Advanced Trade Level 2 WebSocket capture
python -m leakguard.coinbase_ws --product BTC-USD --duration 300 --out results/coinbase_ws_capture.csv

# Put a capture on a fixed time grid (default 100 ms)
python -m leakguard.coinbase_normalize --input results/coinbase_ws_capture.csv --out results/coinbase_normalized.csv

# Audit the normalized data
python -m leakguard.data_quality --input results/coinbase_normalized.csv --out results/data_quality --expected-interval-ms 100
```

### Real-data workflow

**GitHub → Actions → Coinbase Real-Data Research Pipeline → Run workflow** (default: BTC-USD, 900 seconds).

It runs the tests, captures Level 2 data, normalizes it, audits quality, runs the experiment with a 5-second horizon, runs the order-flow-imbalance reproduction and the Red-Team suite, then commits the reports to `reports/real_capture_run_<n>/`. Full artifacts stay on the run page for 30 days.

### Running on your own data

```bash
python -m leakguard.experiment --data data/normalized_order_book.csv --horizon-ms 5000 --splits 5 --fee-bps 1 --slippage-bps 1
```

Required columns: `ts_ms`, `bid_px_1`, `ask_px_1`, `bid_sz_1`, `ask_sz_1`. Optional depth: `bid_px_2/3`, `ask_px_2/3`, `bid_sz_2/3`, `ask_sz_2/3`. Timestamps must be ascending and unique, prices and sizes in consistent units, best bid below best ask, and at least 500 rows. Raw captures must be normalized first.

## Testing and CI

```bash
python -m pytest -q
ruff check .
```

GitHub Actions runs tests, lint, the synthetic experiment and the reproduction on every push. A manual **Lint auto-fix** workflow applies safe `ruff` fixes.

## Limitations

**Important:** Leakguard is not a production HFT engine, exchange execution system, or proof of profitable trading alpha. It is an independent research-engineering prototype.

It does not claim:

- Exchange-grade feed handling, reconnection or missed-message recovery.
- Level 3 reconstruction, queue-position estimation or partial-fill simulation.
- Market-impact, latency or adverse-selection modelling.
- Profitable alpha or live-trading readiness.

Statistical caveats:

- Synthetic data tests the software. It is not evidence about markets.
- A short capture (minutes to hours, one product) gives wide uncertainty.
- The naive-vs-purged gap compares a random 80/20 split with an expanding window, so it is indicative, not a precise measure of leakage.
- Bootstrap intervals are approximate, and fold stability does not predict the future.
- Snapshots on a 100 ms grid miss events between snapshots.
- The cost diagnostic is not an execution simulator.

## Repository structure

```text
leakguard/
├── .github/workflows/      CI, lint auto-fix, market-data and real-data pipelines
├── docs/BENCHMARKS.md
├── leakguard/
│   ├── benchmark.py  checks.py  coinbase_normalize.py  coinbase_ws.py
│   ├── cv.py  data.py  data_quality.py  experiment.py  features.py
│   ├── market_data.py  metrics.py  models.py  null_experiments.py
│   ├── ofi_reproduction.py  reconstruction.py  redteam.py
│   ├── reliability.py  selection_bias.py  simulate.py
├── reports/                committed real-data reports
├── tests/
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Objective

A model result should not be trusted until the data, features, validation method and evaluation assumptions have been challenged. Leakguard combines market-data engineering, time-series validation, statistical diagnostics, controlled fault injection and reproducible reports into one workflow.

## License

MIT. See [LICENSE](LICENSE).
