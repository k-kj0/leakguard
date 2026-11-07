
# Leakguard — Order-Book ML Research Validation

**A research-engineering toolkit for testing the reliability of short-horizon order-book machine-learning experiments.**

Leakguard investigates a fundamental problem in quantitative research: a model's reported performance is not trustworthy if its market data is corrupted, its features contain future information, or its validation methodology leaks information across time.

It combines public market-data collection, order-book reconstruction, data-quality auditing, causal feature engineering, leakage diagnostics, purged chronological validation, statistical reliability checks, controlled Red-Team challenges, and reproducible reports.

**Status:** Independent research-engineering prototype.

**Important:** Leakguard is not a production HFT engine, exchange execution system, or proof of profitable trading alpha.

---

## 1. Research objective

The central research question:

Can order-book imbalance and microstructure features predict short-horizon price movements out of sample after accounting for:

- Market-data integrity problems.
- Future-information leakage.
- Overlapping prediction labels.
- Changes in volatility regimes.
- Statistical uncertainty.
- Multiple-experiment selection bias.
- Simplified transaction-cost assumptions.

The goal is not simply to train a model. It is to make the research process more auditable, reproducible, and difficult to fool.

## 2. Architecture

```text
       Public / Synthetic Market Data
                    |
                    v
             Data Collection
                    |
                    v
          Snapshot + Update Handling
                    |
                    v
          Order-Book Reconstruction
                    |
                    v
            Data-Quality Auditing
                    |
                    v
           Causal Feature Engineering
                    |
                    v
             Leakage Diagnostics
                    |
                    v
         Purged Walk-Forward Validation
                    |
                    v
          Machine-Learning Baselines
                    |
                    v
       Out-of-Sample Model Evaluation
                    |
          +---------+----------+
          |         |          |
          v         v          v
       Regime   Confidence   Fold
      Analysis  Intervals  Stability
          |         |          |
          +---------+----------+
                    |
                    v
       Null + Selection-Bias Experiments
                    |
                    v
           Red-Team Challenges
                    |
                    v
        Reproducible Research Reports
```

## 3. Market-data integration

### Coinbase REST Level 2 snapshot

Leakguard includes a public Coinbase Exchange REST adapter.

It:

- Retrieves a Level 2 order-book snapshot.
- Validates the returned response.
- Preserves the available exchange sequence number.
- Normalizes bid and ask price levels.
- Writes CSV data and JSON metadata.

Example:

```bash
python -m leakguard.market_data \
  --product BTC-USD \
  --depth 10 \
  --out data/coinbase_btc_usd_snapshot.csv
```

### Coinbase Advanced Trade Level 2 WebSocket

Leakguard also includes a public WebSocket collector.

It:

- Connects to the Coinbase Advanced Trade WebSocket feed.
- Subscribes to Level 2 market data.
- Initializes the local order book from a snapshot.
- Applies incremental price-level updates.
- Handles absolute size updates and zero-size deletions.
- Records captured book states.
- Saves CSV data and JSON capture metadata.

Example:

```bash
python -m leakguard.coinbase_ws \
  --product BTC-USD \
  --duration 300 \
  --out results/coinbase_ws_capture.csv
```

No exchange API credentials are required for the public market-data collection workflow.

### Automated real-data research workflow

The repository includes a manually triggered GitHub Actions workflow:

`.github/workflows/coinbase_research_pipeline.yml`

It runs the following sequence:

1. Installs the Python dependencies.
2. Runs the automated test suite.
3. Captures Coinbase Advanced Trade Level 2 data.
4. Normalizes the captured data.
5. Audits data quality.
6. Runs the research experiment on the normalized data.
7. Runs the Red-Team challenge suite.
8. Publishes a workflow summary.
9. Uploads the generated research artifacts.

Run it from:

**GitHub → Actions → Coinbase Real-Data Research Pipeline → Run workflow**

The workflow accepts a Coinbase product and capture duration.

Generated artifacts include capture metadata, normalized data, data-quality reports, experiment reports, and Red-Team reports.

### Market-data limitations

The Coinbase collector records local arrival order and exchange timestamps where available. Its local message index is not an exchange sequence number.

The current collector does not automatically reconnect, recover missed messages, or resume from a new snapshot.

The collection and normalization pipeline should not be interpreted as a complete exchange-grade feed handler.

## 4. Order-book reconstruction and integrity

Leakguard includes an exchange-neutral normalized order-book reconstruction interface.

It supports:

- Initial snapshots.
- Incremental price-level updates.
- Price-level insertion and replacement.
- Zero-size price-level deletion.
- Snapshot sequence tracking.
- Missing sequence detection.
- Duplicate update detection.
- Out-of-order update detection.

It also checks market-data integrity, including:

- Missing values.
- Duplicate timestamps.
- Out-of-order timestamps.
- Invalid prices and sizes.
- Crossed order books.

The generic reconstruction interface is separate from the Coinbase-specific collector. Its sequence checks should not be confused with exchange-level sequence recovery.

## 5. Causal microstructure features

The feature pipeline includes:

- Bid-ask spread.
- Level-one order-book imbalance.
- Multi-level depth imbalance.
- Rolling imbalance.
- Microprice.
- Short-horizon returns.
- Rolling volatility.
- Spread in basis points.

Leakguard also includes a deliberately leaky feature for testing whether its causality diagnostics can identify future-dependent information.

The causality diagnostic compares feature outputs across complete and truncated input histories.

This is a practical diagnostic, not a mathematical proof that every possible leakage mechanism has been eliminated.

## 6. Leakage-aware validation

Random cross-validation can produce misleading results for time-series prediction when label windows overlap.

Leakguard includes:

- Chronological walk-forward evaluation.
- Forward-only test periods.
- Purging of overlapping training-label windows.
- Configurable embargo periods.
- Naive K-fold comparison.
- Label-overlap diagnostics.

The purpose is to reduce artificially optimistic model evaluation caused by temporal leakage.

## 7. Machine-learning baselines

The experiment pipeline compares:

- Logistic Regression.
- Histogram-based Gradient Boosting.

Evaluation metrics include:

- ROC AUC.
- Accuracy.
- Balanced accuracy.
- Brier score.

The pipeline produces out-of-sample predictions and fold-level metrics.

## 8. Statistical reliability

A model's average score alone does not tell the whole story.

Leakguard includes several additional diagnostics.

### Volatility-regime evaluation

The research pipeline evaluates predictions across:

- Low-volatility conditions.
- Normal-volatility conditions.
- High-volatility conditions.

Regime thresholds are calculated from training-fold volatility and then applied to the test observations.

This avoids using the test period to determine its own regime boundaries.

### Confidence intervals

Leakguard implements circular moving-block bootstrap confidence intervals for selected evaluation metrics.

The current experiment reports intervals for:

- ROC AUC.
- Accuracy.
- Brier score.

Blocks preserve local observation ordering better than independently resampling every observation.

These are approximate uncertainty diagnostics, not guarantees of statistical independence or proof of generalization.

### Fold stability

The experiment reports:

- Mean metric across validation folds.
- Standard deviation.
- Minimum.
- Maximum.
- Number of valid folds.

This makes variation between validation periods visible instead of relying exclusively on a pooled score.

## 9. Null experiments and selection bias

Leakguard includes two separate controlled statistical experiments.

### Independent-label null experiment

Module:

`leakguard/null_experiments.py`

This experiment generates independent random labels and prediction scores.

Because the scores contain no predictive information about the labels, the resulting AUC distribution provides a chance-level reference.

Run:

```bash
python -m leakguard.null_experiments \
  --observations 1000 \
  --repetitions 500 \
  --seed 7
```

The report includes:

- Mean and median AUC.
- AUC standard deviation.
- 2.5th and 97.5th percentiles.
- Fraction of trials with AUC at least 0.60.
- Counts above, below, and equal to 0.50.

These percentiles describe the simulated null distribution. They are not confidence intervals for a real market model.

### Multiple-trial selection-bias experiment

Module:

`leakguard/selection_bias.py`

This experiment evaluates multiple independent random predictors against random labels and selects the best observed AUC.

It demonstrates how selecting the best result from many experiments can produce an apparently strong score even when no genuine predictive signal exists.

Run:

```bash
python -m leakguard.selection_bias \
  --n-observations 300 \
  --n-trials 20 \
  --repetitions 30 \
  --seed 7
```

The report includes:

- Mean selected AUC.
- Median selected AUC.
- 95th percentile selected AUC.
- Mean unselected trial AUC.
- Selection uplift relative to the theoretical null AUC.
- Repetitions in which the selected result exceeds the null.

These are controlled synthetic demonstrations. They do not reproduce every form of multiple testing, temporal dependence, or financial non-stationarity.

## 10. Red-Team validation

Module:

`leakguard/redteam.py`

Leakguard includes controlled fault-injection challenges that test whether known data-integrity and feature-causality problems are detected.

Challenge cases include:

- Duplicate timestamps.
- Out-of-order timestamps.
- Missing market values.
- Crossed order books.
- Negative order sizes.
- Deliberately future-dependent features.

Clean controls are included to check that valid synthetic inputs are not incorrectly flagged.

Run:

```bash
python -m leakguard.redteam \
  --n 1200 \
  --seed 7
```

The report records:

- Challenge cases.
- Expected detections.
- Observed detections.
- Passed and failed cases.
- Overall challenge pass rate.

The pass rate applies only to the explicitly implemented challenge cases. It is not a general estimate of detection recall or proof that arbitrary real-world market-data corruption will be detected.

## 11. Runtime and memory benchmarks

Module:

`leakguard/benchmark.py`

Leakguard includes a reproducible benchmark for:

1. Synthetic order-book generation.
2. Causal feature construction.

Default benchmark sizes:

- 10,000 rows.
- 30,000 rows.
- 60,000 rows.

Run:

```bash
python -m leakguard.benchmark
```

Custom row counts:

```bash
python -m leakguard.benchmark --rows 5000 20000 100000
```

The benchmark records:

- Generation wall time.
- Feature-construction wall time.
- Python-tracked peak memory.
- Input DataFrame memory.
- Feature DataFrame memory.
- Python and dependency versions.
- Random seed.

Reports:

```text
results/
    benchmarks/
        latest.json
        latest.md
```

Wall time is measured using `time.perf_counter`.

Peak traced memory is measured using Python's `tracemalloc`. This is not total process resident memory.

These benchmarks cover synthetic generation and feature construction. They do not measure model-training throughput, exchange-feed throughput, network latency, or production trading capacity.

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the measurement methodology and limitations.

## 12. Installation

Requires Python and the dependencies listed in `requirements.txt`.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## 13. Running the research experiment

### Synthetic experiment

```bash
python -m leakguard.experiment \
  --n 30000 \
  --horizon 50
```

### Experiment using normalized order-book data

```bash
python -m leakguard.experiment \
  --data data/normalized_order_book.csv \
  --horizon 50 \
  --splits 5 \
  --fee-bps 1 \
  --slippage-bps 1
```

The normalized CSV input requires:

```text
ts_ms
bid_px_1
ask_px_1
bid_sz_1
ask_sz_1
```

Optional additional depth columns:

```text
bid_px_2
ask_px_2
bid_sz_2
ask_sz_2
bid_px_3
ask_px_3
bid_sz_3
ask_sz_3
```

Each row must represent one consistently sampled order-book observation.

Timestamps must be ascending and unique. Prices and sizes must use consistent units. The best bid must be below the best ask.

The raw REST snapshot and WebSocket capture outputs should not be assumed to be directly compatible with the experiment loader. They must pass through the normalization process first.

The prediction horizon is measured in observations, not automatically in seconds. Its time interpretation depends on the sampling interval of the input data.

## 14. Research reports

The experiment pipeline generates Markdown and JSON reports containing:

- Data-source information.
- Input row counts.
- Data fingerprint.
- Integrity diagnostics.
- Feature-causality results.
- Purged validation information.
- Out-of-sample model metrics.
- Confidence intervals.
- Volatility-regime results.
- Fold-stability summaries.
- Simplified cost diagnostics.

Other modules generate separate reports for:

- Null experiments.
- Selection-bias experiments.
- Red-Team challenges.
- Runtime and memory benchmarks.
- Data-quality audits.

Reports are designed to make assumptions and limitations inspectable rather than presenting a single model score without context.

## 15. Testing and continuous integration

Leakguard uses automated tests and GitHub Actions.

The test suite covers areas including:

- Synthetic order-book generation.
- Data-integrity checks.
- Feature causality.
- Deliberately leaky features.
- Dataset construction.
- Purged chronological validation.
- Label-overlap diagnostics.
- Snapshot and incremental-update reconstruction.
- Sequence-gap, duplicate, and out-of-order detection.
- Coinbase market-data parsing and collection behavior.
- Normalization and data-quality auditing.
- Statistical reliability.
- Red-Team challenges.
- Null experiments.
- Selection-bias experiments.
- Benchmark reporting.

Run locally:

```bash
python -m pytest -q
```

GitHub Actions runs the automated CI workflow on repository changes.

The project also contains manually triggered workflows for public market-data collection and the Coinbase real-data research pipeline.

## 16. Limitations and research boundaries

Leakguard is an independent research prototype.

It does not currently claim:

- Production exchange connectivity or exchange-grade feed recovery.
- Complete sequence-gap recovery for the Coinbase WebSocket collector.
- Level 3 order-book reconstruction.
- Queue-position estimation.
- Realistic partial-fill simulation.
- Market-impact modelling.
- Latency or adverse-selection modelling.
- Production order execution.
- Profitable alpha.
- Live-trading readiness.

Additional statistical limitations:

- Synthetic data does not replace extensive historical market data.
- Bootstrap intervals are approximate.
- Fold stability does not guarantee future stability.
- Null experiments are controlled simulations.
- Selection-bias experiments do not model every real-world research-selection process.
- Red-Team results cover only known challenge cases.
- Simplified cost diagnostics are not realistic execution simulations.

The project is intended to investigate the reliability of research assumptions, not to claim that a trading strategy works.

## 17. Repository structure

```text
leakguard/
├── .github/
│   └── workflows/
├── docs/
│   └── BENCHMARKS.md
├── leakguard/
│   ├── benchmark.py
│   ├── checks.py
│   ├── coinbase_ws.py
│   ├── coinbase_normalize.py
│   ├── cv.py
│   ├── data.py
│   ├── data_quality.py
│   ├── experiment.py
│   ├── features.py
│   ├── market_data.py
│   ├── metrics.py
│   ├── models.py
│   ├── null_experiments.py
│   ├── reliability.py
│   ├── redteam.py
│   ├── selection_bias.py
│   └── simulate.py
├── tests/
├── requirements.txt
└── README.md
```

## 18. Project objective

Leakguard is an order-book ML research-validation toolkit built around one principle:

**A model result should not be trusted until the data, features, validation methodology, and evaluation assumptions have been challenged.**

The project combines market-data engineering, time-series validation, statistical diagnostics, controlled fault injection, and reproducible benchmarking into one research workflow.

It is an independent engineering project demonstrating work in Python, market microstructure data handling, quantitative ML validation, testing, and research reproducibility.

It is not presented as a production HFT engine or as evidence of profitable trading alpha.
