# Leakguard — Order-Book ML Research Validation

**A research-engineering toolkit for testing whether short-horizon order-book machine-learning experiments are trustworthy.**

Leakguard focuses on a critical problem in quantitative ML: a model's reported performance is meaningless if its market data is corrupted, its features contain future information, or its validation methodology leaks information across time.

The project combines:

- Order-book snapshot integrity checks.
- Snapshot and incremental-update reconstruction.
- Sequence-gap, duplicate and out-of-order update detection.
- Causal microstructure feature engineering.
- Future-information leakage checks.
- Purged chronological validation with embargo.
- Logistic regression and gradient-boosting baselines.
- Out-of-sample evaluation and simplified transaction-cost diagnostics.
- Reproducible experiment reports.
- Automated tests and GitHub Actions CI.

**Project status:** Research prototype. Not a production HFT engine, exchange connector, execution system, or proof of profitable alpha.

---

## 1. Research question

Can order-book imbalance and microstructure features predict short-horizon price movements out of sample after accounting for:

1. Data integrity problems.
2. Future-information leakage.
3. Overlapping prediction labels.
4. Transaction costs.
5. Changes in market conditions?

Leakguard is designed to make these experiments easier to inspect, reproduce and challenge.

## 2. Architecture

```text
Market Data
    |
    v
Data Integrity Checks
    |
    v
Order-Book Reconstruction
    |
    v
Sequence Continuity Audit
    |
    v
Causal Feature Engineering
    |
    v
Leakage Detection
    |
    v
Purged Walk-Forward Validation
    |
    v
ML Baselines
    |
    v
Out-of-Sample Evaluation
    |
    v
Research Report
```

## 3. Order-book reconstruction

Leakguard supports reconstruction from a normalized initial snapshot and incremental price-level updates.

The snapshot represents the book immediately after a specified sequence number. Each subsequent update modifies one bid or ask price level.

### Snapshot format

| side | price | size |
|---|---:|---:|
| bid | 99.00 | 10 |
| bid | 98.00 | 20 |
| bid | 97.00 | 30 |
| ask | 101.00 | 11 |
| ask | 102.00 | 21 |
| ask | 103.00 | 31 |

### Incremental update format

| sequence | ts_ms | side | price | size |
|---:|---:|---|---:|---:|
| 101 | 1001 | bid | 99.00 | 0 |
| 102 | 1002 | ask | 101.00 | 15 |
| 103 | 1003 | bid | 100.00 | 5 |

A zero size deletes a price level. A positive size inserts or replaces a price level.


## Public Market Data Integration

Leakguard includes a public Coinbase Exchange Level 2 snapshot adapter.

It fetches aggregated bid and ask levels without API credentials,
validates the response, preserves the exchange sequence number,
and normalizes the book into a side/price/size table.

### Run locally

```bash
python -m leakguard.market_data \
  --product BTC-USD \
  --depth 10 \
  --out data/coinbase_btc_usd_snapshot.csv
```

The command writes:
- A normalized CSV snapshot.
- A JSON metadata sidecar containing the exchange, product,
  sequence number, depth and local receipt timestamp.

### Run through GitHub Actions

Open Actions → Fetch Public Market Data → Run workflow.

Choose a product ID such as BTC-USD and the requested depth.
The resulting CSV and JSON are uploaded as a workflow artifact.

### Data limitations

A REST snapshot is a point-in-time observation, not a historical
time series. The snapshot's local receipt timestamp is not the
exchange event timestamp.

The current adapter does not yet subscribe to WebSocket updates,
recover missed messages, or guarantee continuity between snapshots.
Those capabilities belong to the exchange-specific streaming
and reconstruction layer.

### Sequence auditing

The sequence auditor detects:

- Missing sequence numbers.
- Duplicate updates.
- Out-of-order updates.

Reconstruction refuses to continue across an invalid sequence stream instead of silently producing potentially incorrect book states.

### Python usage

```python
import pandas as pd

from leakguard.reconstruction import (
    audit_update_sequence,
    reconstruct_order_book,
)

snapshot = pd.DataFrame([
    {"side": "bid", "price": 99, "size": 10},
    {"side": "bid", "price": 98, "size": 20},
    {"side": "bid", "price": 97, "size": 30},
    {"side": "ask", "price": 101, "size": 11},
    {"side": "ask", "price": 102, "size": 21},
    {"side": "ask", "price": 103, "size": 31},
])

updates = pd.DataFrame([
    {
        "sequence": 101,
        "ts_ms": 1001,
        "side": "bid",
        "price": 99,
        "size": 0,
    },
    {
        "sequence": 102,
        "ts_ms": 1002,
        "side": "ask",
        "price": 101,
        "size": 15,
    },
])

audit = audit_update_sequence(
    updates,
    snapshot_sequence=100,
)

print(audit)

book_states = reconstruct_order_book(
    snapshot,
    updates,
    snapshot_sequence=100,
    depth=3,
)

print(book_states)
```

This is a normalized research interface, not a live exchange feed adapter.

## 4. Feature engineering

The feature pipeline includes:

- Bid-ask spread.
- Level-one order-book imbalance.
- Multi-level depth imbalance.
- Rolling imbalance.
- Microprice.
- Short-horizon returns.
- Rolling volatility.
- Spread in basis points.

Feature causality is checked by comparing feature outputs on complete and truncated input histories.

The causality check is a diagnostic, not a mathematical proof that every possible future-information leak has been eliminated.

## 5. Leakage-aware validation

Random cross-validation is generally inappropriate for overlapping time-series prediction labels.

Leakguard includes chronological validation with:

- Forward-only test periods.
- Purging of training observations whose label windows overlap the test period.
- An additional configurable embargo.
- A naive K-fold comparison for illustrating the risk of temporal leakage.

The goal is to avoid reporting artificially optimistic results caused by overlapping future information.

## 6. Machine-learning baselines

The experiment pipeline compares:

- Logistic Regression.
- Histogram-based Gradient Boosting.

Evaluation includes:

- ROC AUC where defined.
- Accuracy.
- Balanced accuracy.
- Brier score.

A simplified transaction-cost diagnostic can account for spread, fees and slippage assumptions.

It is not a realistic execution simulator or a profitability estimate.

## 7. Run the project

### Windows PowerShell

Create a virtual environment:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run tests:

```powershell
python -m pytest -q
```

Run the synthetic experiment:

```powershell
python -m leakguard.experiment --n 30000 --horizon 50
```

The experiment generates:

```text
results/
    latest.md
    latest.json
```

### Run on an order-book CSV

```powershell
python -m leakguard.experiment --data data/sample_order_book.csv --horizon 50 --fee-bps 1 --slippage-bps 1
```

Required columns:

- `ts_ms`
- `bid_px_1`
- `ask_px_1`
- `bid_sz_1`
- `ask_sz_1`

Optional depth columns:

- `bid_px_2`
- `ask_px_2`
- `bid_sz_2`
- `ask_sz_2`
- `bid_px_3`
- `ask_px_3`
- `bid_sz_3`
- `ask_sz_3`

Each row represents one consistently sampled book observation.

Timestamps must be ascending and unique. Prices and sizes must use consistent units. The best bid must be below the best ask.

## 8. Automated testing

The repository includes tests for:

- Synthetic order-book integrity.
- Crossed-book detection.
- Feature causality.
- Deliberately leaky features.
- Dataset alignment.
- Purged chronological splits.
- Naive K-fold label overlap.
- Incremental order-book reconstruction.
- Sequence gaps, duplicates and out-of-order updates.
- Invalid updates and crossed reconstructed books.

GitHub Actions automatically installs dependencies, executes the test suite and runs the research experiment when changes are pushed.

## 9. Limitations

The current implementation has important limitations:

- The built-in market simulator is synthetic.
- CSV ingestion expects normalized snapshots, not raw exchange messages.
- Reconstruction supports normalized price-level updates, not every exchange protocol.
- Exchange checksums, order IDs, packet recovery and exchange-specific event semantics are not implemented.
- The label is binary price direction, not expected return.
- The cost diagnostic is not a realistic fill simulator.
- Queue position, partial fills, market impact, latency and adverse selection are not modeled.
- Regime-based evaluation and statistical confidence intervals remain future work.
- No profitability or live-trading-readiness claim is made.

## 10. Roadmap

- [x] Synthetic order-book generation.
- [x] Snapshot integrity checks.
- [x] Causal microstructure features.
- [x] Leakage diagnostics.
- [x] Purged chronological validation.
- [x] ML baseline comparisons.
- [x] Reproducible experiment reports.
- [x] Automated CI tests.
- [x] Normalized order-book reconstruction.
- [x] Sequence continuity auditing.
- [ ] Real public market-data integration.
- [ ] Exchange-specific snapshot and delta handling.
- [ ] Regime-based performance evaluation.
- [ ] Confidence intervals and statistical robustness.
- [ ] Execution simulation.
- [ ] Larger-data and memory benchmarks.

---

**Leakguard is an independent research-engineering project.** Its purpose is to make short-horizon ML research more auditable by testing the integrity of the data, features, validation and evaluation process before trusting model results.
