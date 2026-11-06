
# Leakguard — Order-Book ML Research Validation

**A research-engineering toolkit for testing whether short-horizon order-book machine-learning experiments are trustworthy.**

Leakguard focuses on a fundamental problem in quantitative ML: a model's reported performance is not meaningful if its market data is corrupted, its features contain future information, or its validation methodology leaks information across time.

The project brings together market-data ingestion, order-book integrity, reconstruction, causal feature engineering, leakage diagnostics, chronological validation, baseline models, and reproducible research reports.

**Project status:** Independent research prototype. Not a production HFT engine, trading system, execution platform, or proof of profitable alpha.

---

## 1. Research question

Can order-book imbalance and microstructure features predict short-horizon price movements out of sample after accounting for:

1. Market-data integrity problems.
2. Future-information leakage.
3. Overlapping prediction labels.
4. Transaction costs.
5. Changes in market conditions.

Leakguard is designed to make these experiments easier to inspect, reproduce, and challenge.

## 2. Architecture

```text
Public / Synthetic Market Data
              |
              v
       Data Ingestion
              |
              v
      Integrity Checks
              |
              v
    Order-Book Reconstruction
              |
              v
      Sequence Auditing
              |
              v
    Causal Feature Engineering
              |
              v
       Leakage Checks
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
       Research Reports
```

## 3. Market-data ingestion

Leakguard currently includes two Coinbase Exchange market-data paths.

### 3.1 Coinbase REST Level 2 snapshot

The REST adapter fetches a public Level 2 order-book snapshot without API credentials.

It:

- Requests a product's Level 2 book.
- Validates the response.
- Preserves the returned exchange sequence number.
- Normalizes bid and ask levels into a side/price/size table.
- Saves a CSV snapshot and JSON metadata sidecar.

Run locally:

```bash
python -m leakguard.market_data \
  --product BTC-USD \
  --depth 10 \
  --out data/coinbase_btc_usd_snapshot.csv
```

Run through GitHub Actions:

1. Open the repository's Actions tab.
2. Select **Fetch Public Market Data**.
3. Click **Run workflow**.
4. Choose a product such as `BTC-USD` and a depth.
5. Download the resulting workflow artifact.

A REST snapshot is a point-in-time observation, not a historical time series. Its local receipt timestamp must not be interpreted as the exchange event timestamp.

### 3.2 Coinbase Exchange Level 2 WebSocket

Leakguard also includes a public WebSocket collector for the Coinbase Exchange `level2` channel.

The collector:

- Connects to `wss://ws-feed.exchange.coinbase.com`.
- Subscribes to a selected product, such as `BTC-USD`.
- Initializes the local book from the WebSocket snapshot.
- Processes incremental `l2update` messages.
- Applies absolute sizes to price levels.
- Removes a price level when its updated size is zero.
- Validates price levels and rejects crossed or locked books.
- Records the available top-of-book depth after each accepted message.
- Saves captured states as CSV with a JSON metadata sidecar.

Run locally:

```bash
python -m leakguard.coinbase_ws \
  --product BTC-USD \
  --depth 10 \
  --duration 30 \
  --out results/coinbase_ws_capture.csv
```

Run through GitHub Actions:

1. Open **Actions**.
2. Select **Capture Coinbase WebSocket Data**.
3. Click **Run workflow**.
4. Choose the product, depth, and capture duration.
5. Open the completed run and download the artifact named `coinbase-level2-websocket-capture`.

The collector maintains the full book received from the snapshot. The `depth` parameter controls how many levels per side are written to each output record.

### WebSocket data integrity notes

Coinbase Level 2 snapshot messages contain the product and bid/ask levels. Incremental updates contain price-level changes and an exchange timestamp.

The collector distinguishes between:

- `exchange_time`: timestamp supplied by the exchange, when present.
- `captured_at_utc`: local timestamp when Leakguard records the state.
- `message_index`: local arrival-order counter for accepted snapshot/update messages.

`message_index` is not an exchange sequence number. The collector does not infer missing exchange sequence numbers or combine a REST snapshot sequence with a separate WebSocket session.

A connection failure or invalid book update stops the capture. The current collector does not automatically reconnect, recover missed messages, or resume from a new snapshot.

## 4. Normalized order-book reconstruction

Leakguard includes a separate exchange-neutral reconstruction interface.

It accepts:

- An initial normalized snapshot.
- Incremental price-level updates.
- A snapshot sequence number.
- A requested output depth.

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

The reconstruction layer checks whether the supplied update sequence contains:

- Missing sequence numbers.
- Duplicate updates.
- Out-of-order updates.

It refuses to continue across an invalid normalized sequence rather than silently producing potentially incorrect book states.

This generic reconstruction interface and the Coinbase WebSocket adapter are separate implementations. The generic interface uses an explicit sequence field; the current Coinbase WebSocket collector uses a local message index.

## 5. Feature engineering

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

## 6. Leakage-aware validation

Random cross-validation is generally inappropriate for overlapping time-series prediction labels.

Leakguard includes chronological validation with:

- Forward-only test periods.
- Purging of training observations whose label windows overlap the test period.
- A configurable embargo.
- A naive K-fold comparison to illustrate temporal leakage risk.

The objective is to avoid reporting artificially optimistic results caused by overlapping future information.

## 7. Machine-learning baselines

The experiment pipeline compares:

- Logistic Regression.
- Histogram-based Gradient Boosting.

Evaluation includes:

- ROC AUC where defined.
- Accuracy.
- Balanced accuracy.
- Brier score.

A simplified transaction-cost diagnostic can account for spread, fees, and slippage assumptions.

It is not a realistic execution simulator or a profitability estimate.

## 8. Running the research pipeline

### Install

```bash
python -m venv .venv
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Run tests

```bash
python -m pytest -q
```

### Run the synthetic experiment

```bash
python -m leakguard.experiment --n 30000 --horizon 50
```

The experiment generates:

```text
results/
    latest.md
    latest.json
```

### Run on a normalized order-book CSV

```bash
python -m leakguard.experiment \
  --data data/sample_order_book.csv \
  --horizon 50 \
  --fee-bps 1 \
  --slippage-bps 1
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

The REST snapshot CSV and WebSocket capture CSV are raw market-data outputs, not automatically compatible with this experiment loader. They require a deliberate timestamp and schema-normalization step before being used as experiment input.

## 9. Automated testing and CI

The repository includes tests for:

- Synthetic order-book integrity.
- Crossed-book detection.
- Feature causality.
- Deliberately leaky features.
- Dataset alignment.
- Purged chronological splits.
- Naive K-fold label overlap.
- Normalized incremental order-book reconstruction.
- Sequence gaps, duplicates, and out-of-order updates.
- Coinbase REST snapshot parsing.
- Coinbase WebSocket snapshot initialization.
- Absolute-size update handling.
- Zero-size price-level deletion.
- Invalid and crossed-book update rejection.
- Top-of-book record generation.

GitHub Actions runs the automated test suite and synthetic research experiment when changes are pushed.

The public market-data workflows can be triggered manually from the Actions tab.

## 10. Limitations

Leakguard is a research prototype. Important limitations remain:

- Synthetic data is not a substitute for historical exchange data.
- The REST adapter captures a point-in-time snapshot, not a historical time series.
- The WebSocket collector currently captures bounded live sessions; it does not automatically reconnect or recover missed data.
- The current WebSocket collector records local message order, not an exchange sequence number.
- Exchange-specific checksums, packet recovery, and deeper protocol-specific validation are not implemented.
- The normalized reconstruction interface is not a universal exchange protocol implementation.
- CSV experiment ingestion expects normalized, consistently sampled snapshots.
- The label is binary price direction, not expected return.
- The cost diagnostic is not a realistic fill simulator.
- Queue position, partial fills, market impact, latency, and adverse selection are not modeled.
- Regime-based evaluation and statistical confidence intervals remain future work.
- No profitability or live-trading-readiness claim is made.

## 11. Roadmap

### Data and infrastructure

- [x] Synthetic order-book generation.
- [x] Snapshot integrity checks.
- [x] Normalized snapshot and incremental-update reconstruction.
- [x] Sequence continuity auditing for normalized updates.
- [x] Coinbase REST Level 2 snapshot adapter implementation.
- [x] Coinbase Exchange Level 2 WebSocket collector implementation.
- [ ] Verify successful live WebSocket capture through GitHub Actions.
- [ ] Exchange-specific reconnect and snapshot recovery.
- [ ] Historical market-data storage and replay.
- [ ] Timestamp and schema normalization from captured market data into the experiment pipeline.

### Research reliability

- [x] Causal microstructure features.
- [x] Leakage diagnostics.
- [x] Purged chronological validation.
- [x] Logistic Regression and gradient-boosting baselines.
- [x] Reproducible experiment reports.
- [ ] Regime-based performance evaluation.
- [ ] Confidence intervals and statistical robustness.
- [ ] Multiple-testing and experiment-selection diagnostics.

### Execution realism and performance

- [ ] Execution simulation.
- [ ] Queue-position and partial-fill modeling.
- [ ] Latency and adverse-selection analysis.
- [ ] Larger-data and memory benchmarks.

---

## 12. Project objective

Leakguard is an independent research-engineering project focused on making short-horizon order-book ML experiments more auditable.

Its central objective is to test the integrity of market data, feature construction, validation methodology, and evaluation assumptions before trusting model results.

It is not presented as a production HFT engine or as evidence of profitable trading alpha.
