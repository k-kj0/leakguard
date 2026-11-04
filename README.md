# leakguard

Catching look-ahead leakage in order-book ML pipelines.

## The problem
A model that predicts short-horizon price moves can look great in validation and
fail live. Two usual causes: (1) a feature that uses future data, (2) shuffled
cross-validation on overlapping labels. leakguard detects both.

## What it does
- `simulate.py`: synthetic order book with a known, weak signal
- `features.py`: honest features plus one deliberately leaky feature for testing
- `cv.py`: shuffled K-fold (wrong) vs purged walk-forward with embargo (right)
- `checks.py`: truncation causality test, label-overlap measure, shuffled-label
  sanity test, data integrity checks
- `experiment.py`: runs everything, logs params, data hash and library versions

## Run it
    pip install -r requirements.txt
    pytest -q
    python -m leakguard.experiment --n 60000 --horizon 50 --seed 7
Report is written to `results/latest.md`.

## Results (n=30000, horizon=50, seed=7)
| Setup | AUC |
|---|---|
| Shuffled K-fold (wrong) | 0.616 |
| Purged walk-forward (honest) | 0.558 |
| Honest + leaky feature | 0.712 |

Shuffled K-fold overstates skill by about 0.06 AUC. 100% of its training rows
share label windows with test rows. Purged walk-forward has 0%.

## What I would do differently
- Simulated data has no regime changes or real microstructure. Next step: run on
  public crypto order-book data (e.g. Binance) and compare.
- Add a baseline (logistic regression) so gains from the model are visible.
- Add transaction-cost-aware evaluation: AUC alone is not tradable.
