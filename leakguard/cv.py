"""Cross-validation that respects time.

A label at row i is built from prices up to row i+horizon, so label windows of
neighbouring rows OVERLAP. Ordinary shuffled K-fold puts one neighbour in train
and the other in test, so the model effectively sees test answers. Fix:
walk forward in time and PURGE (drop) training rows whose label window
reaches into the test block, plus an optional EMBARGO gap.
"""
import numpy as np
from sklearn.model_selection import KFold


def naive_kfold(n: int, n_splits: int = 5, seed: int = 0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in kf.split(np.arange(n))]


def purged_walk_forward(n: int, horizon: int, n_splits: int = 5,
                        initial_train_frac: float = 0.4, embargo: int = 0):
    start = int(n * initial_train_frac)
    edges = np.linspace(start, n, n_splits + 1).astype(int)
    splits = []
    for a, b in zip(edges[:-1], edges[1:]):
        train_end = a - horizon - embargo      # last train label window ends before test starts
        if train_end <= 0:
            continue
        splits.append((np.arange(0, train_end), np.arange(a, b)))
    return splits
