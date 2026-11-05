"""Chronological walk-forward validation with label-window purging."""

import numpy as np


def naive_kfold(n: int, n_splits: int = 5, seed: int = 0):
    """Negative-control split: shuffled and unsuitable for time series."""

    from sklearn.model_selection import KFold

    return [
        (tr, te)
        for tr, te in KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        ).split(np.arange(n))
    ]


def purged_walk_forward(
    n: int,
    horizon: int,
    n_splits: int = 5,
    initial_train_frac: float = 0.4,
    embargo: int = 0,
):
    """Expanding-window splits with training-label purging."""

    if n <= 0 or horizon < 1 or n_splits < 1:
        raise ValueError(
            "n, horizon and n_splits must be positive"
        )

    if not 0 < initial_train_frac < 1:
        raise ValueError(
            "initial_train_frac must be between 0 and 1"
        )

    if embargo < 0:
        raise ValueError("embargo cannot be negative")

    edges = np.linspace(
        int(n * initial_train_frac),
        n,
        n_splits + 1,
    ).astype(int)

    splits = []

    for start, end in zip(edges[:-1], edges[1:]):

        train_end = start - horizon - embargo

        if train_end <= 0 or end <= start:
            continue

        train = np.arange(0, train_end)
        test = np.arange(start, end)

        if len(train) and len(test):
            splits.append((train, test))

    return splits


def label_overlap_fraction(
    train_idx,
    test_idx,
    horizon: int,
    n: int,
) -> float:
    """Fraction of training-label intervals touching test-label intervals."""

    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)

    if len(train_idx) == 0:
        return 0.0

    test_starts = np.sort(test_idx)

    overlap = np.zeros(len(train_idx), dtype=bool)

    for j, i in enumerate(train_idx):

        left = np.searchsorted(
            test_starts,
            i - horizon,
            side="left",
        )

        right = np.searchsorted(
            test_starts,
            i + horizon,
            side="right",
        )

        overlap[j] = right > left

    return float(overlap.mean())
