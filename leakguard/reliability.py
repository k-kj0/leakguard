
"""Statistical reliability tools for Leakguard research experiments."""

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)


def volatility_regimes(train_volatility, test_volatility):
    """Classify test observations using volatility thresholds from training data.

    Thresholds are calculated only from the training sample.
    This prevents the test period from determining its own regime boundaries.
    """
    train = np.asarray(train_volatility, dtype=float)
    test = np.asarray(test_volatility, dtype=float)

    if train.ndim != 1 or test.ndim != 1:
        raise ValueError("Volatility inputs must be one-dimensional")

    if len(train) == 0 or len(test) == 0:
        raise ValueError("Training and test volatility cannot be empty")

    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("Volatility inputs must contain finite values")

    low_cutoff, high_cutoff = np.quantile(
        train,
        [1 / 3, 2 / 3],
    )

    labels = np.where(
        test <= low_cutoff,
        "low",
        np.where(
            test <= high_cutoff,
            "normal",
            "high",
        ),
    )

    thresholds = {
        "low_volatility_cutoff": float(low_cutoff),
        "high_volatility_cutoff": float(high_cutoff),
    }

    return labels, thresholds


def _score(y_true, probabilities, metric):
    """Calculate one supported classification metric."""
    if metric == "accuracy":
        predictions = (probabilities >= 0.5).astype(int)
        return float(accuracy_score(y_true, predictions))

    if metric == "balanced_accuracy":
        predictions = (probabilities >= 0.5).astype(int)
        return float(balanced_accuracy_score(y_true, predictions))

    if metric == "brier":
        return float(brier_score_loss(y_true, probabilities))

    if metric == "auc":
        if len(np.unique(y_true)) != 2:
            raise ValueError("AUC requires both target classes")
        return float(roc_auc_score(y_true, probabilities))

    raise ValueError(
        "metric must be auc, accuracy, balanced_accuracy, or brier"
    )


def block_bootstrap_ci(
    y_true,
    probabilities,
    metric="auc",
    block_size=50,
    n_bootstrap=500,
    confidence=0.95,
    seed=7,
):
    """Estimate a metric and its confidence interval using circular blocks.

    Contiguous observations are sampled together instead of independently.
    This is a simple uncertainty diagnostic, not a guarantee of statistical
    independence or a substitute for chronological out-of-sample testing.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)

    if y.ndim != 1 or p.ndim != 1:
        raise ValueError("Targets and probabilities must be one-dimensional")

    if len(y) != len(p) or len(y) < 2:
        raise ValueError("Targets and probabilities must have equal length >= 2")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Targets must contain only 0 and 1")

    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1")

    if block_size < 1:
        raise ValueError("block_size must be positive")

    if n_bootstrap < 20:
        raise ValueError("Use at least 20 bootstrap repetitions")

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    block_size = min(int(block_size), len(y))

    estimate = _score(y, p, metric)

    rng = np.random.default_rng(seed)
    n = len(y)
    number_of_blocks = math.ceil(n / block_size)

    scores = []

    for _ in range(n_bootstrap):
        starts = rng.integers(
            0,
            n,
            size=number_of_blocks,
        )

        indices = np.concatenate([
            (start + np.arange(block_size)) % n
            for start in starts
        ])[:n]

        try:
            score = _score(
                y[indices],
                p[indices],
                metric,
            )
        except ValueError:
            # For example, an AUC sample may contain only one class.
            continue

        scores.append(score)

    if len(scores) < 20:
        raise ValueError(
            "Too few valid bootstrap samples; check target balance "
            "or increase the sample size"
        )

    alpha = 1.0 - confidence

    lower = float(np.quantile(scores, alpha / 2))
    upper = float(np.quantile(scores, 1 - alpha / 2))

    return {
        "metric": metric,
        "estimate": float(estimate),
        "confidence_level": float(confidence),
        "lower": lower,
        "upper": upper,
        "block_size": block_size,
        "bootstrap_samples_requested": int(n_bootstrap),
        "bootstrap_samples_valid": len(scores),
        "method": "circular moving-block bootstrap",
    }
