
import numpy as np
import pytest

from leakguard.reliability import (
    block_bootstrap_ci,
    volatility_regimes,
)


def test_volatility_regimes_use_training_thresholds():
    train = np.arange(1, 10, dtype=float)
    test = np.array([1, 3, 5, 7, 9], dtype=float)

    labels, thresholds = volatility_regimes(train, test)

    assert labels.tolist() == [
        "low",
        "low",
        "normal",
        "high",
        "high",
    ]

    assert thresholds["low_volatility_cutoff"] < (
        thresholds["high_volatility_cutoff"]
    )


def test_volatility_regimes_reject_empty_training_data():
    with pytest.raises(ValueError, match="cannot be empty"):
        volatility_regimes([], [1, 2, 3])


def test_volatility_regimes_reject_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        volatility_regimes([1, 2, np.nan], [1, 2])


def test_block_bootstrap_accuracy_interval():
    y = np.tile([0, 1], 100)
    probabilities = np.where(y == 1, 0.8, 0.2)

    result = block_bootstrap_ci(
        y,
        probabilities,
        metric="accuracy",
        block_size=10,
        n_bootstrap=100,
        seed=7,
    )

    assert result["estimate"] == 1.0
    assert result["lower"] == 1.0
    assert result["upper"] == 1.0
    assert result["bootstrap_samples_valid"] == 100


def test_block_bootstrap_rejects_invalid_probabilities():
    y = np.array([0, 1, 0, 1])
    probabilities = np.array([0.2, 1.5, 0.3, 0.8])

    with pytest.raises(ValueError, match="between 0 and 1"):
        block_bootstrap_ci(
            y,
            probabilities,
            metric="accuracy",
        )


def test_block_bootstrap_rejects_unsupported_metric():
    y = np.tile([0, 1], 20)
    probabilities = np.where(y == 1, 0.8, 0.2)

    with pytest.raises(ValueError, match="metric must be"):
        block_bootstrap_ci(
            y,
            probabilities,
            metric="unknown",
        )
