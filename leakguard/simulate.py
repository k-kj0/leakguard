"""Deterministic synthetic order book for tests and reproducible demos only."""

import numpy as np
import pandas as pd


def simulate_order_book(n: int = 60_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pressure = np.zeros(n)
    eps = rng.normal(size=n)

    for t in range(1, n):
        pressure[t] = 0.97 * pressure[t - 1] + 0.25 * eps[t]

    returns = np.zeros(n)
    returns[1:] = (
        0.004 * pressure[:-1]
        + 0.05 * rng.normal(size=n - 1)
    )

    mid = 100.0 + np.cumsum(returns)

    spread = rng.choice(
        [0.01, 0.02, 0.03],
        size=n,
        p=[0.6, 0.3, 0.1],
    )

    bid = mid - spread / 2
    ask = mid + spread / 2

    ts = (
        1_700_000_000_000
        + np.cumsum(
            1 + rng.exponential(50, size=n)
        ).astype("int64")
    )

    data = {"ts_ms": ts}

    for level in (1, 2, 3):
        data[f"bid_px_{level}"] = (
            bid - 0.01 * (level - 1)
        )

        data[f"ask_px_{level}"] = (
            ask + 0.01 * (level - 1)
        )

        data[f"bid_sz_{level}"] = (
            np.exp(
                5 + 0.5 * pressure
                + 0.6 * rng.normal(size=n)
            ).round() + 1
        )

        data[f"ask_sz_{level}"] = (
            np.exp(
                5 - 0.5 * pressure
                + 0.6 * rng.normal(size=n)
            ).round() + 1
        )

    return pd.DataFrame(data)
