"""Simulate a limit order book with a weak, real, learnable signal.

Why simulate? Real order-book data is huge and licensed. A simulator lets us
KNOW the truth: the future price drifts a little in the direction of a hidden
"order pressure" that we only see through noisy book sizes. So an honest model
should score modestly above 50%, and any score far above that is a red flag.
"""
import numpy as np
import pandas as pd


def simulate_order_book(n: int = 60_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Hidden order pressure: slowly changing, never shown directly.
    eps = rng.normal(size=n)
    pressure = np.zeros(n)
    for t in range(1, n):
        pressure[t] = 0.97 * pressure[t - 1] + 0.25 * eps[t]

    # Price change at t depends on pressure at t-1 plus noise.
    ret = np.zeros(n)
    ret[1:] = 0.004 * pressure[:-1] + 0.05 * rng.normal(size=n - 1)
    mid = 100.0 + np.cumsum(ret)

    spread = rng.choice([0.01, 0.02, 0.03], size=n, p=[0.6, 0.3, 0.1])
    bid1 = mid - spread / 2
    ask1 = mid + spread / 2

    data = {
        "ts_ms": 1_700_000_000_000 + np.cumsum(1 + rng.exponential(50, size=n)).astype("int64"),
    }
    for lvl in (1, 2, 3):
        data[f"bid_px_{lvl}"] = bid1 - 0.01 * (lvl - 1)
        data[f"ask_px_{lvl}"] = ask1 + 0.01 * (lvl - 1)
        # Visible sizes lean with the hidden pressure, but noisily.
        data[f"bid_sz_{lvl}"] = np.exp(5.0 + 0.5 * pressure + 0.6 * rng.normal(size=n)).round() + 1
        data[f"ask_sz_{lvl}"] = np.exp(5.0 - 0.5 * pressure + 0.6 * rng.normal(size=n)).round() + 1
    return pd.DataFrame(data)
