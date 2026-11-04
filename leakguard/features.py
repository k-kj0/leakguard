"""Features and labels. Every feature here only looks at the PAST (rows <= t)."""
import numpy as np
import pandas as pd


def mid_price(df: pd.DataFrame) -> pd.Series:
    return (df["bid_px_1"] + df["ask_px_1"]) / 2


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    mid = mid_price(df)
    f = pd.DataFrame(index=df.index)
    f["spread"] = df["ask_px_1"] - df["bid_px_1"]

    b1, a1 = df["bid_sz_1"], df["ask_sz_1"]
    f["imb_l1"] = (b1 - a1) / (b1 + a1)

    b3 = df[[f"bid_sz_{i}" for i in (1, 2, 3)]].sum(axis=1)
    a3 = df[[f"ask_sz_{i}" for i in (1, 2, 3)]].sum(axis=1)
    f["imb_l3"] = (b3 - a3) / (b3 + a3)
    f["imb_l3_ma20"] = f["imb_l3"].rolling(20).mean()

    micro = (df["bid_px_1"] * a1 + df["ask_px_1"] * b1) / (b1 + a1)
    f["micro_minus_mid"] = micro - mid

    f["ret_5"] = mid - mid.shift(5)      # shift(+k) = look BACK k rows (safe)
    f["ret_20"] = mid - mid.shift(20)
    f["vol_20"] = mid.diff().rolling(20).std()
    return f


def build_features_leaky(df: pd.DataFrame, peek: int = 10) -> pd.DataFrame:
    """DEMO ONLY: same features plus one that peeks `peek` rows into the future.
    shift(-k) = look FORWARD k rows. This is the classic leakage bug."""
    f = build_features(df)
    mid = mid_price(df)
    f["LEAK_future_ret"] = mid.shift(-peek) - mid
    return f


def build_dataset(df: pd.DataFrame, horizon: int, feature_fn=build_features):
    """Return X, y with clean 0..n-1 index. y = 1 if mid price is higher `horizon` rows later."""
    X = feature_fn(df)
    mid = mid_price(df)
    future = mid.shift(-horizon)
    y = (future > mid).astype(int)
    keep = X.notna().all(axis=1) & future.notna()
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    return X, y
