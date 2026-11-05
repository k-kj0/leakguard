"""Causal order-book features and forward-return labels."""

import pandas as pd


def mid_price(df: pd.DataFrame) -> pd.Series:
    return (
        df["bid_px_1"] + df["ask_px_1"]
    ) / 2.0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Every feature at row t uses data available at or before row t."""

    mid = mid_price(df)

    f = pd.DataFrame(index=df.index)

    f["spread"] = (
        df["ask_px_1"] - df["bid_px_1"]
    )

    b1 = df["bid_sz_1"]
    a1 = df["ask_sz_1"]

    f["imb_l1"] = (
        (b1 - a1) / (b1 + a1).replace(0, float("nan"))
    )

    bid_size_cols = [
        c for c in ("bid_sz_1", "bid_sz_2", "bid_sz_3")
        if c in df
    ]

    ask_size_cols = [
        c for c in ("ask_sz_1", "ask_sz_2", "ask_sz_3")
        if c in df
    ]

    bdepth = df[bid_size_cols].sum(axis=1)
    adepth = df[ask_size_cols].sum(axis=1)

    f["imb_depth"] = (
        (bdepth - adepth)
        / (bdepth + adepth).replace(0, float("nan"))
    )

    f["imb_l1_ma20"] = (
        f["imb_l1"].rolling(20, min_periods=20).mean()
    )

    f["imb_depth_ma20"] = (
        f["imb_depth"].rolling(20, min_periods=20).mean()
    )

    f["microprice"] = (
        df["ask_px_1"] * b1 + df["bid_px_1"] * a1
    ) / (b1 + a1).replace(0, float("nan"))

    f["micro_minus_mid"] = f["microprice"] - mid

    f["mid_return_5"] = mid.pct_change(5)
    f["mid_return_20"] = mid.pct_change(20)

    f["volatility_20"] = (
        mid.pct_change()
        .rolling(20, min_periods=20)
        .std()
    )

    f["spread_bps"] = f["spread"] / mid * 10_000

    return f


def build_features_leaky(
    df: pd.DataFrame,
    peek: int = 10,
) -> pd.DataFrame:
    """Intentionally invalid feature used only to test leakage detection."""

    f = build_features(df)
    mid = mid_price(df)

    f["LEAK_future_ret"] = (
        mid.shift(-peek) / mid - 1.0
    )

    return f


def build_dataset(
    df: pd.DataFrame,
    horizon: int,
    feature_fn=build_features,
    return_meta: bool = False,
):
    """Predict whether mid-price rises over the next horizon observations."""

    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    features = feature_fn(df)
    mid = mid_price(df)

    future_mid = mid.shift(-horizon)

    y = (future_mid > mid).astype("int8")

    keep = features.notna().all(axis=1) & future_mid.notna()

    X = features.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)

    if not return_meta:
        return X, y

    meta = pd.DataFrame({
        "mid": mid.loc[keep].reset_index(drop=True),
        "future_mid": future_mid.loc[keep].reset_index(drop=True),
        "spread": (
            df["ask_px_1"] - df["bid_px_1"]
        ).loc[keep].reset_index(drop=True),
        "ts_ms": df["ts_ms"].loc[keep].reset_index(drop=True),
    })

    return X, y, meta
