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


def _future_positions(
    df: pd.DataFrame,
    horizon: int,
    horizon_ms: int | None,
):
    """Row position of the observation each label looks forward to.

    Row horizon: position + horizon.
    Time horizon: first row at or after ts_ms + horizon_ms.
    Positions at or beyond len(df) have no future observation.
    """

    n = len(df)
    positions = pd.Series(range(n), index=df.index)

    if horizon_ms is None:
        return positions + horizon

    if horizon_ms < 1:
        raise ValueError("horizon_ms must be >= 1")

    ts = df["ts_ms"].to_numpy()

    return pd.Series(
        ts.searchsorted(ts + horizon_ms, side="left"),
        index=df.index,
    )


def label_tie_fraction(
    df: pd.DataFrame,
    horizon: int = 50,
    horizon_ms: int | None = None,
) -> float:
    """Fraction of labelable rows whose mid-price did not change."""

    mid = mid_price(df)
    future_pos = _future_positions(df, horizon, horizon_ms)
    valid = future_pos < len(df)

    if not valid.any():
        return 0.0

    future_mid = mid.to_numpy()[future_pos[valid].to_numpy()]
    current_mid = mid[valid].to_numpy()

    return float((future_mid == current_mid).mean())


def build_dataset(
    df: pd.DataFrame,
    horizon: int,
    feature_fn=build_features,
    return_meta: bool = False,
    ties: str = "drop",
    horizon_ms: int | None = None,
):
    """Predict whether mid-price rises over the next horizon.

    The horizon is counted in rows by default, or in milliseconds when
    horizon_ms is given (order-book events arrive irregularly).

    ties="drop" removes rows where the mid-price did not move, so the
    target means "up" versus "down". ties="as_down" keeps the old
    behaviour, where no move is labelled 0.
    """

    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    if ties not in ("drop", "as_down"):
        raise ValueError("ties must be 'drop' or 'as_down'")

    features = feature_fn(df)
    mid = mid_price(df)

    future_pos = _future_positions(df, horizon, horizon_ms)
    has_future = future_pos < len(df)

    safe_pos = future_pos.where(has_future, 0).to_numpy()
    future_mid = pd.Series(
        mid.to_numpy()[safe_pos],
        index=df.index,
    ).where(has_future)

    hold_rows = (
        future_pos - pd.Series(range(len(df)), index=df.index)
    ).where(has_future)

    y = (future_mid > mid).astype("int8")

    keep = features.notna().all(axis=1) & future_mid.notna()

    if ties == "drop":
        keep &= future_mid != mid

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
        "hold_rows": hold_rows.loc[keep].reset_index(drop=True),
    })

    return X, y, meta
