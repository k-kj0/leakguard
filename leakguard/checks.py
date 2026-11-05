"""Data, causality and evaluation sanity checks."""

import numpy as np
import pandas as pd


def check_data_integrity(df: pd.DataFrame) -> list[str]:

    issues = []

    if "ts_ms" not in df:
        return ["missing ts_ms column"]

    if not df["ts_ms"].is_monotonic_increasing:
        issues.append("timestamps are not sorted")

    if df["ts_ms"].duplicated().any():
        issues.append("duplicate timestamps")

    if df.isna().any().any():
        issues.append("NaN values present")

    if (df["bid_px_1"] >= df["ask_px_1"]).any():
        issues.append("crossed book (bid >= ask)")

    for c in [
        x for x in df
        if x.startswith(("bid_sz_", "ask_sz_"))
    ]:
        if (df[c] < 0).any():
            issues.append(f"negative size in {c}")

    return issues


def check_causality(
    feature_fn,
    df: pd.DataFrame,
    n_probes: int = 15,
    seed: int = 0,
    min_row: int = 200,
) -> list[str]:
    """Truncate the future at sampled rows; causal feature values must not change."""

    df = df.reset_index(drop=True)

    full = feature_fn(df)

    if len(df) <= min_row + 1:
        raise ValueError("Not enough rows for causality probes")

    rng = np.random.default_rng(seed)

    probes = rng.choice(
        np.arange(min_row, len(df) - 1),
        size=min(n_probes, len(df) - min_row - 1),
        replace=False,
    )

    bad = set()

    for k in probes:

        truncated = feature_fn(
            df.iloc[:k + 1].reset_index(drop=True)
        ).iloc[-1]

        reference = full.iloc[k]

        for col in full.columns:

            if not np.isclose(
                truncated[col],
                reference[col],
                rtol=1e-9,
                atol=1e-12,
                equal_nan=True,
            ):
                bad.add(col)

    return sorted(bad)
