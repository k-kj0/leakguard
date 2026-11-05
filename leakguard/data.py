"""Load and validate normalized order-book CSV data."""

from pathlib import Path
import pandas as pd


REQUIRED = [
    "ts_ms",
    "bid_px_1",
    "ask_px_1",
    "bid_sz_1",
    "ask_sz_1",
]


OPTIONAL_LEVELS = [
    f"{side}_{kind}_{level}"
    for level in (2, 3)
    for side, kind in (
        ("bid", "px"),
        ("ask", "px"),
        ("bid", "sz"),
        ("ask", "sz"),
    )
]


def load_order_book_csv(
    path: str | Path,
) -> pd.DataFrame:
    """Read and validate a normalized order-book CSV."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path}"
        )

    df = pd.read_csv(path)

    missing = sorted(
        set(REQUIRED) - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required order-book columns: {missing}"
        )

    numeric = list(dict.fromkeys(
        REQUIRED
        + [
            c for c in OPTIONAL_LEVELS
            if c in df
        ]
    ))

    for col in numeric:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    if df[numeric].isna().any().any():

        bad = df[numeric].columns[
            df[numeric].isna().any()
        ].tolist()

        raise ValueError(
            f"Missing or non-numeric values in columns: {bad}"
        )

    if not df["ts_ms"].is_monotonic_increasing:
        raise ValueError(
            "ts_ms must be sorted in ascending order"
        )

    if df["ts_ms"].duplicated().any():
        raise ValueError(
            "ts_ms contains duplicate timestamps"
        )

    if (df["bid_px_1"] >= df["ask_px_1"]).any():
        raise ValueError(
            "Invalid book: best bid must be below best ask"
        )

    size_cols = [
        c for c in df
        if c.startswith(("bid_sz_", "ask_sz_"))
    ]

    if (df[size_cols] < 0).any().any():
        raise ValueError(
            "Order-book sizes cannot be negative"
        )

    if len(df) < 500:
        raise ValueError(
            "At least 500 rows are required for a useful experiment"
        )

    return df
