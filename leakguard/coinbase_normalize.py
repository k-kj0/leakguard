
"""Normalize Coinbase Level 2 capture CSVs for Leakguard experiments."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_COLUMNS = [
    "bid_px_1",
    "ask_px_1",
    "bid_sz_1",
    "ask_sz_1",
]

LEVEL_2_3_COLUMNS = [
    f"{side}_{kind}_{level}"
    for level in (2, 3)
    for side, kind in (
        ("bid", "px"),
        ("ask", "px"),
        ("bid", "sz"),
        ("ask", "sz"),
    )
]


def normalize_capture(input_path, output_path):
    """Convert a Coinbase capture into Leakguard's normalized CSV."""

    source = Path(input_path)
    destination = Path(output_path)

    if not source.is_file():
        raise FileNotFoundError(f"Capture file not found: {source}")

    raw = pd.read_csv(source)

    if raw.empty:
        raise ValueError("Capture file contains no records.")

    # Do not combine different products in one experiment.
    if "product_id" in raw.columns:
        products = raw["product_id"].dropna().unique()

        if len(products) > 1:
            raise ValueError(
                "Capture contains multiple products. "
                "Normalize one product at a time."
            )

    # Use one consistent timestamp source for the entire capture.
    # Exchange time is preferred. If even one exchange timestamp is
    # missing, use local capture/receipt time for every row instead.
    if (
        "exchange_time" in raw.columns
        and raw["exchange_time"].notna().all()
        and raw["exchange_time"].astype(str).str.strip().ne("").all()
    ):
        timestamp_column = "exchange_time"
        timestamp_source = "exchange_time"
    elif "captured_at_utc" in raw.columns:
        timestamp_column = "captured_at_utc"
        timestamp_source = "captured_at_utc"
    else:
        raise ValueError(
            "No complete timestamp source. Expected exchange_time "
            "or captured_at_utc."
        )

    timestamps = pd.to_datetime(
        raw[timestamp_column],
        utc=True,
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError(
            f"Invalid timestamps found in {timestamp_column}."
        )

    # Preserve sub-millisecond precision. The field is still expressed
    # in milliseconds, but may contain fractional milliseconds.
    raw["ts_ms"] = timestamps.astype("int64") / 1_000_000

    # Multiple events can share an exchange timestamp. Keep the last
    # recorded book state at that timestamp rather than inventing time.
    input_rows = len(raw)
    duplicate_mask = raw["ts_ms"].duplicated(keep="last")
    duplicate_rows_removed = int(duplicate_mask.sum())
    raw = raw.loc[~duplicate_mask].copy()

    # Do not silently reorder out-of-order market data.
    if not raw["ts_ms"].is_monotonic_increasing:
        raise ValueError(
            "Timestamps are out of order. Capture must be chronological."
        )

    missing_base = [
        column for column in BASE_COLUMNS
        if column not in raw.columns
    ]

    if missing_base:
        raise ValueError(
            f"Capture is missing required columns: {missing_base}"
        )

    # Include levels 2 and 3 only when all four fields for that level
    # are available.
    selected_columns = ["ts_ms"] + BASE_COLUMNS

    for level in (2, 3):
        level_columns = [
            f"{side}_{kind}_{level}"
            for side, kind in (
                ("bid", "px"),
                ("ask", "px"),
                ("bid", "sz"),
                ("ask", "sz"),
            )
        ]

        if all(column in raw.columns for column in level_columns):
            selected_columns.extend(level_columns)

    normalized = raw[selected_columns].copy()

    numeric_columns = selected_columns

    for column in numeric_columns:
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        )

    if normalized.isna().any().any():
        bad_columns = normalized.columns[
            normalized.isna().any()
        ].tolist()

        raise ValueError(
            f"Missing or non-numeric values in: {bad_columns}"
        )

    values = normalized[numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Capture contains infinite or invalid values.")

    if (normalized["bid_px_1"] <= 0).any():
        raise ValueError("Best bid prices must be positive.")

    if (normalized["ask_px_1"] <= 0).any():
        raise ValueError("Best ask prices must be positive.")

    if (
        normalized["bid_px_1"] >= normalized["ask_px_1"]
    ).any():
        raise ValueError(
            "Invalid order book: best bid must be below best ask."
        )

    size_columns = [
        column for column in normalized.columns
        if column.startswith(("bid_sz_", "ask_sz_"))
    ]

    if (normalized[size_columns] < 0).any().any():
        raise ValueError("Order-book sizes cannot be negative.")

    if (normalized[["bid_sz_1", "ask_sz_1"]] <= 0).any().any():
        raise ValueError("Best bid and ask sizes must be positive.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(destination, index=False)

    output_rows = len(normalized)

    metadata = {
        "input_file": str(source),
        "output_file": str(destination),
        "timestamp_source": timestamp_source,
        "timestamp_unit": "milliseconds",
        "timestamp_precision": "fractional milliseconds preserved",
        "input_rows": input_rows,
        "output_rows": output_rows,
        "duplicate_timestamp_rows_removed": duplicate_rows_removed,
        "columns_written": list(normalized.columns),
        "minimum_rows_for_experiment": 500,
        "experiment_ready_by_row_count": output_rows >= 500,
        "warning": (
            "A successful normalization does not establish that "
            "the capture is representative historical market data."
        ),
    }

    metadata_path = destination.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))

    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        default="results/coinbase_ws_capture.csv",
    )

    parser.add_argument(
        "--out",
        default="results/coinbase_normalized.csv",
    )

    args = parser.parse_args()

    normalize_capture(args.input, args.out)


if __name__ == "__main__":
    main()
