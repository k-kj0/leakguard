
"""Normalize Coinbase Level 2 captures onto a fixed-time grid."""

import argparse
import hashlib
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

LEVEL_COLUMNS = {
    level: [
        f"{side}_{kind}_{level}"
        for side, kind in (
            ("bid", "px"),
            ("ask", "px"),
            ("bid", "sz"),
            ("ask", "sz"),
        )
    ]
    for level in (2, 3)
}


def _validate_book(raw, book_columns):
    """Reject invalid order-book states before sampling."""

    numeric = raw[book_columns].copy()

    for column in book_columns:
        numeric[column] = pd.to_numeric(
            numeric[column],
            errors="coerce",
        )

    if numeric.isna().any().any():
        bad = numeric.columns[
            numeric.isna().any()
        ].tolist()

        raise ValueError(
            f"Missing or non-numeric book values: {bad}"
        )

    values = numeric.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Order book contains infinite or invalid values."
        )

    price_columns = [
        column for column in book_columns
        if "_px_" in column
    ]

    size_columns = [
        column for column in book_columns
        if "_sz_" in column
    ]

    if (numeric[price_columns] <= 0).any().any():
        raise ValueError("Order-book prices must be positive.")

    if (numeric[size_columns] <= 0).any().any():
        raise ValueError(
            "Recorded order-book level sizes must be positive."
        )

    if (
        numeric["bid_px_1"] >= numeric["ask_px_1"]
    ).any():
        raise ValueError(
            "Invalid order book: best bid must be below best ask."
        )

    for level in (2, 3):
        if f"bid_px_{level}" not in numeric:
            continue

        previous = level - 1

        if (
            numeric[f"bid_px_{previous}"]
            <= numeric[f"bid_px_{level}"]
        ).any():
            raise ValueError(
                f"Bid prices are not strictly descending at level {level}."
            )

        if (
            numeric[f"ask_px_{previous}"]
            >= numeric[f"ask_px_{level}"]
        ).any():
            raise ValueError(
                f"Ask prices are not strictly ascending at level {level}."
            )

    return numeric


def normalize_capture(
    input_path,
    output_path,
    sample_ms=100,
    max_staleness_ms=500,
):
    """
    Convert event-driven Coinbase book states into fixed-time observations.

    Each output row uses the latest captured state at or before its
    timestamp. Future observations are never used to fill earlier rows.
    """

    source = Path(input_path)
    destination = Path(output_path)

    if not source.is_file():
        raise FileNotFoundError(
            f"Capture file not found: {source}"
        )

    if sample_ms < 1:
        raise ValueError("sample_ms must be at least 1.")

    if max_staleness_ms < 1:
        raise ValueError(
            "max_staleness_ms must be at least 1."
        )

    raw = pd.read_csv(source)

    if raw.empty:
        raise ValueError("Capture file contains no records.")

    # A research dataset must contain exactly one product.
    if "product_id" in raw.columns:
        products = raw["product_id"].dropna().unique()

        if len(products) != 1:
            raise ValueError(
                "Capture must contain exactly one non-empty product."
            )

    # Local receipt time establishes when Leakguard actually knew
    # each reconstructed state. Exchange timestamps are retained in
    # the original capture and are not used as the sampling clock.
    if "captured_at_utc" not in raw.columns:
        raise ValueError(
            "Capture must contain captured_at_utc timestamps."
        )

    capture_times = pd.to_datetime(
        raw["captured_at_utc"],
        utc=True,
        errors="coerce",
    )

    if capture_times.isna().any():
        raise ValueError(
            "Capture contains missing or invalid receipt timestamps."
        )

    if not capture_times.is_monotonic_increasing:
        raise ValueError(
            "Capture receipt timestamps are out of order."
        )

    # The collector's arrival counter is an independent ordering check.
    if "message_index" in raw.columns:
        message_index = pd.to_numeric(
            raw["message_index"],
            errors="coerce",
        )

        if message_index.isna().any():
            raise ValueError("Invalid message_index values.")

        if not message_index.is_monotonic_increasing:
            raise ValueError(
                "message_index must be in ascending order."
            )

        if message_index.duplicated().any():
            raise ValueError("Duplicate message_index values.")

    missing_base = [
        column for column in BASE_COLUMNS
        if column not in raw.columns
    ]

    if missing_base:
        raise ValueError(
            f"Capture is missing required columns: {missing_base}"
        )

    book_columns = list(BASE_COLUMNS)

    # Optional depth levels must be complete, not partially present.
    for level, columns in LEVEL_COLUMNS.items():
        present = [
            column for column in columns
            if column in raw.columns
        ]

        if present and len(present) != len(columns):
            raise ValueError(
                f"Depth level {level} is incomplete: {present}"
            )

        if len(present) == len(columns):
            book_columns.extend(columns)

    book = _validate_book(raw, book_columns)

    input_rows = len(raw)

    raw_duplicate_capture_times = int(
        capture_times.duplicated().sum()
    )

    duplicate_exchange_times = 0

    if "exchange_time" in raw.columns:
        exchange_times = pd.to_datetime(
            raw["exchange_time"],
            utc=True,
            errors="coerce",
        )

        valid_exchange_times = exchange_times.dropna()

        duplicate_exchange_times = int(
            valid_exchange_times.duplicated().sum()
        )

    # Retain every source record. Duplicate timestamps are not silently
    # deleted. At a sampling instant, merge_asof selects the last state
    # received at or before that instant.
    source_states = book.copy()
    source_states["_source_time"] = capture_times
    source_states["_source_order"] = np.arange(input_rows)

    source_states = source_states.sort_values(
        ["_source_time", "_source_order"],
        kind="stable",
    )

    first_time = capture_times.iloc[0]
    last_time = capture_times.iloc[-1]

    frequency = f"{sample_ms}ms"

    grid_start = first_time.ceil(frequency)
    grid_end = last_time.floor(frequency)

    if grid_start > grid_end:
        raise ValueError(
            "Capture is too short to produce a sampling interval."
        )

    grid = pd.DataFrame({
        "sample_time": pd.date_range(
            start=grid_start,
            end=grid_end,
            freq=frequency,
        )
    })

    sampled = pd.merge_asof(
        grid.sort_values("sample_time"),
        source_states,
        left_on="sample_time",
        right_on="_source_time",
        direction="backward",
        tolerance=pd.Timedelta(
            milliseconds=max_staleness_ms
        ),
    )

    if sampled["_source_time"].isna().any():
        missing_count = int(
            sampled["_source_time"].isna().sum()
        )

        raise ValueError(
            f"{missing_count} sampling points have no sufficiently "
            "recent book state. The capture contains stale-data gaps."
        )

    state_age_ms = (
        sampled["sample_time"] - sampled["_source_time"]
    ).dt.total_seconds() * 1000

    if (state_age_ms < 0).any():
        raise ValueError(
            "Internal error: a future state entered an earlier sample."
        )

    if (state_age_ms > max_staleness_ms).any():
        raise ValueError(
            "One or more sampled book states exceed the staleness limit."
        )

    normalized = pd.DataFrame({
        "ts_ms": (
            sampled["sample_time"].astype("int64") / 1_000_000
        )
    })

    for column in book_columns:
        normalized[column] = sampled[column].to_numpy()

    if normalized["ts_ms"].duplicated().any():
        raise ValueError(
            "Normalized timestamps are not unique."
        )

    if not normalized["ts_ms"].is_monotonic_increasing:
        raise ValueError(
            "Normalized timestamps are not chronological."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized.to_csv(
        destination,
        index=False,
    )

    event_intervals = (
        capture_times.diff().dt.total_seconds() * 1000
    ).dropna()

    source_state_reuse = (
        len(sampled)
        - sampled["_source_order"].nunique()
    )

    metadata = {
        "input_file": str(source),
        "output_file": str(destination),
        "input_sha256": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
        "timestamp_source": "captured_at_utc",
        "exchange_time_used_for_sampling": False,
        "sampling_method": (
            "Backward as-of sampling: latest captured state "
            "at or before each grid timestamp."
        ),
        "timestamp_unit": "milliseconds",
        "sample_interval_ms": sample_ms,
        "maximum_allowed_state_age_ms": max_staleness_ms,
        "input_rows": input_rows,
        "output_rows": len(normalized),
        "raw_duplicate_capture_timestamps": (
            raw_duplicate_capture_times
        ),
        "raw_duplicate_exchange_timestamps": (
            duplicate_exchange_times
        ),
        "source_states_reused_for_multiple_samples": (
            int(source_state_reuse)
        ),
        "source_state_reuse_fraction": (
            float(source_state_reuse / len(sampled))
        ),
        "maximum_observed_state_age_ms": float(
            state_age_ms.max()
        ),
        "median_source_event_interval_ms": (
            float(event_intervals.median())
            if len(event_intervals) else None
        ),
        "p95_source_event_interval_ms": (
            float(event_intervals.quantile(0.95))
            if len(event_intervals) else None
        ),
        "maximum_source_event_interval_ms": (
            float(event_intervals.max())
            if len(event_intervals) else None
        ),
        "columns_written": list(normalized.columns),
        "minimum_rows_for_experiment": 500,
        "experiment_ready_by_row_count": (
            len(normalized) >= 500
        ),
        "limitations": [
            "Sampling uses local receipt time, not exchange event time.",
            "Backward as-of sampling carries the last known book state forward.",
            "This does not recover messages missed before the collector received them.",
            "A sufficient row count does not establish representative market coverage.",
        ],
    }

    metadata_path = destination.with_suffix(".json")

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
        default="results/coinbase_ws_capture.csv",
    )

    parser.add_argument(
        "--out",
        default="results/coinbase_normalized.csv",
    )

    parser.add_argument(
        "--sample-ms",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max-staleness-ms",
        type=int,
        default=500,
    )

    args = parser.parse_args()

    normalize_capture(
        args.input,
        args.out,
        sample_ms=args.sample_ms,
        max_staleness_ms=args.max_staleness_ms,
    )


if __name__ == "__main__":
    main()
