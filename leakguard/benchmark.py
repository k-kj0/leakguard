
"""Reproducible runtime and memory benchmarks for Leakguard."""

import argparse
import gc
import json
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .features import build_features
from .simulate import simulate_order_book


DEFAULT_ROW_COUNTS = (10_000, 30_000, 60_000)


def _validate_row_counts(row_counts):
    if not row_counts:
        raise ValueError("At least one row count is required.")

    normalized = []

    for value in row_counts:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Every row count must be an integer.")

        if value < 100:
            raise ValueError("Every row count must be at least 100.")

        normalized.append(value)

    if len(set(normalized)) != len(normalized):
        raise ValueError("Row counts must be unique.")

    return sorted(normalized)


def _measure(operation):
    """Measure wall time and Python-tracked peak memory for one operation."""
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()

    try:
        result = operation()
        elapsed = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return result, {
        "elapsed_seconds": float(elapsed),
        "python_peak_traced_bytes": int(peak_bytes),
        "python_peak_traced_mib": float(
            peak_bytes / (1024 * 1024)
        ),
    }


def _dataframe_memory_bytes(frame):
    return int(
        frame.memory_usage(index=True, deep=True).sum()
    )


def run_benchmarks(
    row_counts=DEFAULT_ROW_COUNTS,
    seed=7,
    out_dir="results/benchmarks",
):
    """Benchmark synthetic data generation and feature construction."""
    row_counts = _validate_row_counts(list(row_counts))

    results = []

    for n_rows in row_counts:
        book, generation_measurement = _measure(
            lambda: simulate_order_book(
                n=n_rows,
                seed=seed,
            )
        )

        features, feature_measurement = _measure(
            lambda: build_features(book)
        )

        if len(book) != n_rows:
            raise RuntimeError(
                "Synthetic generator returned an unexpected row count."
            )

        if len(features) != n_rows:
            raise RuntimeError(
                "Feature builder returned an unexpected row count."
            )

        results.append({
            "rows": int(n_rows),
            "input_columns": int(len(book.columns)),
            "feature_columns": int(len(features.columns)),
            "generation": generation_measurement,
            "feature_construction": feature_measurement,
            "input_dataframe_bytes": _dataframe_memory_bytes(book),
            "feature_dataframe_bytes": _dataframe_memory_bytes(
                features
            ),
            "input_dataframe_mib": (
                _dataframe_memory_bytes(book) / (1024 * 1024)
            ),
            "feature_dataframe_mib": (
                _dataframe_memory_bytes(features) / (1024 * 1024)
            ),
        })

    report = {
        "benchmark": "leakguard_runtime_memory",
        "version": 1,
        "seed": int(seed),
        "row_counts": row_counts,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "results": results,
        "measurement_notes": [
            (
                "Wall time is measured with time.perf_counter "
                "in the current execution environment."
            ),
            (
                "Python peak traced memory is measured with "
                "tracemalloc and is not total process RSS."
            ),
            (
                "DataFrame memory reports the measured input and "
                "feature DataFrame payloads."
            ),
            (
                "The benchmark measures synthetic generation and "
                "feature construction, not model training, live "
                "market-data ingestion, network latency, or execution."
            ),
            (
                "Results depend on hardware, software versions, "
                "system load, and runtime environment."
            ),
        ],
    }

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    json_path = output / "latest.json"
    markdown_path = output / "latest.md"

    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    markdown_path.write_text(
        render_report(report),
        encoding="utf-8",
    )

    print(render_report(report))
    return report


def render_report(report):
    """Render benchmark results as Markdown."""
    lines = [
        "# Leakguard Runtime and Memory Benchmark",
        "",
        f"- Random seed: **{report['seed']}**",
        f"- Python: **{report['environment']['python']}**",
        f"- NumPy: **{report['environment']['numpy']}**",
        f"- pandas: **{report['environment']['pandas']}**",
        (
            "- scikit-learn: "
            f"**{report['environment']['scikit_learn']}**"
        ),
        "",
        "## Results",
        "",
        (
            "| Rows | Generation (s) | Feature construction (s) "
            "| Generation peak traced memory (MiB) "
            "| Feature peak traced memory (MiB) "
            "| Input DataFrame (MiB) | Feature DataFrame (MiB) |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for item in report["results"]:
        lines.append(
            f"| {item['rows']} "
            f"| {item['generation']['elapsed_seconds']:.4f} "
            f"| {item['feature_construction']['elapsed_seconds']:.4f} "
            f"| {item['generation']['python_peak_traced_mib']:.2f} "
            f"| {item['feature_construction']['python_peak_traced_mib']:.2f} "
            f"| {item['input_dataframe_mib']:.2f} "
            f"| {item['feature_dataframe_mib']:.2f} |"
        )

    lines += [
        "",
        "## Measurement methodology",
        "",
        "- Each row count is measured independently.",
        "- Synthetic data generation and feature construction are timed separately.",
        "- Wall time uses Python's `time.perf_counter`.",
        "- Peak traced memory uses Python's `tracemalloc`.",
        "- DataFrame memory is measured using pandas `memory_usage(deep=True)`.",
        "",
        "## Interpretation and limitations",
        "",
        "- Results describe this execution environment only.",
        "- Runtime can vary with hardware, background load, and software versions.",
        "- `tracemalloc` is not a measurement of total process resident memory.",
        "- DataFrame payload size does not include every temporary allocation.",
        "- This does not benchmark model fitting or prediction.",
        "- This does not measure exchange feed throughput or network latency.",
        "- This does not establish production HFT capacity.",
        "- Larger row counts can require substantially more memory.",
        "",
        "These measurements are engineering diagnostics, not a performance "
        "claim about a production trading system.",
        "",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=list(DEFAULT_ROW_COUNTS),
        help="One or more row counts to benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--out",
        default="results/benchmarks",
    )

    args = parser.parse_args()

    run_benchmarks(
        row_counts=args.rows,
        seed=args.seed,
        out_dir=args.out,
    )
