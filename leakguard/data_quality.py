
"""Independent quality audit for normalized order-book datasets."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "ts_ms",
    "bid_px_1",
    "ask_px_1",
    "bid_sz_1",
    "ask_sz_1",
]


def audit_dataframe(
    df,
    expected_interval_ms=None,
    gap_multiplier=1.5,
):
    """Audit normalized market data and return a JSON-safe report."""

    if gap_multiplier <= 1:
        raise ValueError("gap_multiplier must be greater than 1.")

    if expected_interval_ms is not None and expected_interval_ms <= 0:
        raise ValueError("expected_interval_ms must be positive.")

    issues = []

    def add_issue(code, severity, message, count=None):
        item = {
            "code": code,
            "severity": severity,
            "message": message,
        }
        if count is not None:
            item["count"] = int(count)
        issues.append(item)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        add_issue(
            "missing_required_columns",
            "error",
            f"Missing required columns: {missing_columns}",
        )

        return {
            "status": "fail",
            "rows": int(len(df)),
            "columns": list(df.columns),
            "issues": issues,
            "metrics": {},
            "checks": {},
        }

    if df.empty:
        add_issue(
            "empty_dataset",
            "error",
            "Dataset contains no observations.",
        )

        return {
            "status": "fail",
            "rows": 0,
            "columns": list(df.columns),
            "issues": issues,
            "metrics": {},
            "checks": {},
        }

    data = df.copy()

    numeric_columns = [
        column for column in data.columns
        if column == "ts_ms"
        or column.endswith(("_px_1", "_px_2", "_px_3"))
        or column.endswith(("_sz_1", "_sz_2", "_sz_3"))
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    missing_counts = data[numeric_columns].isna().sum()
    missing_total = int(missing_counts.sum())

    if missing_total:
        add_issue(
            "missing_or_non_numeric_values",
            "error",
            "Numeric market-data fields contain missing or non-numeric values.",
            missing_total,
        )

    finite_mask = np.isfinite(
        data[numeric_columns].to_numpy(dtype=float)
    )

    nonfinite_count = int((~finite_mask).sum())

    if nonfinite_count:
        add_issue(
            "non_finite_values",
            "error",
            "Numeric market-data fields contain infinite values.",
            nonfinite_count,
        )

    # Do not perform numeric comparisons on invalid rows.
    valid = data.dropna(subset=REQUIRED_COLUMNS).copy()

    for column in REQUIRED_COLUMNS:
        valid[column] = pd.to_numeric(
            valid[column],
            errors="coerce",
        )

    valid = valid.loc[
        np.isfinite(valid[REQUIRED_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    ]

    timestamp = valid["ts_ms"]

    duplicate_count = int(timestamp.duplicated().sum())

    if duplicate_count:
        add_issue(
            "duplicate_timestamps",
            "error",
            "Timestamps are not unique.",
            duplicate_count,
        )

    chronological = bool(timestamp.is_monotonic_increasing)

    if not chronological:
        add_issue(
            "out_of_order_timestamps",
            "error",
            "Timestamps are not in ascending order.",
        )

    invalid_bid = int((valid["bid_px_1"] <= 0).sum())
    invalid_ask = int((valid["ask_px_1"] <= 0).sum())

    if invalid_bid or invalid_ask:
        add_issue(
            "non_positive_prices",
            "error",
            "Best bid and ask prices must be positive.",
            invalid_bid + invalid_ask,
        )

    crossed = int(
        (valid["bid_px_1"] >= valid["ask_px_1"]).sum()
    )

    if crossed:
        add_issue(
            "crossed_or_locked_books",
            "error",
            "Best bid must be strictly below best ask.",
            crossed,
        )

    size_columns = [
        column for column in data.columns
        if column.startswith(("bid_sz_", "ask_sz_"))
    ]

    negative_sizes = 0
    non_positive_sizes = 0

    for column in size_columns:
        values = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        negative_sizes += int((values < 0).sum())
        non_positive_sizes += int((values <= 0).sum())

    if negative_sizes:
        add_issue(
            "negative_sizes",
            "error",
            "Order-book sizes cannot be negative.",
            negative_sizes,
        )

    if non_positive_sizes:
        add_issue(
            "zero_or_negative_sizes",
            "warning",
            "One or more recorded order-book levels have zero size.",
            non_positive_sizes,
        )

    depth_order_issues = 0

    for level in (2, 3):
        bid_column = f"bid_px_{level}"
        ask_column = f"ask_px_{level}"

        if bid_column in data.columns:
            previous = f"bid_px_{level - 1}"

            if previous in data.columns:
                bid_values = pd.to_numeric(
                    data[bid_column],
                    errors="coerce",
                )
                previous_values = pd.to_numeric(
                    data[previous],
                    errors="coerce",
                )

                depth_order_issues += int(
                    (
                        bid_values.notna()
                        & previous_values.notna()
                        & (previous_values <= bid_values)
                    ).sum()
                )

        if ask_column in data.columns:
            previous = f"ask_px_{level - 1}"

            if previous in data.columns:
                ask_values = pd.to_numeric(
                    data[ask_column],
                    errors="coerce",
                )
                previous_values = pd.to_numeric(
                    data[previous],
                    errors="coerce",
                )

                depth_order_issues += int(
                    (
                        ask_values.notna()
                        & previous_values.notna()
                        & (previous_values >= ask_values)
                    ).sum()
                )

    if depth_order_issues:
        add_issue(
            "invalid_depth_order",
            "error",
            "Bid levels must descend and ask levels must ascend.",
            depth_order_issues,
        )

    # Time-gap analysis.
    intervals = timestamp.diff().dropna()
    positive_intervals = intervals[intervals > 0]

    median_interval = (
        float(positive_intervals.median())
        if len(positive_intervals) else None
    )

    if expected_interval_ms is not None:
        reference_interval = float(expected_interval_ms)
    else:
        reference_interval = median_interval

    gap_threshold = (
        reference_interval * gap_multiplier
        if reference_interval is not None else None
    )

    gap_count = 0
    maximum_gap = None
    p95_gap = None

    if len(positive_intervals):
        maximum_gap = float(positive_intervals.max())
        p95_gap = float(positive_intervals.quantile(0.95))

        if gap_threshold is not None:
            gap_count = int(
                (positive_intervals > gap_threshold).sum()
            )

            if gap_count:
                add_issue(
                    "timestamp_gaps",
                    "warning",
                    (
                        "Observation intervals exceed the configured "
                        "gap threshold."
                    ),
                    gap_count,
                )

    # Spread anomaly check: a robust warning, not an automatic rejection.
    spread = (
        valid["ask_px_1"] - valid["bid_px_1"]
    )

    spread_median = (
        float(spread.median())
        if len(spread) else None
    )

    spread_mad = (
        float((spread - spread.median()).abs().median())
        if len(spread) else None
    )

    spread_outliers = 0

    if spread_mad is not None and spread_mad > 0:
        robust_scale = 1.4826 * spread_mad
        spread_outliers = int(
            (
                spread
                > spread_median + 10 * robust_scale
            ).sum()
        )

        if spread_outliers:
            add_issue(
                "unusual_spread_observations",
                "warning",
                "Some spreads are unusually large relative to the median.",
                spread_outliers,
            )

    metrics = {
        "rows": int(len(data)),
        "valid_required_rows": int(len(valid)),
        "columns": int(len(data.columns)),
        "missing_numeric_values": missing_total,
        "non_finite_values": nonfinite_count,
        "duplicate_timestamps": duplicate_count,
        "chronological": chronological,
        "median_interval_ms": median_interval,
        "expected_interval_ms": expected_interval_ms,
        "gap_threshold_ms": gap_threshold,
        "timestamp_gap_count": gap_count,
        "maximum_interval_ms": maximum_gap,
        "p95_interval_ms": p95_gap,
        "crossed_or_locked_books": crossed,
        "negative_sizes": negative_sizes,
        "zero_or_negative_sizes": non_positive_sizes,
        "invalid_depth_order": depth_order_issues,
        "median_spread": spread_median,
        "spread_outliers": spread_outliers,
        "time_span_ms": (
            float(timestamp.max() - timestamp.min())
            if len(timestamp) else None
        ),
    }

    errors = sum(
        issue["severity"] == "error"
        for issue in issues
    )

    warnings = sum(
        issue["severity"] == "warning"
        for issue in issues
    )

    status = (
        "fail" if errors
        else "warn" if warnings
        else "pass"
    )

    return {
        "status": status,
        "rows": int(len(data)),
        "columns": list(data.columns),
        "issues": issues,
        "metrics": metrics,
        "checks": {
            "error_count": errors,
            "warning_count": warnings,
            "passed": status == "pass",
        },
    }


def render_report(report):
    """Render the audit result as Markdown."""

    metrics = report["metrics"]

    lines = [
        "# Leakguard Data Quality Audit",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Rows: **{report['rows']}**",
        f"- Errors: **{report['checks']['error_count']}**",
        f"- Warnings: **{report['checks']['warning_count']}**",
        "",
        "## Dataset metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Findings",
        "",
    ]

    if not report["issues"]:
        lines.append("No configured audit findings.")
    else:
        for issue in report["issues"]:
            lines.append(
                f"- **{issue['severity'].upper()} — "
                f"{issue['code']}:** {issue['message']}"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "- A PASS means only that the configured checks found no issues.",
        "- A WARN means the dataset has observations requiring review.",
        "- A FAIL means at least one hard integrity check failed.",
        "- A PASS does not establish that the feed is complete or representative.",
        "- This audit does not establish predictive performance or profitability.",
        "",
    ]

    return "\n".join(lines)


def run_audit(
    input_path,
    out_dir="results/data_quality",
    expected_interval_ms=100,
):
    """Load a normalized CSV and save JSON and Markdown audit reports."""

    source = Path(input_path)

    if not source.is_file():
        raise FileNotFoundError(
            f"Input dataset not found: {source}"
        )

    df = pd.read_csv(source)

    report = audit_dataframe(
        df,
        expected_interval_ms=expected_interval_ms,
    )

    report["input_file"] = str(source)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "latest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    markdown = render_report(report)

    (output / "latest.md").write_text(
        markdown,
        encoding="utf-8",
    )

    print(markdown)

    if report["status"] == "fail":
        raise RuntimeError(
            "Data-quality audit failed. Inspect latest.json."
        )

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        default="results/coinbase_normalized.csv",
    )

    parser.add_argument(
        "--out",
        default="results/data_quality",
    )

    parser.add_argument(
        "--expected-interval-ms",
        type=float,
        default=100,
    )

    args = parser.parse_args()

    run_audit(
        args.input,
        out_dir=args.out,
        expected_interval_ms=args.expected_interval_ms,
    )


if __name__ == "__main__":
    main()
