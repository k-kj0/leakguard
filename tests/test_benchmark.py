
import json

import pytest

from leakguard.benchmark import (
    _validate_row_counts,
    render_report,
    run_benchmarks,
)


def test_benchmark_generates_json_and_markdown(tmp_path):
    report = run_benchmarks(
        row_counts=[500, 1_000],
        seed=7,
        out_dir=tmp_path,
    )

    assert len(report["results"]) == 2
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "latest.md").exists()

    saved = json.loads(
        (tmp_path / "latest.json").read_text(
            encoding="utf-8"
        )
    )

    assert saved["benchmark"] == "leakguard_runtime_memory"
    assert saved["row_counts"] == [500, 1_000]


def test_benchmark_records_expected_row_counts(tmp_path):
    report = run_benchmarks(
        row_counts=[500],
        out_dir=tmp_path,
    )

    result = report["results"][0]

    assert result["rows"] == 500
    assert result["input_columns"] > 0
    assert result["feature_columns"] > 0


def test_benchmark_records_nonnegative_runtime(tmp_path):
    report = run_benchmarks(
        row_counts=[500],
        out_dir=tmp_path,
    )

    result = report["results"][0]

    assert result["generation"]["elapsed_seconds"] >= 0
    assert (
        result["feature_construction"]["elapsed_seconds"] >= 0
    )


def test_benchmark_records_memory_measurements(tmp_path):
    report = run_benchmarks(
        row_counts=[500],
        out_dir=tmp_path,
    )

    result = report["results"][0]

    assert (
        result["generation"]["python_peak_traced_bytes"] >= 0
    )
    assert (
        result["feature_construction"]["python_peak_traced_bytes"]
        >= 0
    )
    assert result["input_dataframe_bytes"] > 0
    assert result["feature_dataframe_bytes"] > 0


def test_benchmark_is_reproducible_in_row_counts(tmp_path):
    report = run_benchmarks(
        row_counts=[1_000, 500],
        seed=11,
        out_dir=tmp_path,
    )

    assert report["row_counts"] == [500, 1_000]
    assert [item["rows"] for item in report["results"]] == [
        500,
        1_000,
    ]


def test_markdown_report_contains_limitations(tmp_path):
    report = run_benchmarks(
        row_counts=[500],
        out_dir=tmp_path,
    )

    markdown = render_report(report)

    assert "Runtime and Memory Benchmark" in markdown
    assert "tracemalloc" in markdown
    assert "production HFT capacity" in markdown


def test_rejects_empty_row_counts():
    with pytest.raises(ValueError):
        _validate_row_counts([])


def test_rejects_row_count_below_minimum():
    with pytest.raises(ValueError):
        _validate_row_counts([99])


def test_rejects_non_integer_row_count():
    with pytest.raises(ValueError):
        _validate_row_counts([500.5])


def test_rejects_duplicate_row_counts():
    with pytest.raises(ValueError):
        _validate_row_counts([500, 500])
