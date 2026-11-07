
import numpy as np
import pandas as pd

from leakguard.data_quality import audit_dataframe


def clean_data(rows=20):
    return pd.DataFrame({
        "ts_ms": np.arange(rows, dtype=float) * 100,
        "bid_px_1": np.full(rows, 99.0),
        "ask_px_1": np.full(rows, 101.0),
        "bid_sz_1": np.full(rows, 2.0),
        "ask_sz_1": np.full(rows, 3.0),
        "bid_px_2": np.full(rows, 98.0),
        "ask_px_2": np.full(rows, 102.0),
        "bid_sz_2": np.full(rows, 4.0),
        "ask_sz_2": np.full(rows, 5.0),
        "bid_px_3": np.full(rows, 97.0),
        "ask_px_3": np.full(rows, 103.0),
        "bid_sz_3": np.full(rows, 6.0),
        "ask_sz_3": np.full(rows, 7.0),
    })


def issue_codes(report):
    return {
        issue["code"]
        for issue in report["issues"]
    }


def test_clean_fixed_interval_data_passes():
    report = audit_dataframe(
        clean_data(),
        expected_interval_ms=100,
    )

    assert report["status"] == "pass"
    assert report["checks"]["error_count"] == 0
    assert report["checks"]["warning_count"] == 0
    assert report["metrics"]["median_interval_ms"] == 100.0


def test_detects_missing_values():
    data = clean_data()
    data.loc[5, "bid_sz_1"] = np.nan

    report = audit_dataframe(data)

    assert report["status"] == "fail"
    assert "missing_or_non_numeric_values" in issue_codes(report)


def test_detects_duplicate_timestamps():
    data = clean_data()
    data.loc[5, "ts_ms"] = data.loc[4, "ts_ms"]

    report = audit_dataframe(data)

    assert report["status"] == "fail"
    assert "duplicate_timestamps" in issue_codes(report)


def test_detects_out_of_order_timestamps():
    data = clean_data()
    data.loc[5, "ts_ms"] = 9999

    report = audit_dataframe(data)

    assert report["status"] == "fail"
    assert "out_of_order_timestamps" in issue_codes(report)


def test_detects_crossed_book():
    data = clean_data()
    data.loc[5, "bid_px_1"] = 102.0

    report = audit_dataframe(data)

    assert report["status"] == "fail"
    assert "crossed_or_locked_books" in issue_codes(report)


def test_detects_negative_sizes():
    data = clean_data()
    data.loc[5, "bid_sz_1"] = -1.0

    report = audit_dataframe(data)

    assert report["status"] == "fail"
    assert "negative_sizes" in issue_codes(report)


def test_detects_invalid_depth_order():
    data = clean_data()
    data.loc[5, "bid_px_2"] = 100.0

    report = audit_dataframe(data)

    assert report["status"] == "fail"
    assert "invalid_depth_order" in issue_codes(report)


def test_warns_about_large_timestamp_gap():
    data = clean_data()
    data.loc[10:, "ts_ms"] += 1000

    report = audit_dataframe(
        data,
        expected_interval_ms=100,
    )

    assert report["status"] == "warn"
    assert "timestamp_gaps" in issue_codes(report)


def test_missing_required_column_fails():
    data = clean_data().drop(columns=["ask_sz_1"])

    report = audit_dataframe(data)

    assert report["status"] == "fail"
    assert "missing_required_columns" in issue_codes(report)


def test_empty_dataset_fails():
    report = audit_dataframe(clean_data().iloc[:0])

    assert report["status"] == "fail"
    assert "empty_dataset" in issue_codes(report)


def test_invalid_configuration_is_rejected():
    data = clean_data()

    try:
        audit_dataframe(
            data,
            expected_interval_ms=100,
            gap_multiplier=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid gap multiplier to fail.")
