
import pandas as pd
import pytest

from leakguard.coinbase_normalize import normalize_capture


def make_capture():
    return pd.DataFrame({
        "product_id": ["BTC-USD"] * 4,
        "exchange_time": [
            "2026-01-01T00:00:00.000Z",
            "2026-01-01T00:00:00.001Z",
            "2026-01-01T00:00:00.002Z",
            "2026-01-01T00:00:00.003Z",
        ],
        "captured_at_utc": [
            "2026-01-01T00:00:00.010Z",
            "2026-01-01T00:00:00.011Z",
            "2026-01-01T00:00:00.012Z",
            "2026-01-01T00:00:00.013Z",
        ],
        "bid_px_1": [99.0, 99.1, 99.2, 99.3],
        "ask_px_1": [101.0, 101.1, 101.2, 101.3],
        "bid_sz_1": [2.0, 2.1, 2.2, 2.3],
        "ask_sz_1": [3.0, 3.1, 3.2, 3.3],
        "bid_px_2": [98.0, 98.1, 98.2, 98.3],
        "ask_px_2": [102.0, 102.1, 102.2, 102.3],
        "bid_sz_2": [4.0, 4.1, 4.2, 4.3],
        "ask_sz_2": [5.0, 5.1, 5.2, 5.3],
        "bid_px_3": [97.0, 97.1, 97.2, 97.3],
        "ask_px_3": [103.0, 103.1, 103.2, 103.3],
        "bid_sz_3": [6.0, 6.1, 6.2, 6.3],
        "ask_sz_3": [7.0, 7.1, 7.2, 7.3],
    })


def test_normalizes_coinbase_capture(tmp_path):
    source = tmp_path / "capture.csv"
    output = tmp_path / "normalized.csv"

    make_capture().to_csv(source, index=False)

    result = normalize_capture(source, output)

    normalized = pd.read_csv(output)

    assert len(normalized) == 4
    assert "ts_ms" in normalized.columns
    assert "bid_px_1" in normalized.columns
    assert "ask_px_1" in normalized.columns
    assert result["timestamp_source"] == "exchange_time"
    assert result["experiment_ready_by_row_count"] is False
    assert output.with_suffix(".json").is_file()


def test_duplicate_timestamps_keep_last_book_state(tmp_path):
    source = tmp_path / "capture.csv"
    output = tmp_path / "normalized.csv"

    data = make_capture()
    data.loc[1, "exchange_time"] = data.loc[0, "exchange_time"]
    data.to_csv(source, index=False)

    result = normalize_capture(source, output)

    normalized = pd.read_csv(output)

    assert len(normalized) == 3
    assert result["duplicate_timestamp_rows_removed"] == 1
    assert normalized.iloc[0]["bid_px_1"] == 99.1


def test_uses_receive_time_if_exchange_time_is_missing(tmp_path):
    source = tmp_path / "capture.csv"
    output = tmp_path / "normalized.csv"

    data = make_capture()
    data.loc[1, "exchange_time"] = None
    data.to_csv(source, index=False)

    result = normalize_capture(source, output)

    assert result["timestamp_source"] == "captured_at_utc"


def test_rejects_crossed_order_book(tmp_path):
    source = tmp_path / "capture.csv"
    output = tmp_path / "normalized.csv"

    data = make_capture()
    data.loc[0, "bid_px_1"] = 105.0
    data.to_csv(source, index=False)

    with pytest.raises(ValueError, match="best bid"):
        normalize_capture(source, output)


def test_rejects_multiple_products(tmp_path):
    source = tmp_path / "capture.csv"
    output = tmp_path / "normalized.csv"

    data = make_capture()
    data.loc[1, "product_id"] = "ETH-USD"
    data.to_csv(source, index=False)

    with pytest.raises(ValueError, match="multiple products"):
        normalize_capture(source, output)
