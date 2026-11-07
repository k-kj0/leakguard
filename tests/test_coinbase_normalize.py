
import pandas as pd
import pytest

from leakguard.coinbase_normalize import normalize_capture


def make_capture():
    return pd.DataFrame({
        "product_id": ["BTC-USD"] * 4,
        "message_index": [1, 2, 3, 4],
        "exchange_time": [
            "2026-01-01T00:00:00.000Z",
            "2026-01-01T00:00:00.000Z",
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


def test_normalizes_capture_onto_fixed_time_grid(tmp_path):
    source = tmp_path / "capture.csv"
    output = tmp_path / "normalized.csv"

    make_capture().to_csv(source, index=False)

    result = normalize_capture(
        source,
        output,
        sample_ms=1,
        max_staleness_ms=5,
    )

    normalized = pd.read_csv(output)

    assert len(normalized) == 4
    assert normalized["ts_ms"].is_monotonic_increasing
    assert normalized["ts_ms"].is_unique
    assert "bid_px_1" in normalized.columns
    assert "ask_px_1" in normalized.columns
    assert result["timestamp_source"] == "captured_at_utc"
    assert result["sample_interval_ms"] == 1
    assert result["experiment_ready_by_row_count"] is False
    assert output.with_suffix(".json").is_file()


def test_duplicate_exchange_times_do_not_delete_capture_states(tmp_path):
    source = tmp_path / "capture.csv"
    output = tmp_path / "normalized.csv"

    data = make_capture()
    data.to_csv(source, index=False)

    result = normalize_capture(
        source,
        output,
        sample_ms=1,
        max_staleness_ms=5,
    )

    normalized = pd.read_csv(output)

    assert result["input_rows"] == 4
    assert result["output_rows"] == 4
    assert result["raw_duplicate_exchange_timestamps"] == 1
    assert len(normalized) == 4


def test_sampling_uses_only_previously_received_states(tmp_path):
    source = tmp_path / "capture.csv"
    output = tmp_path / "normalized.csv"

    data = make_capture()

    # Remove the second and fourth source observations.
    data = data.iloc[[0, 2, 3]].copy()
    data["message_index"] = [1, 2, 3]

    data.to_csv(source, index=False)

    normalize_capture(
        source,
        output,
        sample_ms=1,
        max_staleness_ms=5,
    )

    normalized = pd.read_csv(output)

    # At 11 ms, the 10 ms state is still the latest available state.
    assert normalized.iloc[1]["bid_px_1"] == 99.0

    # At 12 ms, the new state is available.
    assert normalized.iloc[2]["bid_px_1"] == 99.2


def test_rejects_stale_state_gap(tmp_path):
    source = tmp_path / "capture.csv"
    output = tmp_path / "normalized.csv"

    data = make_capture().iloc[[0, 3]].copy()
    data["message_index"] = [1, 2]
    data.to_csv(source, index=False)

    with pytest.raises(ValueError, match="stale-data gaps"):
        normalize_capture(
            source,
            output,
            sample_ms=1,
            max_staleness_ms=1,
        )


def test_rejects_out_of_order_capture_times(tmp_path):
    source = tmp_path / "capture.csv"
    output = tmp_path / "normalized.csv"

    data = make_capture()
    data.loc[1, "captured_at_utc"] = (
        data.loc[0, "captured_at_utc"]
    )
    data.loc[0, "captured_at_utc"] = (
        "2026-01-01T00:00:00.012Z"
    )

    data.to_csv(source, index=False)

    with pytest.raises(ValueError, match="out of order"):
        normalize_capture(
            source,
            output,
            sample_ms=1,
        )


def test_rejects_crossed_order_book(tmp_path):
    source = tmp_path / "capture.csv"
    output = tmp_path / "normalized.csv"

    data = make_capture()
    data.loc[0, "bid_px_1"] = 105.0
    data.to_csv(source, index=False)

    with pytest.raises(ValueError, match="best bid"):
        normalize_capture(
            source,
            output,
            sample_ms=1,
        )


def test_rejects_multiple_products(tmp_path):
    source = tmp_path / "capture.csv"
    output = tmp_path / "normalized.csv"

    data = make_capture()
    data.loc[1, "product_id"] = "ETH-USD"
    data.to_csv(source, index=False)

    with pytest.raises(ValueError, match="exactly one"):
        normalize_capture(
            source,
            output,
            sample_ms=1,
        )


def test_rejects_non_increasing_message_index(tmp_path):
    source = tmp_path / "capture.csv"
    output = tmp_path / "normalized.csv"

    data = make_capture()
    data.loc[2, "message_index"] = 1
    data.to_csv(source, index=False)

    with pytest.raises(ValueError, match="message_index"):
        normalize_capture(
            source,
            output,
            sample_ms=1,
        )
