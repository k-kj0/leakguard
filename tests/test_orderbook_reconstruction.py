
import pandas as pd
import pytest

from leakguard.reconstruction import (
    audit_update_sequence,
    reconstruct_order_book,
)


@pytest.fixture
def initial_snapshot():
    return pd.DataFrame([
        {"side": "bid", "price": 99.0, "size": 10},
        {"side": "bid", "price": 98.0, "size": 20},
        {"side": "bid", "price": 97.0, "size": 30},
        {"side": "ask", "price": 101.0, "size": 11},
        {"side": "ask", "price": 102.0, "size": 21},
        {"side": "ask", "price": 103.0, "size": 31},
    ])


@pytest.fixture
def valid_updates():
    return pd.DataFrame([
        {
            "sequence": 101,
            "ts_ms": 1001,
            "side": "bid",
            "price": 99.0,
            "size": 0,
        },
        {
            "sequence": 102,
            "ts_ms": 1002,
            "side": "ask",
            "price": 101.0,
            "size": 15,
        },
        {
            "sequence": 103,
            "ts_ms": 1003,
            "side": "bid",
            "price": 100.0,
            "size": 5,
        },
    ])


def test_sequence_audit_accepts_contiguous_updates(valid_updates):
    result = audit_update_sequence(
        valid_updates,
        snapshot_sequence=100,
    )

    assert result["is_contiguous"] is True
    assert result["gaps"] == []
    assert result["duplicates"] == []
    assert result["out_of_order"] == []


def test_sequence_audit_detects_missing_update():
    updates = pd.DataFrame({
        "sequence": [101, 103, 104],
    })

    result = audit_update_sequence(
        updates,
        snapshot_sequence=100,
    )

    assert result["is_contiguous"] is False
    assert result["gaps"] == [
        {"missing_from": 102, "missing_to": 102}
    ]


def test_sequence_audit_detects_duplicate():
    updates = pd.DataFrame({
        "sequence": [101, 102, 102, 103],
    })

    result = audit_update_sequence(
        updates,
        snapshot_sequence=100,
    )

    assert result["duplicates"] == [102]
    assert result["is_contiguous"] is False


def test_sequence_audit_detects_out_of_order_update():
    updates = pd.DataFrame({
        "sequence": [101, 103, 102, 104],
    })

    result = audit_update_sequence(
        updates,
        snapshot_sequence=100,
    )

    assert result["out_of_order"] == [102]
    assert result["is_contiguous"] is False


def test_reconstruction_applies_updates(
    initial_snapshot,
    valid_updates,
):
    result = reconstruct_order_book(
        initial_snapshot,
        valid_updates,
        snapshot_sequence=100,
        depth=3,
    )

    assert len(result) == 3

    # First update deletes the old best bid at 99.
    assert result.iloc[0]["bid_px_1"] == 98.0

    # Second update changes the best ask quantity.
    assert result.iloc[1]["ask_px_1"] == 101.0
    assert result.iloc[1]["ask_sz_1"] == 15.0

    # Third update inserts a new best bid at 100.
    assert result.iloc[2]["bid_px_1"] == 100.0
    assert result.iloc[2]["bid_sz_1"] == 5.0

    assert result["sequence"].tolist() == [101, 102, 103]


def test_reconstruction_refuses_sequence_gaps(
    initial_snapshot,
):
    updates = pd.DataFrame([
        {
            "sequence": 101,
            "ts_ms": 1001,
            "side": "bid",
            "price": 99.0,
            "size": 5,
        },
        {
            "sequence": 103,
            "ts_ms": 1003,
            "side": "ask",
            "price": 101.0,
            "size": 5,
        },
    ])

    with pytest.raises(ValueError, match="invalid update sequence"):
        reconstruct_order_book(
            initial_snapshot,
            updates,
            snapshot_sequence=100,
        )


def test_reconstruction_rejects_crossed_book(
    initial_snapshot,
):
    updates = pd.DataFrame([
        {
            "sequence": 101,
            "ts_ms": 1001,
            "side": "ask",
            "price": 98.5,
            "size": 10,
        },
    ])

    with pytest.raises(ValueError, match="crossed or locked"):
        reconstruct_order_book(
            initial_snapshot,
            updates,
            snapshot_sequence=100,
        )


def test_reconstruction_rejects_negative_size(
    initial_snapshot,
):
    updates = pd.DataFrame([
        {
            "sequence": 101,
            "ts_ms": 1001,
            "side": "bid",
            "price": 99.0,
            "size": -5,
        },
    ])

    with pytest.raises(ValueError, match="non-negative"):
        reconstruct_order_book(
            initial_snapshot,
            updates,
            snapshot_sequence=100,
        )


def test_reconstruction_rejects_bad_timestamps(
    initial_snapshot,
):
    updates = pd.DataFrame([
        {
            "sequence": 101,
            "ts_ms": 1002,
            "side": "bid",
            "price": 99.0,
            "size": 5,
        },
        {
            "sequence": 102,
            "ts_ms": 1001,
            "side": "ask",
            "price": 101.0,
            "size": 5,
        },
    ])

    with pytest.raises(ValueError, match="non-decreasing"):
        reconstruct_order_book(
            initial_snapshot,
            updates,
            snapshot_sequence=100,
        )
