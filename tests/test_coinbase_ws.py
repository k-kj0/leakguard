
from decimal import Decimal

import pytest

from leakguard.coinbase_ws import (
    CoinbaseLevel2Book,
    CoinbaseWebSocketError,
    _exchange_time,
)


@pytest.fixture
def snapshot_message():
    # Coinbase Exchange Level 2 snapshots contain bids and asks.
    # They do not necessarily contain an exchange "time" field.
    return {
        "type": "snapshot",
        "product_id": "BTC-USD",
        "bids": [
            ["100.00", "2.0"],
            ["99.00", "3.0"],
            ["98.00", "4.0"],
        ],
        "asks": [
            ["101.00", "1.5"],
            ["102.00", "2.5"],
            ["103.00", "3.5"],
        ],
    }


@pytest.fixture
def book(snapshot_message):
    result = CoinbaseLevel2Book("BTC-USD", depth=3)
    result.apply_snapshot(snapshot_message)
    return result


def test_snapshot_initializes_book(book):
    assert book.initialized is True
    assert max(book.bids) == Decimal("100.00")
    assert min(book.asks) == Decimal("101.00")


def test_update_changes_absolute_size(book):
    book.apply_update({
        "type": "l2update",
        "product_id": "BTC-USD",
        "changes": [["buy", "100.00", "7.25"]],
        "time": "2026-11-06T10:00:01.000Z",
    })

    assert book.bids[Decimal("100.00")] == Decimal("7.25")


def test_zero_size_deletes_level(book):
    book.apply_update({
        "type": "l2update",
        "product_id": "BTC-USD",
        "changes": [["buy", "100.00", "0"]],
        "time": "2026-11-06T10:00:01.000Z",
    })

    assert Decimal("100.00") not in book.bids
    assert max(book.bids) == Decimal("99.00")


def test_update_can_insert_new_price_level(book):
    book.apply_update({
        "type": "l2update",
        "product_id": "BTC-USD",
        "changes": [["sell", "100.50", "1.2"]],
        "time": "2026-11-06T10:00:01.000Z",
    })

    assert min(book.asks) == Decimal("100.50")
    assert book.asks[Decimal("100.50")] == Decimal("1.2")


def test_rejects_update_before_snapshot():
    result = CoinbaseLevel2Book("BTC-USD", depth=3)

    with pytest.raises(CoinbaseWebSocketError, match="before"):
        result.apply_update({
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [["buy", "100", "1"]],
        })


def test_rejects_wrong_product(snapshot_message):
    result = CoinbaseLevel2Book("ETH-USD", depth=3)

    with pytest.raises(CoinbaseWebSocketError, match="does not match"):
        result.apply_snapshot(snapshot_message)


def test_rejects_negative_update_size(book):
    with pytest.raises(CoinbaseWebSocketError, match="negative"):
        book.apply_update({
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [["buy", "100", "-1"]],
        })


def test_rejects_crossed_update_and_preserves_old_book(book):
    with pytest.raises(CoinbaseWebSocketError, match="crossed"):
        book.apply_update({
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [["sell", "99.00", "2"]],
        })

    assert min(book.asks) == Decimal("101.00")
    assert max(book.bids) == Decimal("100.00")


def test_snapshot_without_time_is_valid(book):
    assert _exchange_time({
        "type": "snapshot",
        "product_id": "BTC-USD",
        "bids": [["100", "1"]],
        "asks": [["101", "1"]],
    }) is None


def test_update_exchange_timestamp_is_normalized():
    result = _exchange_time({
        "type": "l2update",
        "time": "2026-11-06T10:00:01.000Z",
    })

    assert result == "2026-11-06T10:00:01+00:00"


def test_records_top_of_book_without_fake_sequence(book):
    record = book.to_record(
        message_index=1,
        message_type="snapshot",
        exchange_time=None,
    )

    assert record["bid_px_1"] == 100.0
    assert record["ask_px_1"] == 101.0
    assert record["message_index"] == 1
    assert record["exchange_time"] is None
    assert "sequence" not in record


def test_invalid_update_does_not_partially_mutate_book(book):
    original_bids = book.bids.copy()
    original_asks = book.asks.copy()

    with pytest.raises(CoinbaseWebSocketError):
        book.apply_update({
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [
                ["buy", "100.00", "8.0"],
                ["unknown", "99.00", "2.0"],
            ],
        })

    assert book.bids == original_bids
    assert book.asks == original_asks
