
import pytest

from leakguard.coinbase_ws import (
    CoinbaseLevel2Book,
    CoinbaseWebSocketError,
)


@pytest.fixture
def snapshot_message():
    return {
        "type": "snapshot",
        "product_id": "BTC-USD",
        "bids": [
            ["100.00", "2.0", 3],
            ["99.00", "3.0", 2],
            ["98.00", "4.0", 1],
        ],
        "asks": [
            ["101.00", "1.5", 2],
            ["102.00", "2.5", 1],
            ["103.00", "3.5", 4],
        ],
        "time": "2026-11-06T10:00:00.000Z",
    }


@pytest.fixture
def book(snapshot_message):
    result = CoinbaseLevel2Book("BTC-USD", depth=3)
    result.apply_snapshot(snapshot_message)
    return result


def test_snapshot_initializes_book(book):
    assert book.initialized is True
    assert max(book.bids) == 100
    assert min(book.asks) == 101


def test_update_changes_absolute_size(book):
    book.apply_update({
        "type": "l2update",
        "product_id": "BTC-USD",
        "changes": [["buy", "100.00", "7.25"]],
        "time": "2026-11-06T10:00:01.000Z",
    })

    assert book.bids[next(p for p in book.bids if p == 100)] == 7.25


def test_zero_size_deletes_level(book):
    book.apply_update({
        "type": "l2update",
        "product_id": "BTC-USD",
        "changes": [["buy", "100.00", "0"]],
        "time": "2026-11-06T10:00:01.000Z",
    })

    assert 100 not in book.bids
    assert max(book.bids) == 99


def test_update_can_insert_new_price_level(book):
    book.apply_update({
        "type": "l2update",
        "product_id": "BTC-USD",
        "changes": [["sell", "100.50", "1.2"]],
        "time": "2026-11-06T10:00:01.000Z",
    })

    assert min(book.asks) == 100.5
    assert book.asks[next(p for p in book.asks if p == 100.5)] == 1.2


def test_rejects_update_before_snapshot():
    book = CoinbaseLevel2Book("BTC-USD", depth=3)

    with pytest.raises(CoinbaseWebSocketError, match="before"):
        book.apply_update({
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [["buy", "100", "1"]],
        })


def test_rejects_wrong_product(snapshot_message):
    book = CoinbaseLevel2Book("ETH-USD", depth=3)

    with pytest.raises(CoinbaseWebSocketError, match="does not match"):
        book.apply_snapshot(snapshot_message)


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

    assert min(book.asks) == 101
    assert max(book.bids) == 100


def test_records_top_of_book_without_fake_sequence(book):
    record = book.to_record(
        message_index=1,
        message_type="snapshot",
        exchange_time="2026-11-06T10:00:00+00:00",
    )

    assert record["bid_px_1"] == 100
    assert record["ask_px_1"] == 101
    assert record["message_index"] == 1
    assert "sequence" not in record
