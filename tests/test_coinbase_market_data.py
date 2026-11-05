
import pytest

from leakguard.market_data import (
    MarketDataError,
    parse_coinbase_l2_snapshot,
)


@pytest.fixture
def coinbase_payload():
    return {
        "sequence": 123456,
        "bids": [
            ["100.00", "2.5", 3],
            ["99.00", "4.0", 2],
            ["98.00", "5.0", 1],
        ],
        "asks": [
            ["101.00", "1.5", 2],
            ["102.00", "3.0", 1],
            ["103.00", "6.0", 4],
        ],
    }


def test_parses_coinbase_l2_snapshot(coinbase_payload):
    snapshot, metadata = parse_coinbase_l2_snapshot(
        coinbase_payload,
        "BTC-USD",
        depth=3,
    )

    assert len(snapshot) == 6
    assert metadata["exchange"] == "coinbase_exchange"
    assert metadata["sequence"] == 123456
    assert snapshot.iloc[0]["side"] == "bid"
    assert snapshot.iloc[0]["price"] == 100.0
    assert snapshot.iloc[3]["side"] == "ask"
    assert snapshot.iloc[3]["price"] == 101.0


def test_snapshot_is_sorted_best_price_first(coinbase_payload):
    snapshot, _ = parse_coinbase_l2_snapshot(
        coinbase_payload,
        "BTC-USD",
        depth=3,
    )

    bids = snapshot[snapshot["side"] == "bid"]["price"].tolist()
    asks = snapshot[snapshot["side"] == "ask"]["price"].tolist()

    assert bids == [100.0, 99.0, 98.0]
    assert asks == [101.0, 102.0, 103.0]


def test_rejects_crossed_book(coinbase_payload):
    coinbase_payload["asks"][0][0] = "99.00"

    with pytest.raises(MarketDataError, match="crossed or locked"):
        parse_coinbase_l2_snapshot(
            coinbase_payload,
            "BTC-USD",
            depth=3,
        )


def test_rejects_insufficient_depth(coinbase_payload):
    with pytest.raises(MarketDataError, match="levels"):
        parse_coinbase_l2_snapshot(
            coinbase_payload,
            "BTC-USD",
            depth=4,
        )


def test_rejects_invalid_size(coinbase_payload):
    coinbase_payload["bids"][0][1] = "-2"

    with pytest.raises(MarketDataError, match="size"):
        parse_coinbase_l2_snapshot(
            coinbase_payload,
            "BTC-USD",
            depth=3,
        )


def test_rejects_missing_sequence(coinbase_payload):
    del coinbase_payload["sequence"]

    with pytest.raises(MarketDataError, match="sequence"):
        parse_coinbase_l2_snapshot(
            coinbase_payload,
            "BTC-USD",
            depth=3,
        )


def test_rejects_duplicate_price_level(coinbase_payload):
    coinbase_payload["bids"].append(["100.00", "3.0", 1])

    with pytest.raises(MarketDataError, match="duplicate"):
        parse_coinbase_l2_snapshot(
            coinbase_payload,
            "BTC-USD",
            depth=3,
        )
