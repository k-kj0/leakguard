
"""Capture Coinbase Advanced Trade Level 2 market data.

This is a research-data collector, not a production trading connection.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd
import websocket


COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"


class CoinbaseWebSocketError(Exception):
    """Raised when Coinbase WebSocket data is invalid or unusable."""


def _exchange_time(message: dict[str, Any]) -> str | None:
    """Return an exchange timestamp in UTC, or None if not supplied."""
    value = message.get("time", message.get("timestamp"))

    if value is None:
        return None

    if not isinstance(value, str) or not value:
        raise CoinbaseWebSocketError(
            "Market-data timestamp must be a non-empty string"
        )

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        raise CoinbaseWebSocketError(
            f"Invalid exchange timestamp: {value!r}"
        ) from None

    if timestamp.tzinfo is None:
        raise CoinbaseWebSocketError(
            "Exchange timestamp must include a timezone"
        )

    return timestamp.tz_convert("UTC").isoformat()


def _decimal(value: Any, field: str) -> Decimal:
    """Parse a finite Decimal value."""
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise CoinbaseWebSocketError(
            f"Invalid {field}: {value!r}"
        ) from None

    if not result.is_finite():
        raise CoinbaseWebSocketError(
            f"Invalid non-finite {field}: {value!r}"
        )

    return result


def _normalise_changes(message: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Support both legacy Exchange and Advanced Trade update formats."""
    if "changes" in message:
        changes = message["changes"]

        if not isinstance(changes, list):
            raise CoinbaseWebSocketError("Changes must be a list")

        return [tuple(change) for change in changes]

    updates = message.get("updates")

    if not isinstance(updates, list):
        raise CoinbaseWebSocketError("Updates must be a list")

    changes = []

    for update in updates:
        if not isinstance(update, dict):
            raise CoinbaseWebSocketError(
                f"Invalid order-book update: {update!r}"
            )

        side = update.get("side")

        side_mapping = {
            "bid": "buy",
            "buy": "buy",
            "offer": "sell",
            "ask": "sell",
            "sell": "sell",
        }

        if side not in side_mapping:
            raise CoinbaseWebSocketError(
                f"Unknown order-book side: {side!r}"
            )

        price = update.get("price_level", update.get("price"))
        size = update.get("new_quantity", update.get("size"))

        changes.append((side_mapping[side], price, size))

    return changes


class CoinbaseLevel2Book:
    """Maintain a local Coinbase Level 2 price-level book."""

    def __init__(self, product_id: str, depth: int = 10):
        if not product_id:
            raise ValueError("product_id must not be empty")

        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.product_id = product_id
        self.depth = depth
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self.initialized = False

    def apply_snapshot(self, message: dict[str, Any]) -> None:
        """Initialize the book from a snapshot."""
        if message.get("type") != "snapshot":
            raise CoinbaseWebSocketError("Expected a snapshot message")

        if message.get("product_id") != self.product_id:
            raise CoinbaseWebSocketError(
                "Snapshot product does not match configured product"
            )

        bids: dict[Decimal, Decimal] = {}
        asks: dict[Decimal, Decimal] = {}

        # Advanced Trade represents snapshots using an updates array.
        if "updates" in message:
            changes = _normalise_changes(message)

            for side, raw_price, raw_size in changes:
                price = _decimal(raw_price, "price")
                size = _decimal(raw_size, "size")

                if price <= 0 or size <= 0:
                    raise CoinbaseWebSocketError(
                        "Snapshot prices and sizes must be positive"
                    )

                target = bids if side == "buy" else asks
                target[price] = size

        # Also support the legacy Exchange snapshot structure.
        else:
            for side_name, target in (
                ("bids", bids),
                ("asks", asks),
            ):
                levels = message.get(side_name)

                if not isinstance(levels, list):
                    raise CoinbaseWebSocketError(
                        f"Snapshot {side_name} must be a list"
                    )

                for level in levels:
                    if not isinstance(level, (list, tuple)) or len(level) != 2:
                        raise CoinbaseWebSocketError(
                            f"Invalid snapshot level: {level!r}"
                        )

                    price = _decimal(level[0], "price")
                    size = _decimal(level[1], "size")

                    if price <= 0 or size <= 0:
                        raise CoinbaseWebSocketError(
                            "Snapshot prices and sizes must be positive"
                        )

                    target[price] = size

        self._validate_book(bids, asks)

        self.bids = bids
        self.asks = asks
        self.initialized = True

    def apply_update(self, message: dict[str, Any]) -> None:
        """Apply absolute-size updates to the local order book."""
        if not self.initialized:
            raise CoinbaseWebSocketError(
                "Cannot apply update before snapshot"
            )

        if message.get("type") != "l2update":
            raise CoinbaseWebSocketError("Expected an l2update message")

        if message.get("product_id") != self.product_id:
            raise CoinbaseWebSocketError(
                "Update product does not match configured product"
            )

        changes = _normalise_changes(message)

        if not changes:
            raise CoinbaseWebSocketError(
                "Update must contain at least one change"
            )

        # Apply changes to copies first. Invalid updates must not
        # partially mutate the accepted order book.
        bids = self.bids.copy()
        asks = self.asks.copy()

        for change in changes:
            if len(change) != 3:
                raise CoinbaseWebSocketError(
                    f"Invalid order-book change: {change!r}"
                )

            side, raw_price, raw_size = change

            price = _decimal(raw_price, "price")
            size = _decimal(raw_size, "size")

            if price <= 0:
                raise CoinbaseWebSocketError(
                    "Update price must be positive"
                )

            if size < 0:
                raise CoinbaseWebSocketError(
                    "Update size cannot be negative"
                )

            if side == "buy":
                target = bids
            elif side == "sell":
                target = asks
            else:
                raise CoinbaseWebSocketError(
                    f"Unknown order-book side: {side!r}"
                )

            # Coinbase sends the new absolute size, not a size delta.
            if size == 0:
                target.pop(price, None)
            else:
                target[price] = size

        self._validate_book(bids, asks)

        self.bids = bids
        self.asks = asks

    @staticmethod
    def _validate_book(
        bids: dict[Decimal, Decimal],
        asks: dict[Decimal, Decimal],
    ) -> None:
        """Reject empty, locked, or crossed books."""
        if not bids or not asks:
            raise CoinbaseWebSocketError(
                "Order book must contain both bids and asks"
            )

        if max(bids) >= min(asks):
            raise CoinbaseWebSocketError(
                "Order book is crossed or locked"
            )

    def to_record(
        self,
        message_index: int,
        message_type: str,
        exchange_time: str | None,
        exchange_sequence_num: int | None = None,
    ) -> dict[str, Any]:
        """Return a CSV-friendly record of the current book."""
        if not self.initialized:
            raise CoinbaseWebSocketError(
                "Cannot record book before snapshot"
            )

        sorted_bids = sorted(
            self.bids.items(),
            key=lambda item: item[0],
            reverse=True,
        )[:self.depth]

        sorted_asks = sorted(
            self.asks.items(),
            key=lambda item: item[0],
        )[:self.depth]

        record: dict[str, Any] = {
            "product_id": self.product_id,
            "message_index": message_index,
            "message_type": message_type,
            "exchange_time": exchange_time,
            "captured_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "exchange_sequence_num": exchange_sequence_num,
            "bid_levels_available": len(self.bids),
            "ask_levels_available": len(self.asks),
        }

        for level in range(1, self.depth + 1):
            if level <= len(sorted_bids):
                price, size = sorted_bids[level - 1]
                record[f"bid_px_{level}"] = float(price)
                record[f"bid_sz_{level}"] = float(size)
            else:
                record[f"bid_px_{level}"] = None
                record[f"bid_sz_{level}"] = None

            if level <= len(sorted_asks):
                price, size = sorted_asks[level - 1]
                record[f"ask_px_{level}"] = float(price)
                record[f"ask_sz_{level}"] = float(size)
            else:
                record[f"ask_px_{level}"] = None
                record[f"ask_sz_{level}"] = None

        return record


def capture_coinbase_level2(
    product_id: str = "BTC-USD",
    depth: int = 10,
    duration: int = 30,
    output: str = "results/coinbase_ws_capture.csv",
) -> dict[str, Any]:
    """Capture a bounded Advanced Trade Level 2 WebSocket session."""
    if duration < 1:
        raise ValueError("duration must be at least 1 second")

    if depth < 1:
        raise ValueError("depth must be at least 1")

    book = CoinbaseLevel2Book(product_id, depth=depth)

    records: list[dict[str, Any]] = []
    started = time.monotonic()
    deadline = started + duration

    message_index = 0
    update_count = 0
    snapshot_received = False
    last_sequence_num: int | None = None

    connection = None

    try:
        connection = websocket.create_connection(
            COINBASE_WS_URL,
            timeout=10,
            enable_multithread=True,
        )

        # Advanced Trade public market-data subscription.
        # No API key or JWT is required for this public channel.
        connection.send(json.dumps({
            "type": "subscribe",
            "product_ids": [product_id],
            "channel": "level2",
        }))

        # Heartbeats help keep an otherwise quiet subscription alive.
        connection.send(json.dumps({
            "type": "subscribe",
            "channel": "heartbeats",
        }))

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            connection.settimeout(max(1, min(10, remaining)))

            try:
                raw_message = connection.recv()
            except websocket.WebSocketTimeoutException:
                continue

            if not raw_message:
                raise CoinbaseWebSocketError(
                    "WebSocket connection closed during capture"
                )

            try:
                envelope = json.loads(raw_message)
            except json.JSONDecodeError:
                raise CoinbaseWebSocketError(
                    "Received invalid JSON from Coinbase"
                ) from None

            if not isinstance(envelope, dict):
                continue

            channel = envelope.get("channel")

            if envelope.get("type") == "error" or channel == "error":
                raise CoinbaseWebSocketError(
                    f"Coinbase returned an error: {envelope}"
                )

            # Subscription acknowledgements and heartbeat messages
            # do not contain order-book snapshots or updates.
            if channel in ("subscriptions", "heartbeats"):
                continue

            if channel != "l2_data":
                continue

            events = envelope.get("events")

            if not isinstance(events, list):
                raise CoinbaseWebSocketError(
                    "Level 2 message has no valid events list"
                )

            exchange_timestamp = _exchange_time(envelope)
            sequence_num = envelope.get("sequence_num")

            if sequence_num is not None:
                try:
                    sequence_num = int(sequence_num)
                except (TypeError, ValueError):
                    raise CoinbaseWebSocketError(
                        f"Invalid exchange sequence number: {sequence_num!r}"
                    ) from None

                if (
                    last_sequence_num is not None
                    and sequence_num <= last_sequence_num
                ):
                    raise CoinbaseWebSocketError(
                        "Non-increasing exchange sequence number"
                    )

                last_sequence_num = sequence_num

            for event in events:
                if not isinstance(event, dict):
                    continue

                event_product = event.get("product_id")

                if event_product != product_id:
                    continue

                event_type = event.get("type")

                # Convert Advanced Trade events into the common
                # internal representation used by the order book.
                if event_type == "snapshot":
                    if snapshot_received:
                        raise CoinbaseWebSocketError(
                            "Received an unexpected second snapshot"
                        )

                    internal_message = {
                        "type": "snapshot",
                        "product_id": product_id,
                        "updates": event.get("updates", []),
                    }

                    book.apply_snapshot(internal_message)
                    snapshot_received = True

                elif event_type == "update":
                    if not snapshot_received:
                        raise CoinbaseWebSocketError(
                            "Received update before snapshot"
                        )

                    internal_message = {
                        "type": "l2update",
                        "product_id": product_id,
                        "updates": event.get("updates", []),
                    }

                    book.apply_update(internal_message)
                    update_count += 1

                else:
                    continue

                message_index += 1

                records.append(
                    book.to_record(
                        message_index=message_index,
                        message_type=event_type,
                        exchange_time=exchange_timestamp,
                        exchange_sequence_num=sequence_num,
                    )
                )

    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    if not snapshot_received:
        raise CoinbaseWebSocketError(
            "Capture ended without receiving a Level 2 snapshot"
        )

    if update_count == 0:
        raise CoinbaseWebSocketError(
            "Capture received a snapshot but no incremental updates"
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records).to_csv(output_path, index=False)

    metadata = {
        "exchange": "Coinbase Advanced Trade",
        "endpoint": COINBASE_WS_URL,
        "channel": "level2",
        "product_id": product_id,
        "depth": depth,
        "duration_seconds_requested": duration,
        "duration_seconds_actual": round(
            time.monotonic() - started,
            3,
        ),
        "snapshot_received": snapshot_received,
        "incremental_updates_received": update_count,
        "records_written": len(records),
        "last_exchange_sequence_num": last_sequence_num,
        "sequence_note": (
            "Coinbase Advanced Trade envelope sequence_num; "
            "not a substitute for independent packet-recovery validation."
        ),
        "timestamp_note": (
            "Exchange timestamp from the WebSocket envelope. "
            "Missing timestamps are recorded as null."
        ),
        "output_csv": str(output_path),
    }

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture Coinbase Advanced Trade Level 2 market data."
        )
    )

    parser.add_argument(
        "--product",
        default="BTC-USD",
        help="Product ID, for example BTC-USD",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="Number of bid and ask levels to record",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Capture duration in seconds",
    )
    parser.add_argument(
        "--out",
        default="results/coinbase_ws_capture.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    metadata = capture_coinbase_level2(
        product_id=args.product,
        depth=args.depth,
        duration=args.duration,
        output=args.out,
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
