
"""Capture and reconstruct Coinbase Exchange Level 2 WebSocket data.

The Exchange level2 channel sends an initial snapshot followed by
incremental updates. Update sizes are absolute sizes at a price level;
zero removes that level.

This is a research collector, not an execution system.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
import websocket


COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"


class CoinbaseWebSocketError(RuntimeError):
    """Raised for invalid or incomplete Coinbase WebSocket data."""


def _decimal(value, field):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise CoinbaseWebSocketError(
            f"Invalid {field}: {value!r}"
        ) from None

    if not result.is_finite():
        raise CoinbaseWebSocketError(f"{field} must be finite")

    return result


class CoinbaseLevel2Book:
    """Maintain a local Level 2 book from Coinbase snapshot/update messages."""

    def __init__(self, product_id="BTC-USD", depth=10):
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError("product_id must be a non-empty string")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be a positive integer")

        self.product_id = product_id.strip()
        self.depth = depth
        self.bids = {}
        self.asks = {}
        self.initialized = False

    def _parse_levels(self, levels, side):
        if not isinstance(levels, list):
            raise CoinbaseWebSocketError(
                f"Snapshot {side} must be a list"
            )

        result = {}

        for level in levels:
            if not isinstance(level, list) or len(level) < 2:
                raise CoinbaseWebSocketError(
                    f"Invalid snapshot level: {level!r}"
                )

            price = _decimal(level[0], "price")
            size = _decimal(level[1], "size")

            if price <= 0:
                raise CoinbaseWebSocketError("Price must be positive")
            if size <= 0:
                raise CoinbaseWebSocketError("Snapshot size must be positive")
            if price in result:
                raise CoinbaseWebSocketError(
                    f"Duplicate {side} price level: {price}"
                )

            result[price] = size

        if not result:
            raise CoinbaseWebSocketError(
                f"Snapshot has no {side} levels"
            )

        return result

    def _validate_book(self):
        if not self.bids or not self.asks:
            raise CoinbaseWebSocketError("Order book has an empty side")

        if max(self.bids) >= min(self.asks):
            raise CoinbaseWebSocketError(
                "Order book is crossed or locked"
            )

    def apply_snapshot(self, message):
        if message.get("type") != "snapshot":
            raise CoinbaseWebSocketError("Expected snapshot message")

        if message.get("product_id") != self.product_id:
            raise CoinbaseWebSocketError("Snapshot product does not match")

        bids = self._parse_levels(message.get("bids"), "bid")
        asks = self._parse_levels(message.get("asks"), "ask")

        self.bids = bids
        self.asks = asks
        self._validate_book()
        self.initialized = True

    def apply_update(self, message):
        if not self.initialized:
            raise CoinbaseWebSocketError(
                "Cannot apply updates before the initial snapshot"
            )

        if message.get("type") != "l2update":
            raise CoinbaseWebSocketError("Expected l2update message")

        if message.get("product_id") != self.product_id:
            raise CoinbaseWebSocketError("Update product does not match")

        changes = message.get("changes")
        if not isinstance(changes, list) or not changes:
            raise CoinbaseWebSocketError(
                "Update must contain a non-empty changes list"
            )

        # Validate all changes before mutating the live book.
        parsed_changes = []

        for change in changes:
            if not isinstance(change, list) or len(change) != 3:
                raise CoinbaseWebSocketError(
                    f"Invalid update change: {change!r}"
                )

            raw_side, raw_price, raw_size = change
            side = str(raw_side).lower()

            if side == "buy":
                book_side = "bid"
            elif side == "sell":
                book_side = "ask"
            else:
                raise CoinbaseWebSocketError(
                    f"Unknown update side: {raw_side!r}"
                )

            price = _decimal(raw_price, "update price")
            size = _decimal(raw_size, "update size")

            if price <= 0:
                raise CoinbaseWebSocketError("Update price must be positive")
            if size < 0:
                raise CoinbaseWebSocketError("Update size cannot be negative")

            parsed_changes.append((book_side, price, size))

        # Apply changes to copies so a bad update cannot partially mutate
        # the current book.
        new_bids = self.bids.copy()
        new_asks = self.asks.copy()

        for side, price, size in parsed_changes:
            target = new_bids if side == "bid" else new_asks

            if size == 0:
                target.pop(price, None)
            else:
                target[price] = size

        old_bids, old_asks = self.bids, self.asks
        self.bids, self.asks = new_bids, new_asks

        try:
            self._validate_book()
        except CoinbaseWebSocketError:
            self.bids, self.asks = old_bids, old_asks
            raise

    def to_record(self, message_index, message_type, exchange_time):
        """Return one fixed-width top-of-book record.

        Missing levels are represented as None, never fabricated.
        """
        if not self.initialized:
            raise CoinbaseWebSocketError("Book has no initial snapshot")

        bids = sorted(self.bids.items(), reverse=True)[:self.depth]
        asks = sorted(self.asks.items())[:self.depth]

        record = {
            "product_id": self.product_id,
            "message_index": int(message_index),
            "message_type": message_type,
            "exchange_time": exchange_time,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "bid_levels_available": len(bids),
            "ask_levels_available": len(asks),
        }

        for level in range(1, self.depth + 1):
            if level <= len(bids):
                price, size = bids[level - 1]
                record[f"bid_px_{level}"] = float(price)
                record[f"bid_sz_{level}"] = float(size)
            else:
                record[f"bid_px_{level}"] = None
                record[f"bid_sz_{level}"] = None

            if level <= len(asks):
                price, size = asks[level - 1]
                record[f"ask_px_{level}"] = float(price)
                record[f"ask_sz_{level}"] = float(size)
            else:
                record[f"ask_px_{level}"] = None
                record[f"ask_sz_{level}"] = None

        return record


def _exchange_time(message):
    value = message.get("time")
    if not isinstance(value, str) or not value:
        raise CoinbaseWebSocketError(
            "Market-data message has no exchange timestamp"
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


def capture_coinbase_level2(
    product_id="BTC-USD",
    depth=10,
    duration_seconds=30,
    output_path="results/coinbase_ws_btc_usd.csv",
    connection_timeout=10,
):
    """Subscribe, capture a bounded live sample, and save CSV + metadata."""
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(duration_seconds)
        or duration_seconds <= 0
    ):
        raise ValueError("duration_seconds must be positive and finite")

    if connection_timeout <= 0:
        raise ValueError("connection_timeout must be positive")

    book = CoinbaseLevel2Book(product_id, depth)
    records = []
    message_index = 0
    update_count = 0
    started_at = datetime.now(timezone.utc)

    ws = None

    try:
        ws = websocket.create_connection(
            COINBASE_WS_URL,
            timeout=connection_timeout,
            enable_multithread=True,
        )
        ws.settimeout(1)

        subscription = {
            "type": "subscribe",
            "product_ids": [product_id],
            "channels": ["level2"],
        }
        ws.send(json.dumps(subscription))

        deadline = time.monotonic() + duration_seconds

        while time.monotonic() < deadline:
            try:
                raw_message = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except websocket.WebSocketConnectionClosedException as exc:
                raise CoinbaseWebSocketError(
                    "WebSocket connection closed before capture completed"
                ) from exc

            if not raw_message:
                continue

            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                raise CoinbaseWebSocketError(
                    "Coinbase sent invalid JSON"
                ) from None

            if not isinstance(message, dict):
                raise CoinbaseWebSocketError(
                    "Coinbase message must be a JSON object"
                )

            message_type = message.get("type")

            if message_type == "error":
                raise CoinbaseWebSocketError(
                    f"Coinbase feed error: {message}"
                )

            if message_type == "subscriptions":
                continue

            if message_type in ("heartbeat", "status", "ticker"):
                continue

            if message_type == "snapshot":
                if book.initialized:
                    raise CoinbaseWebSocketError(
                        "Received an unexpected second snapshot"
                    )
                book.apply_snapshot(message)

            elif message_type == "l2update":
                if not book.initialized:
                    raise CoinbaseWebSocketError(
                        "Received an update before the initial snapshot"
                    )
                book.apply_update(message)
                update_count += 1

            else:
                continue

            message_index += 1
            records.append(
                book.to_record(
                    message_index=message_index,
                    message_type=message_type,
                    exchange_time=_exchange_time(message),
                )
            )

    except websocket.WebSocketException as exc:
        raise CoinbaseWebSocketError(
            f"WebSocket transport error: {exc}"
        ) from exc
    finally:
        if ws is not None:
            ws.close()

    if not book.initialized:
        raise CoinbaseWebSocketError(
            "No initial Level 2 snapshot received"
        )

    if update_count == 0:
        raise CoinbaseWebSocketError(
            "Snapshot received, but no incremental updates arrived "
            "during the capture window"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(records)
    frame.to_csv(output_path, index=False)

    metadata = {
        "exchange": "coinbase_exchange",
        "product_id": product_id,
        "channel": "level2",
        "depth_per_side": depth,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds_requested": duration_seconds,
        "messages_recorded": message_index,
        "updates_recorded": update_count,
        "initial_snapshot_received": True,
        "sequence_number_note": (
            "The Exchange Level2 messages are recorded in arrival order "
            "using a local message_index. No exchange sequence number "
            "is inferred or fabricated."
        ),
    }

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return frame, metadata, output_path, metadata_path


def main():
    parser = argparse.ArgumentParser(
        description="Capture Coinbase Exchange Level 2 WebSocket data."
    )
    parser.add_argument("--product", default="BTC-USD")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument(
        "--out",
        default="results/coinbase_ws_btc_usd.csv",
    )
    parser.add_argument("--timeout", type=float, default=10)

    args = parser.parse_args()

    frame, metadata, csv_path, json_path = capture_coinbase_level2(
        product_id=args.product,
        depth=args.depth,
        duration_seconds=args.duration,
        output_path=args.out,
        connection_timeout=args.timeout,
    )

    print(f"Product: {metadata['product_id']}")
    print(f"Messages recorded: {metadata['messages_recorded']}")
    print(f"Updates recorded: {metadata['updates_recorded']}")
    print(f"CSV saved: {csv_path}")
    print(f"Metadata saved: {json_path}")
    print(f"Rows captured: {len(frame)}")


if __name__ == "__main__":
    main()
