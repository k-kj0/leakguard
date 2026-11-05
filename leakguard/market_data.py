
"""Public market-data ingestion for Leakguard.

Currently supports Coinbase Exchange REST Level 2 snapshots.

This module fetches snapshots only. It does not implement the
WebSocket feed or incremental update reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


COINBASE_EXCHANGE_BASE = "https://api.exchange.coinbase.com"


class MarketDataError(RuntimeError):
    """Raised when public market data cannot be fetched or validated."""


def _parse_levels(levels, side, depth):
    if not isinstance(levels, list):
        raise MarketDataError(f"Coinbase {side} levels must be a list")

    parsed = []

    for item in levels:
        if not isinstance(item, list) or len(item) < 2:
            raise MarketDataError(
                f"Invalid Coinbase {side} level: {item!r}"
            )

        try:
            price = float(item[0])
            size = float(item[1])
        except (TypeError, ValueError):
            raise MarketDataError(
                f"Non-numeric Coinbase {side} level: {item!r}"
            ) from None

        if not math.isfinite(price) or price <= 0:
            raise MarketDataError(f"Invalid {side} price: {price}")

        if not math.isfinite(size) or size <= 0:
            raise MarketDataError(f"Invalid {side} size: {size}")

        parsed.append((price, size))

    prices = [price for price, _ in parsed]
    if len(prices) != len(set(prices)):
        raise MarketDataError(
            f"Coinbase returned duplicate {side} price levels"
        )

    parsed.sort(
        key=lambda level: level[0],
        reverse=(side == "bid"),
    )

    if len(parsed) < depth:
        raise MarketDataError(
            f"Coinbase returned {len(parsed)} {side} levels; "
            f"{depth} required"
        )

    return parsed[:depth]


def parse_coinbase_l2_snapshot(payload, product_id, depth=10):
    """Validate and normalize a Coinbase Exchange Level 2 snapshot.

    Returns:
        snapshot: DataFrame with side, price, size columns.
        metadata: Exchange sequence, product and receipt timestamp.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")

    if not isinstance(payload, dict):
        raise MarketDataError("Coinbase response must be a JSON object")

    try:
        sequence = int(payload["sequence"])
    except (KeyError, TypeError, ValueError):
        raise MarketDataError(
            "Coinbase snapshot has no valid sequence number"
        ) from None

    if sequence < 0:
        raise MarketDataError("Snapshot sequence cannot be negative")

    bids = _parse_levels(payload.get("bids"), "bid", depth)
    asks = _parse_levels(payload.get("asks"), "ask", depth)

    if bids[0][0] >= asks[0][0]:
        raise MarketDataError(
            "Coinbase snapshot is crossed or locked"
        )

    rows = []

    for price, size in bids:
        rows.append({
            "side": "bid",
            "price": price,
            "size": size,
        })

    for price, size in asks:
        rows.append({
            "side": "ask",
            "price": price,
            "size": size,
        })

    snapshot = pd.DataFrame(
        rows,
        columns=["side", "price", "size"],
    )

    metadata = {
        "exchange": "coinbase_exchange",
        "product_id": product_id,
        "level": 2,
        "sequence": sequence,
        "depth_per_side": depth,
        "received_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    return snapshot, metadata


def fetch_coinbase_l2_snapshot(
    product_id="BTC-USD",
    depth=10,
    timeout=10,
):
    """Fetch one public Coinbase Exchange Level 2 snapshot."""
    if not isinstance(product_id, str) or not product_id.strip():
        raise ValueError("product_id must be a non-empty string")

    if timeout <= 0:
        raise ValueError("timeout must be positive")

    encoded_product = quote(product_id.strip(), safe="-")
    url = (
        f"{COINBASE_EXCHANGE_BASE}/products/"
        f"{encoded_product}/book?level=2"
    )

    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Leakguard/0.3 public-market-data-research",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise MarketDataError(
                    f"Coinbase returned HTTP {response.status}"
                )
            payload = json.loads(response.read().decode("utf-8"))

    except HTTPError as exc:
        raise MarketDataError(
            f"Coinbase HTTP error: {exc.code}"
        ) from exc
    except URLError as exc:
        raise MarketDataError(
            f"Could not reach Coinbase: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise MarketDataError("Coinbase request timed out") from exc
    except json.JSONDecodeError as exc:
        raise MarketDataError(
            "Coinbase returned invalid JSON"
        ) from exc

    return parse_coinbase_l2_snapshot(
        payload,
        product_id=product_id.strip(),
        depth=depth,
    )


def save_snapshot(snapshot, metadata, output_path):
    """Save normalized snapshot CSV and a JSON metadata sidecar."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot.to_csv(output_path, index=False)

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return output_path, metadata_path


def main():
    parser = argparse.ArgumentParser(
        description="Fetch a public Coinbase Exchange Level 2 snapshot."
    )
    parser.add_argument("--product", default="BTC-USD")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument(
        "--out",
        default="data/coinbase_btc_usd_snapshot.csv",
    )
    parser.add_argument("--timeout", type=float, default=10)

    args = parser.parse_args()

    snapshot, metadata = fetch_coinbase_l2_snapshot(
        product_id=args.product,
        depth=args.depth,
        timeout=args.timeout,
    )

    csv_path, json_path = save_snapshot(
        snapshot,
        metadata,
        args.out,
    )

    print(f"Exchange: {metadata['exchange']}")
    print(f"Product: {metadata['product_id']}")
    print(f"Snapshot sequence: {metadata['sequence']}")
    print(f"Levels per side: {metadata['depth_per_side']}")
    print(f"CSV saved: {csv_path}")
    print(f"Metadata saved: {json_path}")


if __name__ == "__main__":
    main()
