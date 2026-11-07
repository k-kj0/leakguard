
"""Reconstruct order-book snapshots from a snapshot and incremental updates.

Input format:
- Initial snapshot: columns side, price, size.
- Updates: columns sequence, ts_ms, side, price, size.
- A zero update size removes that price level.
- Sequence numbers must be contiguous after snapshot_sequence.

This is a normalized research interface, not an exchange-specific feed adapter.
"""

from decimal import Decimal, InvalidOperation

import pandas as pd

SNAPSHOT_COLUMNS = {"side", "price", "size"}
UPDATE_COLUMNS = {"sequence", "ts_ms", "side", "price", "size"}


def _decimal_price(value, field="price"):
    """Convert a price to an exact decimal representation."""
    try:
        price = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"Invalid {field}: {value!r}") from None

    if not price.is_finite() or price <= 0:
        raise ValueError(f"{field} must be finite and positive")

    return price


def _positive_integer(value, field):
    """Validate a positive integer parameter."""
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def audit_update_sequence(updates, snapshot_sequence):
    """Report missing, duplicate and out-of-order update sequence numbers.

    The first expected update is snapshot_sequence + 1.
    Gaps are represented as inclusive missing ranges.
    """
    if "sequence" not in updates.columns:
        raise ValueError("Updates must contain a sequence column")

    if (
        isinstance(snapshot_sequence, bool)
        or int(snapshot_sequence) != snapshot_sequence
        or snapshot_sequence < 0
    ):
        raise ValueError("snapshot_sequence must be a non-negative integer")

    numeric = pd.to_numeric(
        updates["sequence"],
        errors="coerce",
    )

    if numeric.isna().any():
        raise ValueError("Sequence values must be numeric and non-null")

    if ((numeric % 1) != 0).any():
        raise ValueError("Sequence values must be integers")

    seen = set()
    expected = int(snapshot_sequence) + 1

    gaps = []
    duplicates = []
    out_of_order = []

    for raw_value in numeric:
        sequence = int(raw_value)

        if sequence in seen:
            duplicates.append(sequence)
            continue

        if sequence > expected:
            gaps.append({
                "missing_from": expected,
                "missing_to": sequence - 1,
            })
        elif sequence < expected:
            out_of_order.append(sequence)

        seen.add(sequence)

        if sequence >= expected:
            expected = sequence + 1

    return {
        "is_contiguous": not gaps and not duplicates and not out_of_order,
        "gaps": gaps,
        "duplicates": duplicates,
        "out_of_order": out_of_order,
        "updates_checked": len(numeric),
        "snapshot_sequence": int(snapshot_sequence),
    }


def _validate_snapshot(snapshot, depth):
    missing = SNAPSHOT_COLUMNS - set(snapshot.columns)
    if missing:
        raise ValueError(
            f"Snapshot missing columns: {sorted(missing)}"
        )

    book = {"bid": {}, "ask": {}}

    for row in snapshot.itertuples(index=False):
        side = str(row.side).lower()

        if side not in book:
            raise ValueError("Snapshot side must be 'bid' or 'ask'")

        price = _decimal_price(row.price)

        try:
            size = float(row.size)
        except (TypeError, ValueError):
            raise ValueError("Snapshot sizes must be numeric") from None

        if not pd.notna(size) or size <= 0:
            raise ValueError("Snapshot sizes must be finite and positive")

        if price in book[side]:
            raise ValueError(
                f"Duplicate snapshot price level: {side} {price}"
            )

        book[side][price] = size

    for side in ("bid", "ask"):
        if len(book[side]) < depth:
            raise ValueError(
                f"Snapshot needs at least {depth} {side} price levels"
            )

    if max(book["bid"]) >= min(book["ask"]):
        raise ValueError("Snapshot contains a crossed or locked book")

    return book


def _make_snapshot_row(book, sequence, ts_ms, depth):
    """Convert the current price-level dictionaries to one normalized row."""
    bids = sorted(book["bid"].items(), reverse=True)[:depth]
    asks = sorted(book["ask"].items())[:depth]

    if len(bids) < depth or len(asks) < depth:
        raise ValueError(
            "Not enough price levels after update; "
            "provide a deeper initial snapshot"
        )

    row = {
        "sequence": int(sequence),
        "ts_ms": int(ts_ms),
    }

    for level, (price, size) in enumerate(bids, start=1):
        row[f"bid_px_{level}"] = float(price)
        row[f"bid_sz_{level}"] = size

    for level, (price, size) in enumerate(asks, start=1):
        row[f"ask_px_{level}"] = float(price)
        row[f"ask_sz_{level}"] = size

    if max(book["bid"]) >= min(book["ask"]):
        raise ValueError(
            f"Update {sequence} created a crossed or locked book"
        )

    return row


def reconstruct_order_book(
    snapshot,
    updates,
    snapshot_sequence,
    depth=3,
):
    """Reconstruct one normalized book observation per incremental update.

    Parameters
    ----------
    snapshot:
        DataFrame with side, price, size columns. It represents the book
        immediately after snapshot_sequence was applied.
    updates:
        DataFrame with sequence, ts_ms, side, price, size columns.
        Rows must be in sequence order.
    snapshot_sequence:
        Sequence number represented by the initial snapshot.
    depth:
        Number of bid and ask levels to emit.

    Returns
    -------
    pandas.DataFrame
        One reconstructed book state per update. Sequence and timestamp
        columns are retained for downstream validation.
    """
    depth = _positive_integer(depth, "depth")

    missing = UPDATE_COLUMNS - set(updates.columns)
    if missing:
        raise ValueError(
            f"Updates missing columns: {sorted(missing)}"
        )

    sequence_report = audit_update_sequence(
        updates,
        snapshot_sequence,
    )

    if not sequence_report["is_contiguous"]:
        raise ValueError(
            "Cannot reconstruct across invalid update sequence: "
            f"{sequence_report}"
        )

    timestamps = pd.to_numeric(
        updates["ts_ms"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError("Update timestamps must be numeric and non-null")

    if ((timestamps % 1) != 0).any():
        raise ValueError("Update timestamps must be integer milliseconds")

    if not timestamps.is_monotonic_increasing:
        raise ValueError("Update timestamps must be non-decreasing")

    book = _validate_snapshot(snapshot, depth)
    records = []

    for row in updates.itertuples(index=False):
        sequence = int(row.sequence)
        ts_ms = int(row.ts_ms)
        side = str(row.side).lower()

        if side not in ("bid", "ask"):
            raise ValueError(
                f"Update {sequence}: side must be 'bid' or 'ask'"
            )

        price = _decimal_price(row.price)

        try:
            size = float(row.size)
        except (TypeError, ValueError):
            raise ValueError(
                f"Update {sequence}: size must be numeric"
            ) from None

        if not pd.notna(size) or size < 0:
            raise ValueError(
                f"Update {sequence}: size must be finite and non-negative"
            )

        if size == 0:
            book[side].pop(price, None)
        else:
            book[side][price] = size

        records.append(
            _make_snapshot_row(
                book,
                sequence,
                ts_ms,
                depth,
            )
        )

    return pd.DataFrame(records)
