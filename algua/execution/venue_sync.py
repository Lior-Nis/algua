"""Paper-venue sync: fill ingest and crash-stranded order recovery (Story 2.2, #685).

``ingest_paper_venue`` and ``recover_stranded`` moved here unchanged from ``cli/paper_venue.py`` so
the paper commands and the paper exit drain (``execution/paper_exit_drain.py``) share one body. They
reach only the ledger and the audit log. Only the paper operator's ``ingest_paper_venue`` advances
the shared paper fill cursor; the drain uses ``ingest_paper_venue_keep_cursor``, which never does.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from algua.audit.log import append as audit_append
from algua.contracts.types import ActivityWindowBroker, OrderLookupBroker
from algua.execution.live_ledger import (
    LedgerKind,
    fill_cursor,
    ingest_activities,
    recover_stranded_broker_order_ids,
)

_PAPER_CURSOR_FAR_PAST = "1970-01-01T00:00:00Z"
#: How far before the stored cursor each sync re-opens its window (#691). The cursor is the broker
#: clock at the previous read, but Alpaca can publish a fill's activity after that read with a
#: ``transaction_time`` before it; re-reading this overlap (free: ingest de-duplicates by activity
#: id) catches a feed lag up to this long.
PAPER_INGEST_LOOKBACK = timedelta(minutes=30)


def _window_start(cursor: str | None) -> str:
    """The ``after`` bound of a paper sync window: the cursor minus ``PAPER_INGEST_LOOKBACK``. The
    far-past sentinel (no cursor yet) is returned unshifted. An unparseable cursor raises
    ``ValueError`` (fail closed) rather than silently re-reading from a guessed point."""
    if not cursor or cursor == _PAPER_CURSOR_FAR_PAST:
        return _PAPER_CURSOR_FAR_PAST
    parsed = datetime.fromisoformat(cursor)
    if parsed.tzinfo is None:
        raise ValueError(f"paper fill cursor {cursor!r} carries no timezone")
    return (parsed - PAPER_INGEST_LOOKBACK).astimezone(UTC).isoformat()


def recover_stranded(
    conn: sqlite3.Connection, broker: OrderLookupBroker, kind: LedgerKind
) -> None:
    """#312: backfill broker_order_id onto any crash-stranded NULL order row (asks the venue for the
    order carrying its client_order_id; never submits). ACCOUNT-WIDE, so the audit is too
    (strategy=None) — a per-strategy label would misattribute a sibling's order."""
    outcome = recover_stranded_broker_order_ids(conn, broker, kind=kind)
    if outcome.recovered:
        audit_append(conn, actor="system", action="stranded_order_recovered",
                     reason=f"{len(outcome.recovered)} backfilled: {outcome.recovered}",
                     strategy=None)
    if outcome.mismatched:
        audit_append(conn, actor="system", action="stranded_recovery_mismatch",
                     reason=f"{len(outcome.mismatched)} broker mismatch: {outcome.mismatched}",
                     strategy=None)


def ingest_paper_venue(
    conn: sqlite3.Connection, broker: ActivityWindowBroker, until: str
) -> None:
    """Exhaustively ingest the paper venue's activities into paper_venue_fills, fail-closed.
    Cursor is a broker-time high-water: fetch (cursor - PAPER_INGEST_LOOKBACK, until] (raises on a
    partial page; the overlap catches a late-published activity, #691), dedup by activity_id,
    persist `until` as the new cursor in the SAME transaction. The caller resolves `until` itself
    (never calls broker.clock() here), so a clock failure stays in its hands."""
    after = fill_cursor(conn, LedgerKind.PAPER) or _PAPER_CURSOR_FAR_PAST
    acts = broker.account_activities_window(_window_start(after), until)
    ingest_activities(conn, acts, LedgerKind.PAPER, cursor_value=until)


def ingest_paper_venue_keep_cursor(
    conn: sqlite3.Connection, broker: ActivityWindowBroker, until: str
) -> None:
    """The paper exit drain's ingest (Story 2.2 §3.2): fetch ``(cursor, until]`` exactly like
    ``ingest_paper_venue``, then re-store the cursor it read instead of ``until``, so the drain
    never moves the shared cursor. Every reader derives the same value (``fill_cursor(...) or
    _PAPER_CURSOR_FAR_PAST``), so an absent row becoming the sentinel reads identically.
    Re-fetching is idempotent (ingest de-duplicates by activity id), so a fill the venue publishes
    late, with a ``transaction_time`` before this ``until``, is still inside the next sync's window.
    Never pass ``cursor_value=None`` here: that stores an activity id, the live ledger's form."""
    after = fill_cursor(conn, LedgerKind.PAPER) or _PAPER_CURSOR_FAR_PAST
    acts = broker.account_activities_window(_window_start(after), until)
    ingest_activities(conn, acts, LedgerKind.PAPER, cursor_value=after)
