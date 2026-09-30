"""Venue-ledger plumbing for the paper commands, carved out of ``paper_cmd.py`` (Story 1.3c).

Broker net positions for the account reconcile, strategy-scoped cancel, exhaustive paper-venue
fill ingest, crash-stranded order-id recovery, and the account-wide LIVE flatness check that
``paper resume`` runs before clearing a live strategy's kill-switch. Bodies are unchanged; only the
home moved, so ``paper_cmd.py`` stays under its size pin while it gains new command logic.
"""
from __future__ import annotations

import sqlite3

from algua.audit.log import append as audit_append
from algua.contracts.types import (
    ActivityWindowBroker,
    LiveReconcileBroker,
    OrderLookupBroker,
    PositionsBroker,
    ScopedCancelBroker,
)
from algua.execution.live_ledger import (
    LedgerKind,
    believed_positions,
    fill_cursor,
    ingest_activities,
    owned_open_order_ids,
    recover_stranded_broker_order_ids,
    strategy_live_symbols,
)
from algua.execution.live_reconcile import attributed_live_net
from algua.live.live_loop import _RECONCILE_TOL


def paper_broker_net(broker: PositionsBroker) -> dict[str, float]:
    """Paper broker's net positions per symbol (nonzero only) for account reconcile.

    Paper-lane only: the live analog (_broker_net_positions) lives in live_cmd, which paper_cmd
    cannot import (cli->cli).
    """
    pos = broker.get_positions()  # pandas Series symbol -> qty
    return {sym: float(q) for sym, q in pos.items() if float(q) != 0.0}


_PAPER_CURSOR_FAR_PAST = "1970-01-01T00:00:00Z"


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


def paper_scoped_cancel(conn, broker: ScopedCancelBroker, name: str) -> None:
    """Cancel only THIS strategy's open PAPER orders (never a sibling's)."""
    for oid in owned_open_order_ids(conn, broker, name, kind=LedgerKind.PAPER):
        broker.cancel_order(oid)


def ingest_paper_venue(
    conn: sqlite3.Connection, broker: ActivityWindowBroker, until: str
) -> None:
    """Exhaustively ingest the paper venue's activities into paper_venue_fills, fail-closed.
    Cursor is a broker-time high-water: fetch (cursor, until] (raises on a partial page), dedup by
    activity_id, persist `until` as the new cursor in the SAME transaction. The caller resolves
    `until` itself (never calls broker.clock() here), so a clock failure stays in its hands."""
    after = fill_cursor(conn, LedgerKind.PAPER) or _PAPER_CURSOR_FAR_PAST
    acts = broker.account_activities_window(after, until)
    ingest_activities(conn, acts, LedgerKind.PAPER, cursor_value=until)


def live_strategy_flat(
    conn: sqlite3.Connection, name: str, universe: list[str], broker: LiveReconcileBroker,
) -> tuple[bool, dict]:
    """Ingest pending broker activities, then ACCOUNT-WIDE reconcile: the strategy is flat iff its
    own believed_positions is empty AND the broker holds no UNEXPLAINED qty (broker net minus the
    books' LIVE-attributed net) in any symbol it is responsible for. A sibling LIVE strategy that
    legitimately holds the same symbol explains the broker qty and does not block resume; an orphan
    (unattributed/manual) or non-live holding does NOT explain it, so it fails closed (refuse)."""
    cursor = fill_cursor(conn, LedgerKind.LIVE)
    ingest_activities(conn, broker.account_activities(after=cursor), LedgerKind.LIVE)
    # #312: recover any crash-stranded NULL-broker_order_id live row before the flatness check, so a
    # stranded (accepted-but-not-backfilled) fill does not block resume as an unexplained residual.
    recover_stranded(conn, broker, LedgerKind.LIVE)
    own = believed_positions(conn, name, LedgerKind.LIVE)
    broker_net = {s: float(q) for s, q in broker.get_positions().items()
                  if float(q) != 0.0}
    expected = attributed_live_net(conn)
    syms = set(universe) | strategy_live_symbols(conn, name)
    unexplained = {
        s: broker_net.get(s, 0.0) - expected.get(s, 0.0)
        for s in syms
        if abs(broker_net.get(s, 0.0) - expected.get(s, 0.0)) > _RECONCILE_TOL
    }
    is_flat = (not own) and (not unexplained)
    return is_flat, {"believed": own, "broker_unexplained": unexplained}
