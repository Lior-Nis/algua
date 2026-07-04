"""Single-sourced emergency offset-liquidation loop (issue #336).

The breach/emergency flatten loop — cancel resting orders, reconcile the account, then offset every
believed position via ``broker.submit_offset`` (RECORDING each offset in the books first so its fill
attributes back to the strategy and ``believed_positions`` drops to flat) — used to be copy-pasted
across three CLI sites (paper trade-tick breach, paper ``flatten`` command, live trade-tick breach)
and had drifted between the copies. It lives here, once, so the safety-critical control flow is
single-sourced; each call site injects only what genuinely varies (``cancel`` scoping and the
``ingest`` cursor) and builds its own result payload from the returned facts.
"""
from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from algua.audit.log import append as audit_append
from algua.contracts.types import OffsetBroker
from algua.execution.live_ledger import (
    LedgerKind,
    backfill_broker_order_id,
    backfill_paper_venue_broker_order_id,
    believed_positions,
    paper_believed_positions,
    record_live_order,
    record_paper_venue_order,
)
from algua.execution.order_state import client_order_id
from algua.execution.reconcile_core import DEFAULT_TOLERANCE
from algua.observability.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class FlattenResult:
    """The primitive facts of a flatten attempt; each call site builds its own payload from these.

    ``n_offsets`` is the number of offset orders that ACTUALLY went out; ``flatten_error`` is the
    stringified exception if the loop failed part-way (``None`` on a clean run)."""

    n_offsets: int
    flatten_error: str | None


def _believed(conn: sqlite3.Connection, name: str, kind: LedgerKind) -> dict[str, float]:
    if kind is LedgerKind.LIVE:
        return believed_positions(conn, name, LedgerKind.LIVE)
    return paper_believed_positions(conn, name)


def _record(
    conn: sqlite3.Connection, name: str, symbol: str, side: str, coid: str,
    kind: LedgerKind, strategy_id: int | None,
) -> None:
    if kind is LedgerKind.LIVE:
        record_live_order(conn, name, symbol, side, None, coid)
        return
    if strategy_id is None:  # PAPER venue records need a strategy_id for forward-gate attribution
        raise ValueError("flatten_strategy: PAPER kind requires strategy_id")
    record_paper_venue_order(conn, name, symbol, side, None, coid, strategy_id=strategy_id)


def _backfill(
    conn: sqlite3.Connection, coid: str, oid: str, kind: LedgerKind
) -> None:
    if kind is LedgerKind.LIVE:
        backfill_broker_order_id(conn, coid, oid)
        return
    backfill_paper_venue_broker_order_id(conn, coid, oid)


def flatten_strategy(  # noqa: PLR0913
    conn: sqlite3.Connection,
    broker: OffsetBroker,
    name: str,
    kind: LedgerKind,
    *,
    lane: str,
    cancel: Callable[[], None],
    ingest: Callable[[], None],
    strategy_id: int | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    held: Callable[[], dict[str, float]] | None = None,
) -> FlattenResult:
    """Cancel this strategy's resting orders, reconcile the account, then offset every believed
    position above the dust tolerance.

    ``cancel`` and ``ingest`` are injected because they genuinely vary per call site: ``cancel`` is
    a scoped cancel on the live multi-strategy loop but an account-wide cancel on the paper sites;
    ``ingest`` resolves a different broker cursor per lane. ``strategy_id`` is required for the
    PAPER lane (forward-gate attribution) and unused for LIVE. Each offset is RECORDED in the books
    before it is submitted so its fill attributes back to the strategy and ``believed_positions``
    drops to flat — else the resume gate would block resume forever. The kill-switch (tripped by the
    caller before this runs) prevents a re-run from re-offsetting, so the per-attempt
    ``client_order_id`` is safe.

    When ``held`` is provided (LIVE lane only), each offset is capped to the actually broker-held
    signed quantity: this ensures the offset can never increase exposure or flip sign. The held
    getter is called once after ingest() and reconciles the broker account.

    Sub-tolerance ("dust") positions are skipped: ``submit_offset`` already noops sub-nano residuals
    (#269), and recording a spurious order for a rounding residual would leak a phantom fill.

    Fails SAFE: ANY exception (not only ``BrokerError``) is captured into ``flatten_error`` plus an
    audited ``flatten_failed`` row, so the emergency exit never crashes with an unstructured
    traceback. Returns the ``FlattenResult`` facts; the call site owns payload construction.
    """
    n_offsets = 0
    flatten_error: str | None = None
    try:
        cancel()
        ingest()
        held_positions = held() if held is not None else None
        for symbol, qty in _believed(conn, name, kind).items():
            if abs(qty) <= DEFAULT_TOLERANCE:
                continue
            # If held is provided, cap the offset to the actually-held signed quantity (LIVE lane).
            if held_positions is not None:
                h = held_positions.get(symbol, 0.0)
                if not math.isfinite(h):
                    # A non-finite broker qty (NaN/inf) cannot bound the offset: min(|qty|, nan)
                    # would fall back to the full believed qty and bypass the cap. We cannot prove
                    # risk-reduction, so fail closed loudly (the caller surfaces break-glass).
                    raise ValueError(
                        f"non-finite broker-held quantity for {symbol!r} ({h!r}); "
                        "cannot prove a risk-reducing offset — refusing to submit"
                    )
                if abs(h) <= DEFAULT_TOLERANCE:
                    continue  # Nothing actually held; don't open a fresh position off stale belief
                close = min(abs(qty), abs(h))
                if close <= DEFAULT_TOLERANCE:
                    continue  # Rounding residual; skip
                offset_qty = math.copysign(close, h)  # Sign follows held direction
            else:
                offset_qty = qty
            coid = client_order_id(name, now(), symbol)
            side = "sell" if offset_qty > 0 else "buy"
            _record(conn, name, symbol, side, coid, kind, strategy_id)
            oid = broker.submit_offset(symbol, offset_qty, coid)
            _backfill(conn, coid, oid, kind)
            n_offsets += 1
    except Exception as exc:  # noqa: BLE001 — emergency path must fail safe, never propagate
        flatten_error = str(exc)
        audit_append(conn, actor="system", action="flatten_failed", reason=str(exc), strategy=name)
        log.error("flatten_failed", extra={"fields": {"strategy": name, "lane": lane}},
                  exc_info=True)
    return FlattenResult(n_offsets=n_offsets, flatten_error=flatten_error)
