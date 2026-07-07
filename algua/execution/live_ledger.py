"""Per-strategy live books: order recording, crash-safe activity ingestion, and average-cost
P&L / NAV derivations. The broker account is the netted custodian; this ledger is the source of
truth for per-strategy attribution. Pure derivations are kept side-effect-free for testing."""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import Enum

from algua.contracts.types import OpenOrderReader, OrderLookupBroker


@dataclass(frozen=True)
class LedgerTables:
    fills: str
    activities: str
    cursor: str
    orders: str
    quarantine: str


class LedgerKind(Enum):
    LIVE = "live"
    PAPER = "paper"


_TABLES = {
    LedgerKind.LIVE: LedgerTables(
        "live_fills", "live_activities", "live_fill_cursor",
        "live_orders", "live_activity_quarantine",
    ),
    LedgerKind.PAPER: LedgerTables(
        "paper_venue_fills", "paper_venue_activities", "paper_venue_fill_cursor",
        "paper_venue_orders", "paper_venue_activity_quarantine",
    ),
}


def fills_table(kind: LedgerKind) -> str:
    """The fills table name for a ledger kind (live_fills / paper_venue_fills). Lets callers read a
    kind's fills without importing the private _TABLES mapping."""
    return _TABLES[kind].fills


@dataclass(frozen=True)
class PositionPnl:
    qty: float          # signed net position
    avg_cost: float     # average cost of the open position (0.0 when flat)
    realized: float     # realized P&L across the fill sequence
    unrealized: float   # (mark - avg_cost) * qty  (correct for long AND short)


def position_pnl(fills: list[tuple[float, float]], mark: float) -> PositionPnl:
    """Average-cost P&L from a time-ordered list of (signed_qty, price) fills.

    Same-direction (or from-flat) fills update the average cost; opposite-direction fills realize
    P&L on the closed quantity; a fill crossing through zero closes the old side then opens the new
    side at the fill price. Unrealized uses the signed qty so it is correct for shorts:
    (mark - avg) * qty."""
    qty = 0.0
    avg = 0.0
    realized = 0.0
    for f_qty, price in fills:
        if qty == 0.0 or (qty > 0) == (f_qty > 0):
            # opening or adding in the same direction: weighted-average the cost
            new_qty = qty + f_qty
            avg = (avg * abs(qty) + price * abs(f_qty)) / abs(new_qty) if new_qty != 0.0 else 0.0
            qty = new_qty
        else:
            # reducing / closing the opposite side
            closing = min(abs(f_qty), abs(qty))
            # realized: long close gains (price-avg); short close gains (avg-price)
            realized += (price - avg) * closing if qty > 0 else (avg - price) * closing
            remaining = abs(f_qty) - closing
            qty = qty + f_qty
            if remaining > 0.0:        # crossed through zero -> open the new side at this price
                avg = price
            elif qty == 0.0:
                avg = 0.0
    unrealized = (mark - avg) * qty
    return PositionPnl(qty=qty, avg_cost=avg, realized=realized, unrealized=unrealized)


def _entitlement_as_of(ts: object) -> str | None:
    """The date (``YYYY-MM-DD``) that bounds a ``DIV`` row's entitlement window, or ``None`` if
    ``ts`` is not a FULLY-valid ISO value we can trust.

    A broker ``DIV`` row carries either a date-only ``date`` field (``2026-06-06``) or a full ISO
    ``transaction_time`` (``2026-06-06T15:30:00+00:00``); both are accepted and reduced to their
    date. Validation is on the WHOLE string, never a prefix slice: an earlier ``str(ts)[:10]`` cut
    would have accepted a value like ``2026-06-06 garbage`` on its clean 10-char date prefix while
    the rest was malformed. Anything that is not entirely a valid ISO date OR ISO datetime is
    un-boundable and returns ``None`` (the caller drops the row to an account-level residual) — this
    is the fail-closed guard against a malformed ``ts`` that, sorting below every fill date, would
    otherwise credit an unbounded all-fills window."""
    s = str(ts)
    try:
        return date.fromisoformat(s).isoformat()      # date-only broker field, fully validated
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(s).date().isoformat()   # full ISO timestamp, fully validated
    except ValueError:
        return None


def strategy_cash_credit(
    conn: sqlite3.Connection, strategy: str, kind: LedgerKind
) -> float:
    """The dividend cash-flow this strategy's virtual NAV should include (#437, minimal slice).

    Broker dividend (``DIV``) activities are booked at the ACCOUNT level per symbol (the
    ``{kind}_activities`` table has no strategy column), while the backtest simulates on the
    total-return ``adj_close`` which reinvests dividends. So live/paper NAV must credit that cash or
    it understates the sizing denominator and trips the drawdown breaker on a phantom drawdown.

    Attribution is SIGNED and per-SIDE — each ``DIV`` activity row is attributed to the entitled
    side from each strategy's OWN signed position, never from a netted account total:

    - Only ``type = 'DIV'`` rows are attributed. Non-dividend cash (interest, fees, journals,
      deposits) and NULL/symbol-less rows are NOT per-security total-return and are excluded.
    - Each row is handled INDIVIDUALLY (never summed across rows first). A positive amount is a
      long-side credit, split across the strategies LONG the symbol pro-rata by their long shares; a
      negative amount is a short-dividend debit, split across the strategies SHORT the symbol
      pro-rata by their short shares. A long is thus credited and a short debited independently —
      there is NO shared long+short divisor, so an offsetting book (A long, B short, booked as a
      credit row and a debit row) attributes both sides correctly and oppositely-signed instead of
      cancelling to ~0.
    - Entitlement is bounded to the dividend's own date: a strategy's share base is its signed
      position reconstructed from fills on or before the activity date (``date(fill_ts) <=
      date(ts)``), so a past dividend's attributed credit is DETERMINISTIC and never drifts as
      later, unrelated fills accumulate. The bound is DELIBERATELY date-level, not full-timestamp:
      broker ``DIV`` rows carry a date-only ``date`` field (no intraday time), so a finer bound
      would be false precision. A consequence — DISCLOSED, not a bug — is that a fill placed the
      SAME DAY the dividend posts, even one timestamped strictly AFTER the ``DIV`` row's own
      instant, is counted as entitled (it shares the activity's date). On the regular-hours daily
      rail this is a close approximation of the exchange record-date convention (see design-doc
      limitation #3); the exact ex-date/record-date rule is deferred to the declaration-sourced one.

    Divides only when the same-side share base is positive, so it never divides by zero; a row whose
    entitled side is empty in the ledger (e.g. a short-debit row with no ledger short) contributes
    nothing to any strategy and stays an account-level residual.

    Fail-closed on an un-boundable timestamp: a ``DIV`` row whose ``ts`` is NULL, or is not
    ENTIRELY a valid ISO date/datetime, carries no entitlement window we can trust, so it is
    EXCLUDED from attribution (SQL ``AND ts IS NOT NULL`` plus the ``_entitlement_as_of`` full-
    string parse guard) and left as an unattributed account-level residual — it is never credited
    against an unbounded (all-fills) window, which a malformed ``ts`` sorting below every fill date
    would otherwise produce. The parse validates the WHOLE ``ts`` (not a 10-char prefix slice), so a
    value like ``2026-06-06 garbage`` with a well-formed date prefix but malformed suffix also fails
    closed rather than being accepted on its prefix.

    The SAME full-string validation is applied per FILL when reconstructing the entitled share base:
    each fill's ``fill_ts`` is run through ``_entitlement_as_of``, and only fills whose own date
    parses AND is on-or-before the dividend's date count toward the base. This closes the mirror of
    the ``ts`` hole — a fill with an empty/malformed ``fill_ts`` (e.g. ``''``) would otherwise sort
    below every real date under a raw ``substr(fill_ts, 1, 10) <= as_of`` comparison and be wrongly
    counted as entitled, inflating the base; instead it is excluded rather than silently satisfying
    ``<=``.

    Non-finite guard: a corrupt ``DIV`` ``amount``, a non-finite summed fill ``qty``, or a
    non-finite computed contribution is dropped (``math.isfinite`` — the convention used at
    ``assert_marks_usable``) so an ``inf``/``nan`` can never propagate into NAV and the persisted
    drawdown peak.

    Minimal-slice limitation: the per-share figure is IMPLIED from the account cash, not sourced
    from a corporate-action declaration, and a single broker-netted row (one row for an internally
    long+short book) cannot recover the per-side split — the full declaration-sourced design in
    ``docs/superpowers/specs/2026-07-05-dividend-nav-accrual-437-design.md`` is deferred."""
    t = _TABLES[kind]
    divs = conn.execute(
        f"SELECT symbol, amount, ts FROM {t.activities} "
        "WHERE type = 'DIV' AND symbol IS NOT NULL AND amount IS NOT NULL AND ts IS NOT NULL"
    ).fetchall()
    if not divs:
        return 0.0
    credit = 0.0
    for row in divs:
        sym = row["symbol"]
        amt = float(row["amount"])
        # Fail closed on a corrupt (non-finite) DIV amount so an inf/nan can never propagate into
        # NAV and the persisted drawdown peak (matches the assert_marks_usable isfinite convention).
        if not math.isfinite(amt):
            continue
        # date-vs-date entitlement bound (clean, tz-free), validating the WHOLE ts; a not-fully-ISO
        # ts is un-boundable, so the row fails closed to a residual rather than crediting an
        # all-fills window (see _entitlement_as_of).
        as_of = _entitlement_as_of(row["ts"])
        if as_of is None:
            continue
        # Reconstruct each strategy's entitled share base by summing ONLY fills whose OWN fill_ts
        # parses as a full ISO date/datetime (via _entitlement_as_of) AND falls on-or-before the
        # dividend's date. Applying the same full-string validation per fill closes the mirror of
        # the DIV-ts hole: a fill with an empty/malformed fill_ts (e.g. '') would, under a raw
        # `substr(fill_ts,1,10) <= as_of` comparison, sort BELOW every real date and be wrongly
        # counted as entitled — inflating the share base. A non-boundable fill_ts is un-datable, so
        # it is excluded from the entitled base rather than silently satisfying '<='.
        positions: dict[str, float] = {}
        for r in conn.execute(
            f"SELECT strategy, qty, fill_ts FROM {t.fills} "
            "WHERE strategy IS NOT NULL AND symbol = ?",
            (sym,),
        ):
            fill_as_of = _entitlement_as_of(r["fill_ts"])
            if fill_as_of is None or fill_as_of > as_of:
                continue
            q = float(r["qty"])
            # A non-finite fill qty can't be a real position — fail it closed out of the base so it
            # can't drive an inf/nan into the credit.
            if not math.isfinite(q):
                continue
            positions[r["strategy"]] = positions.get(r["strategy"], 0.0) + q
        my_q = positions.get(strategy, 0.0)
        if amt >= 0.0:  # long-side credit
            base = sum(q for q in positions.values() if q > 0.0)
            my_share = my_q if my_q > 0.0 else 0.0
        else:  # short-side dividend debit
            base = sum(-q for q in positions.values() if q < 0.0)
            my_share = -my_q if my_q < 0.0 else 0.0
        if base > 0.0 and math.isfinite(base) and my_share > 0.0:
            contribution = amt * (my_share / base)
            if math.isfinite(contribution):
                credit += contribution
    return credit


def record_live_order(
    conn: sqlite3.Connection,
    strategy: str,
    symbol: str,
    side: str,
    intended_notional: float | None,
    client_order_id: str,
) -> None:
    """Record a live order at submit time, keyed by client_order_id (the durable identity). A retry
    that re-submits the same client_order_id is a no-op (INSERT OR IGNORE on the UNIQUE column)."""
    conn.execute(
        "INSERT OR IGNORE INTO live_orders"
        "(strategy, symbol, side, intended_notional, client_order_id, status, submitted_ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (strategy, symbol, side, intended_notional, client_order_id, "submitted",
         datetime.now(UTC).isoformat()),
    )
    conn.commit()


def record_paper_venue_order(
    conn: sqlite3.Connection,
    strategy: str,
    symbol: str,
    side: str,
    intended_notional: float | None,
    client_order_id: str,
    *,
    strategy_id: int,
) -> bool:
    """Record a paper venue order at submit time, keyed by client_order_id. A retry that re-submits
    the same client_order_id is a no-op (INSERT OR IGNORE on the UNIQUE column). strategy_id is
    required for forward-gate attribution.

    Returns True iff this call actually inserted a NEW row (rowcount == 1); False if a row for this
    client_order_id already existed and the INSERT was IGNORED. The caller uses this to tell a row
    it freshly created THIS attempt (safe to retract on a noop/skipped — no POST) apart from a
    pre-existing row that a PRIOR run may have created and POSTed then crashed before backfill (must
    be preserved) — see delete_paper_venue_order (#311)."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO paper_venue_orders"
        "(strategy, symbol, side, intended_notional, client_order_id,"
        " strategy_id, status, submitted_ts)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (strategy, symbol, side, intended_notional, client_order_id, strategy_id, "submitted",
         datetime.now(UTC).isoformat()),
    )
    conn.commit()
    return cur.rowcount == 1


def delete_paper_venue_order(conn: sqlite3.Connection, client_order_id: str) -> None:
    """Retract a paper venue INTENT row that never became a real order — submit_sized reported
    'noop'/'skipped', so no order ever reached the venue (both sentinels return before the POST).
    The crash-safe before_submit intent (#249) is only durable-worthy once a POST could have
    happened; a phantom row with NULL broker_order_id otherwise inflates the venue order count and
    flips paper-show onto the venue-strategy display branch (#311).

    Caller MUST only invoke this for a coid it FRESHLY inserted this attempt
    (record_paper_venue_order returned True): a pre-existing NULL row is INDISTINGUISHABLE from a
    real accepted order whose
    broker_order_id backfill was lost to a crash, so it must never be deleted. The
    'broker_order_id IS NULL' guard is belt-and-suspenders on top of that caller-side gate — a real
    order's row is only ever NULL transiently before on_submitted backfills it, and this deletes
    nothing that carries a broker id."""
    conn.execute(
        "DELETE FROM paper_venue_orders WHERE client_order_id = ? AND broker_order_id IS NULL",
        (client_order_id,),
    )
    conn.commit()


def _backfill_order(
    conn: sqlite3.Connection, kind: LedgerKind, client_order_id: str, broker_order_id: str
) -> bool:
    """Attach `broker_order_id` to the {kind} order row keyed by `client_order_id`, and
    back-attribute any fill already ingested under that broker id while the mapping was missing
    (strategy was NULL) so an early fill is never orphaned. Shared by the on-submit backfill
    (live/paper) and the #312 crash-recovery pass.

    The UPDATE is CONDITIONAL (`broker_order_id IS NULL`) so a row already backfilled by a
    tick/replay is never blind-overwritten. Fill back-attribution runs ONLY when we set the row
    (rowcount == 1) or a re-read confirms the existing id already equals THIS `broker_order_id`
    (idempotent replay); if the row already carries a DIFFERENT id, we own nothing, attribute no
    fills, and return False (leave the strategy-NULL fills to fail closed at reconcile). Returns
    True iff this call owns the (coid -> broker_order_id) mapping."""
    t = _TABLES[kind]
    row = conn.execute(
        f"SELECT strategy FROM {t.orders} WHERE client_order_id = ?", (client_order_id,)
    ).fetchone()
    cur = conn.execute(
        f"UPDATE {t.orders} SET broker_order_id = ?"
        " WHERE client_order_id = ? AND broker_order_id IS NULL",
        (broker_order_id, client_order_id),
    )
    if cur.rowcount == 0:
        # No NULL row to set: either the row is absent, or it was already backfilled. Only proceed
        # to attribute fills when it was already backfilled to THIS SAME id (idempotent replay); a
        # different id means we do not own this mapping.
        existing = conn.execute(
            f"SELECT broker_order_id FROM {t.orders} WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
        if existing is None or existing["broker_order_id"] != broker_order_id:
            conn.commit()
            return False
    if row is not None:
        conn.execute(
            f"UPDATE {t.fills} SET strategy = ?"
            " WHERE broker_order_id = ? AND strategy IS NULL",
            (row["strategy"], broker_order_id),
        )
    conn.commit()
    return True


def backfill_paper_venue_broker_order_id(
    conn: sqlite3.Connection, client_order_id: str, broker_order_id: str
) -> None:
    """Attach the broker's order id to a paper venue order once the broker accepts it. Also
    back-attributes any fills already ingested under this broker order id while the mapping was
    missing (strategy was NULL), so an early fill is never orphaned in the books."""
    _backfill_order(conn, LedgerKind.PAPER, client_order_id, broker_order_id)


def backfill_broker_order_id(
    conn: sqlite3.Connection, client_order_id: str, broker_order_id: str
) -> None:
    """Attach the broker's order id once the broker accepts (covers a submit that timed out after
    Alpaca accepted it: the client_order_id row exists, the broker id arrives later). Also
    back-attributes any fills already ingested under this broker order id while the mapping was
    missing (strategy was NULL), so an early fill is never orphaned in the books."""
    _backfill_order(conn, LedgerKind.LIVE, client_order_id, broker_order_id)


@dataclass(frozen=True)
class StrandedRecovery:
    """Outcome of a #312 stranded-order recovery pass. `recovered` = client_order_ids whose broker
    order id was backfilled (crash-stranded rows resolved); `mismatched` = client_order_ids the
    broker returned an inconsistent order for (symbol/coid/id mismatch) — left NULL, not attributed,
    so they still fail closed at reconcile."""
    recovered: list[str]
    mismatched: list[str]


def recover_stranded_broker_order_ids(
    conn: sqlite3.Connection, broker: OrderLookupBroker, *, kind: LedgerKind,
) -> StrandedRecovery:
    """Auto-recover orders stranded by a crash between broker-accept and the broker_order_id
    backfill commit (#312): for each local {kind} order row with `broker_order_id IS NULL`, ask
    the order carrying that exact `client_order_id` and, on a verified match, backfill the broker id
    (which also back-attributes any fill already ingested under it).

    Read-only against the broker (never submits), so it can never double-submit. DB-first: with no
    NULL rows it returns without a broker round-trip. A 404 (`get_order_by_client_order_id`->None)
    means the coid never reached the venue (a pure #365 phantom / genuine noop) -> the row is
    preserved. As a safety boundary it REJECTS an inconsistent broker payload (returned
    client_order_id != coid, symbol != the local row's symbol, or an empty id) -> the row is left
    NULL and flagged mismatched rather than mis-attributed. Side is NOT checked (the POSTed side is
    delta-derived and can differ from the recorded intent side)."""
    t = _TABLES[kind]
    stranded = {
        r["client_order_id"]: r["symbol"]
        for r in conn.execute(
            f"SELECT client_order_id, symbol FROM {t.orders} WHERE broker_order_id IS NULL"
        )
    }
    recovered: list[str] = []
    mismatched: list[str] = []
    for coid, symbol in stranded.items():
        order = broker.get_order_by_client_order_id(coid)
        if order is None:
            continue  # not at the venue -> preserve (crash-safety / #365 phantom)
        boid = order.get("id")
        # Reject a malformed/inconsistent broker payload (safety boundary): the returned coid must
        # match exactly, the symbol must equal the local row's, and the id must be a non-empty
        # string (a truthy non-str would coerce to a bogus broker id). Any failure -> skip + flag,
        # never backfill (a coid collision must not mis-attribute; the NULL row still fails closed).
        if (order.get("client_order_id") != coid or order.get("symbol") != symbol
                or not isinstance(boid, str) or not boid.strip()):
            mismatched.append(coid)
            continue
        if _backfill_order(conn, kind, coid, boid):
            recovered.append(coid)
    return StrandedRecovery(recovered=recovered, mismatched=mismatched)


def believed_positions(
    conn: sqlite3.Connection, strategy: str, kind: LedgerKind
) -> dict[str, float]:
    """Per-symbol signed net position for a strategy = Σ its own fills.qty (nonzero only)."""
    t = _TABLES[kind]
    rows = conn.execute(
        f"SELECT symbol, SUM(qty) AS q FROM {t.fills} WHERE strategy = ? GROUP BY symbol",
        (strategy,),
    ).fetchall()
    return {r["symbol"]: float(r["q"]) for r in rows if float(r["q"]) != 0.0}


def paper_believed_positions(conn: sqlite3.Connection, strategy: str) -> dict[str, float]:
    """Convenience alias: believed_positions for the paper ledger."""
    return believed_positions(conn, strategy, LedgerKind.PAPER)


def strategy_live_symbols(conn: sqlite3.Connection, strategy: str) -> set[str]:
    """Every symbol a strategy is responsible for = the union of symbols in its live_orders and its
    live_fills. Used by the resume reconcile to scope the broker-truth check to the strategy's own
    symbols (a held-but-dropped symbol is in its fills even after it left the universe)."""
    rows = conn.execute(
        "SELECT symbol FROM live_orders WHERE strategy = ? "
        "UNION SELECT symbol FROM live_fills WHERE strategy = ?",
        (strategy, strategy),
    ).fetchall()
    return {r["symbol"] for r in rows}


def _fills_for(
    conn: sqlite3.Connection, strategy: str, symbol: str
) -> list[tuple[float, float]]:
    rows = conn.execute(
        "SELECT qty, price FROM live_fills WHERE strategy = ? AND symbol = ? ORDER BY fill_ts, id",
        (strategy, symbol),
    ).fetchall()
    return [(float(r["qty"]), float(r["price"])) for r in rows]


def strategy_nav(
    conn: sqlite3.Connection, strategy: str, allocation: float, marks: dict[str, float]
) -> float:
    """NAV = allocation + Σ realized + Σ unrealized across the strategy's symbols, plus credited
    dividend/cash activities (#437). `marks` supplies the current price per symbol (a missing mark
    falls back to the average cost → 0 unrealized)."""
    symbols = {
        r["symbol"]
        for r in conn.execute(
            "SELECT DISTINCT symbol FROM live_fills WHERE strategy = ?", (strategy,)
        )
    }
    total = allocation
    for sym in symbols:
        fills = _fills_for(conn, strategy, sym)
        pnl = position_pnl(fills, mark=marks.get(sym, fills[-1][1] if fills else 0.0))
        total += pnl.realized + pnl.unrealized
    total += strategy_cash_credit(conn, strategy, LedgerKind.LIVE)
    return total


def fill_cursor(conn: sqlite3.Connection, kind: LedgerKind) -> str | None:
    t = _TABLES[kind]
    row = conn.execute(
        f"SELECT cursor FROM {t.cursor} WHERE name = 'activities'"
    ).fetchone()
    return row["cursor"] if row else None


def ingest_activities(
    conn: sqlite3.Connection,
    activities: list[dict],
    kind: LedgerKind,
    *,
    cursor_value: str | None = None,
) -> None:
    """Idempotently record a batch of broker activities and advance the cursor in ONE transaction.

    FILL activities become signed fills rows (buy +qty, sell -qty), attributed to a strategy
    via order_id -> orders.broker_order_id; non-fill activities become activities rows.
    Dedupe is by `activity_id` (UNIQUE), so re-pulling an overlap window never double-counts; a
    re-pull DOES back-fill a previously-missing strategy/broker_order_id (COALESCE on conflict), so
    a fill ingested before its order mapping existed is attributed once the mapping lands. The
    cursor advances in the same transaction as the inserts, so a crash leaves books and cursor
    consistent (overlap replay re-dedupes).

    When `cursor_value` is provided (paper path), it is stored as the cursor regardless of the
    activity ids seen. When `None` (live path), `max(activity_id)` is stored as today.

    A single malformed activity must NOT wedge the loop: a shape error (`ValueError`/`KeyError`/
    `TypeError`) dead-letters that one activity into the quarantine table and processing continues,
    so the cursor still advances PAST the poison item and the next cycle does not re-fetch it
    forever (#250). Quarantining is recoverable — the raw payload is preserved for human triage,
    and a quarantined fill the book is now missing still surfaces as broker-vs-ledger drift at the
    reconcile guard (fail-closed backstop). Real infrastructure errors (e.g. `sqlite3.Error`) are
    NOT shape errors: they propagate and roll back the whole batch, preserving the old
    all-or-nothing fail-closed behavior."""
    t = _TABLES[kind]
    try:
        max_id: str | None = None
        for act in activities:
            # A missing/empty `id` is NOT a quarantinable shape error: the id IS the cursor, so
            # there is no safe way to advance past an id-less item, and quarantining it (NULL id)
            # would re-quarantine the same item every cycle. Fail closed instead — matching the
            # broker adapter, which raises on an id-less activity (alpaca_broker.py).
            if not act.get("id"):
                raise ValueError(f"activity missing 'id'; cannot advance cursor: {act!r}")
            aid = str(act["id"])
            try:
                _ingest_one_activity(conn, act, aid, kind)
            except (ValueError, KeyError, TypeError) as exc:
                _quarantine_activity(conn, aid, act, exc, kind)
            max_id = aid if max_id is None or aid > max_id else max_id
        cursor_to_store = cursor_value if cursor_value is not None else max_id
        if cursor_to_store is not None:
            conn.execute(
                f"INSERT INTO {t.cursor}(name, cursor) VALUES ('activities', ?) "
                "ON CONFLICT(name) DO UPDATE SET cursor = excluded.cursor",
                (cursor_to_store,),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _ingest_one_activity(
    conn: sqlite3.Connection, act: dict, aid: str, kind: LedgerKind
) -> None:
    """Record one activity (FILL -> signed fills table, else -> activities table). Raises a shape
    error (`ValueError`/`KeyError`/`TypeError`) on a malformed activity; caller quarantines it."""
    t = _TABLES[kind]
    if act.get("activity_type") == "FILL":
        side = act.get("side")
        if side not in {"buy", "sell"}:
            raise ValueError(f"bad fill side {side!r}")
        qty = float(act["qty"])
        price = float(act["price"])
        if qty <= 0.0 or price <= 0.0:
            raise ValueError("fill qty and price must be positive")
        signed = qty if side == "buy" else -qty
        boid = act.get("order_id")
        strat_row = conn.execute(
            f"SELECT strategy FROM {t.orders} WHERE broker_order_id = ?", (boid,)
        ).fetchone()
        conn.execute(
            f"INSERT INTO {t.fills}"
            "(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
            " VALUES (?,?,?,?,?,?,?)"
            f" ON CONFLICT(activity_id) DO UPDATE SET"
            f"  strategy = COALESCE({t.fills}.strategy, excluded.strategy),"
            f"  broker_order_id = COALESCE({t.fills}.broker_order_id,"
            "                             excluded.broker_order_id)",
            (
                aid, boid,
                strat_row["strategy"] if strat_row else None,
                act["symbol"], signed, price,
                act.get("transaction_time", ""),
            ),
        )
    else:
        conn.execute(
            f"INSERT OR IGNORE INTO {t.activities}"
            "(activity_id, type, symbol, amount, ts, raw) VALUES (?,?,?,?,?,?)",
            (
                aid, act.get("activity_type", "UNKNOWN"), act.get("symbol"),
                float(act["net_amount"]) if act.get("net_amount") is not None else None,
                act.get("date") or act.get("transaction_time"),
                json.dumps(act),
            ),
        )


def _quarantine_activity(
    conn: sqlite3.Connection, aid: str, act: dict, exc: Exception, kind: LedgerKind
) -> None:
    """Dead-letter a malformed (but id-bearing) activity so the loop can advance past it (#250).
    Dedup is by `activity_id` (INSERT OR IGNORE), so re-pulling an overlap window never
    double-quarantines the same item. Id-less activities never reach here — they fail closed."""
    t = _TABLES[kind]
    try:
        raw = json.dumps(act, default=str)
    except (TypeError, ValueError):
        raw = repr(act)
    conn.execute(
        f"INSERT OR IGNORE INTO {t.quarantine}(activity_id, error, raw) VALUES (?,?,?)",
        (aid, f"{type(exc).__name__}: {exc}", raw),
    )


def owned_open_order_ids(
    conn: sqlite3.Connection, broker: OpenOrderReader, strategy: str,
    *, kind: LedgerKind = LedgerKind.LIVE,
) -> list[str]:
    """The broker order ids of THIS strategy's currently-open orders: list the account's open
    orders and keep those whose client_order_id maps (via the order ledger) to `strategy`. Used to
    scope cancellation so one strategy never cancels a sibling's orders. `kind` selects the order
    ledger (live_orders / paper_venue_orders)."""
    open_orders = broker.list_open_orders()
    owned = {
        r["client_order_id"]
        for r in conn.execute(
            f"SELECT client_order_id FROM {_TABLES[kind].orders} WHERE strategy = ?", (strategy,)
        )
    }
    return [o["id"] for o in open_orders if o.get("client_order_id") in owned]
