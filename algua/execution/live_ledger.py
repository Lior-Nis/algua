"""Per-strategy live books: order recording, crash-safe activity ingestion, and average-cost
P&L / NAV derivations. The broker account is the netted custodian; this ledger is the source of
truth for per-strategy attribution. Pure derivations are kept side-effect-free for testing."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum


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
) -> None:
    """Record a paper venue order at submit time, keyed by client_order_id. A retry that re-submits
    the same client_order_id is a no-op (INSERT OR IGNORE on the UNIQUE column). strategy_id is
    required for forward-gate attribution."""
    conn.execute(
        "INSERT OR IGNORE INTO paper_venue_orders"
        "(strategy, symbol, side, intended_notional, client_order_id,"
        " strategy_id, status, submitted_ts)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (strategy, symbol, side, intended_notional, client_order_id, strategy_id, "submitted",
         datetime.now(UTC).isoformat()),
    )
    conn.commit()


def backfill_paper_venue_broker_order_id(
    conn: sqlite3.Connection, client_order_id: str, broker_order_id: str
) -> None:
    """Attach the broker's order id to a paper venue order once the broker accepts it. Also
    back-attributes any fills already ingested under this broker order id while the mapping was
    missing (strategy was NULL), so an early fill is never orphaned in the books."""
    row = conn.execute(
        "SELECT strategy FROM paper_venue_orders WHERE client_order_id = ?", (client_order_id,)
    ).fetchone()
    conn.execute(
        "UPDATE paper_venue_orders SET broker_order_id = ? WHERE client_order_id = ?",
        (broker_order_id, client_order_id),
    )
    if row is not None:
        conn.execute(
            "UPDATE paper_venue_fills SET strategy = ?"
            " WHERE broker_order_id = ? AND strategy IS NULL",
            (row["strategy"], broker_order_id),
        )
    conn.commit()


def backfill_broker_order_id(
    conn: sqlite3.Connection, client_order_id: str, broker_order_id: str
) -> None:
    """Attach the broker's order id once the broker accepts (covers a submit that timed out after
    Alpaca accepted it: the client_order_id row exists, the broker id arrives later). Also
    back-attributes any fills already ingested under this broker order id while the mapping was
    missing (strategy was NULL), so an early fill is never orphaned in the books."""
    row = conn.execute(
        "SELECT strategy FROM live_orders WHERE client_order_id = ?", (client_order_id,)
    ).fetchone()
    conn.execute(
        "UPDATE live_orders SET broker_order_id = ? WHERE client_order_id = ?",
        (broker_order_id, client_order_id),
    )
    if row is not None:
        conn.execute(
            "UPDATE live_fills SET strategy = ? WHERE broker_order_id = ? AND strategy IS NULL",
            (row["strategy"], broker_order_id),
        )
    conn.commit()


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
    """NAV = allocation + Σ realized + Σ unrealized across the strategy's symbols. `marks` supplies
    the current price per symbol (a missing mark falls back to the average cost → 0 unrealized)."""
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
    conn: sqlite3.Connection, broker: object, strategy: str
) -> list[str]:
    """The broker order ids of THIS strategy's currently-open orders: list the account's open
    orders and keep those whose client_order_id maps (via live_orders) to `strategy`. Used to
    scope cancellation so one strategy never cancels a sibling's orders."""
    open_orders = broker.list_open_orders()  # type: ignore[attr-defined]
    owned = {
        r["client_order_id"]
        for r in conn.execute(
            "SELECT client_order_id FROM live_orders WHERE strategy = ?", (strategy,)
        )
    }
    return [o["id"] for o in open_orders if o.get("client_order_id") in owned]
