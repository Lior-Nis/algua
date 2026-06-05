from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime

import pandas as pd

from algua.live.paper_loop import PaperRunResult

# Alpaca client_order_id allows up to 128 chars; keep ours under that and strip anything outside
# [A-Za-z0-9_-] so a symbol or strategy name with odd characters can't produce an invalid id.
_COID_SANITIZE = re.compile(r"[^A-Za-z0-9_-]")


def client_order_id(strategy: str, decision_ts: datetime, symbol: str) -> str:
    """Deterministic Alpaca client_order_id for one (strategy, decision_ts, symbol). Identical
    inputs always produce the same id, so a retried submit (after a transient failure) or a re-run
    of the same tick reuses the id and Alpaca de-duplicates rather than double-filling (#18, #24).
    The decision timestamp is normalised to UTC so the id does not depend on the caller's tzinfo."""
    ts = decision_ts.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    raw = f"{strategy}-{ts}-{symbol}"
    return _COID_SANITIZE.sub("_", raw)[:128]


def persist_run(conn: sqlite3.Connection, result: PaperRunResult) -> None:
    """Persist a run's orders and fills. Fills link to their order by broker_order_id.

    A paper run is a full from-scratch replay, so it REPLACES this strategy's paper book
    (orders + fills) rather than appending — otherwise re-running a replay would double the
    persisted positions. Cross-run history lives in audit_log, not here. (Incremental /
    session semantics arrive with the real wall-clock paper adapter.)
    """
    conn.execute(
        "DELETE FROM paper_fills WHERE order_id IN "
        "(SELECT id FROM paper_orders WHERE strategy = ?)",
        (result.strategy,),
    )
    conn.execute("DELETE FROM paper_orders WHERE strategy = ?", (result.strategy,))

    now = datetime.now(UTC).isoformat()
    fills_by_order: dict[str, list] = {}
    for f in result.fills:
        fills_by_order.setdefault(f.broker_order_id, []).append(f)

    for record in result.orders:
        intent = record.intent
        # Read the broker id submit() returned rather than reconstructing sim-{seq} positionally,
        # so a skipped/"noop" submit can't shift the mapping (#30).
        broker_order_id = record.broker_order_id
        # A rejected fill (zero-qty) carries no shares; the order's status reflects whether any
        # shares actually executed, and only executing fills are persisted as fills.
        executed = [f for f in fills_by_order.get(broker_order_id, []) if f.qty != 0.0]
        # Derive status from the fills' own status rather than hardcoding "filled": a buy clamped
        # to available cash produces Fill.status="partial", which must be preserved here so callers
        # can distinguish fully-filled orders from cash-constrained ones.
        if any(f.status == "partial" for f in executed):
            status = "partial"
        elif executed:
            status = "filled"
        else:
            status = "rejected"
        cols = (
            "(strategy, symbol, side, target_weight,"
            " decision_ts, submitted_ts, status, broker_order_id)"
        )
        cur = conn.execute(
            f"INSERT INTO paper_orders{cols} VALUES (?,?,?,?,?,?,?,?)",
            (result.strategy, intent.symbol, intent.side.value, intent.target_weight,
             intent.decision_ts.isoformat(), now, status, broker_order_id),
        )
        order_row_id = cur.lastrowid
        for f in executed:
            conn.execute(
                "INSERT INTO paper_fills(order_id, symbol, qty, price, fill_ts) VALUES (?,?,?,?,?)",
                (order_row_id, f.symbol, f.qty, f.price, f.fill_ts.isoformat()),
            )
    conn.commit()


def count_orders(conn: sqlite3.Connection, strategy: str) -> int:
    """Number of persisted paper orders for a strategy (the `paper show` order count)."""
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM paper_orders WHERE strategy = ?", (strategy,)
        ).fetchone()[0]
    )


def derive_positions(conn: sqlite3.Connection, strategy: str) -> dict[str, float]:
    rows = conn.execute(
        "SELECT f.symbol AS symbol, SUM(f.qty) AS qty FROM paper_fills f "
        "JOIN paper_orders o ON o.id = f.order_id WHERE o.strategy = ? GROUP BY f.symbol",
        (strategy,),
    ).fetchall()
    return {r["symbol"]: float(r["qty"]) for r in rows if float(r["qty"]) != 0.0}


def reconcile(derived: dict[str, float], broker_positions: pd.Series) -> bool:
    broker = {s: float(q) for s, q in broker_positions.items() if float(q) != 0.0}
    return {s: q for s, q in derived.items() if q != 0.0} == broker


def record_submitted_order(
    conn: sqlite3.Connection, strategy: str, symbol: str, side: str,
    target_weight: float, decision_ts: str | None, broker_order_id: str,
) -> None:
    """Persist ONE accepted live order IMMEDIATELY after the broker accepts it, so a mid-tick death
    can never leave Alpaca holding an order the DB never recorded (#18). Each row commits on its own
    rather than being batched after the whole loop.

    Idempotent on (strategy, broker_order_id): a crash/retry or a duplicate Alpaca client_order_id
    path that re-returns the SAME broker order leaves the existing row untouched instead of writing
    a duplicate (the unique index enforces this; INSERT OR IGNORE makes it a no-op)."""
    conn.execute(
        "INSERT OR IGNORE INTO paper_orders"
        "(strategy, symbol, side, target_weight, decision_ts, submitted_ts,"
        " status, broker_order_id) VALUES (?,?,?,?,?,?,?,?)",
        (strategy, symbol, side, target_weight, decision_ts,
         datetime.now(UTC).isoformat(), "submitted", broker_order_id),
    )
    conn.commit()


def get_peak_equity(conn: sqlite3.Connection, strategy: str) -> float | None:
    row = conn.execute(
        "SELECT peak_equity FROM strategy_peaks WHERE strategy = ?", (strategy,)
    ).fetchone()
    return float(row["peak_equity"]) if row is not None else None


def update_peak_equity(conn: sqlite3.Connection, strategy: str, equity: float) -> float:
    """Persist the running peak equity for a strategy (the drawdown denominator across ticks) and
    return the new peak. The peak only ever ratchets up; a tick's equity below it is a drawdown the
    breaker can act on (#27)."""
    prior = get_peak_equity(conn, strategy)
    peak = equity if prior is None else max(prior, equity)
    conn.execute(
        "INSERT INTO strategy_peaks(strategy, peak_equity, updated_at) VALUES (?,?,?) "
        "ON CONFLICT(strategy) DO UPDATE SET peak_equity=excluded.peak_equity, "
        "updated_at=excluded.updated_at",
        (strategy, peak, datetime.now(UTC).isoformat()),
    )
    conn.commit()
    return peak


def clear_peak_equity(conn: sqlite3.Connection, strategy: str) -> None:
    """Drop a strategy's persisted peak so the next tick re-bases the high-water mark to current
    equity. Called on resume after a trip: without it, a strategy halted by the drawdown breaker
    and flattened to cash would re-trip every tick against its stale pre-loss peak (#27).

    Semantics: the peak is thus the high-water mark *since the last tripped resume*, not lifetime.
    A manual `paper kill` then `resume` also re-bases it, so a manual halt/resume can lower the
    drawdown denominator to current equity — intentional (the operator is re-baselining)."""
    conn.execute("DELETE FROM strategy_peaks WHERE strategy = ?", (strategy,))
    conn.commit()


def record_tick_snapshot(
    conn: sqlite3.Connection, strategy: str, *, tick_ts: str, decision_ts: str | None,
    equity: float, peak_equity: float | None, positions: dict[str, float], n_submitted: int,
    reconcile_ok: bool,
) -> None:
    """Append one completed-tick snapshot (equity + positions) for a strategy — the per-tick
    operability/equity-curve record read by `paper show`."""
    conn.execute(
        "INSERT INTO tick_snapshots(strategy, tick_ts, decision_ts, equity, peak_equity, "
        "positions, n_submitted, reconcile_ok) VALUES (?,?,?,?,?,?,?,?)",
        (strategy, tick_ts, decision_ts, equity, peak_equity, json.dumps(positions),
         n_submitted, 1 if reconcile_ok else 0),
    )
    conn.commit()


def latest_tick_snapshot(conn: sqlite3.Connection, strategy: str) -> dict | None:
    """The most recent tick snapshot for a strategy (positions parsed back to a dict), or None."""
    row = conn.execute(
        "SELECT tick_ts, decision_ts, equity, peak_equity, positions, n_submitted, reconcile_ok "
        "FROM tick_snapshots WHERE strategy = ? ORDER BY id DESC LIMIT 1", (strategy,)
    ).fetchone()
    if row is None:
        return None
    return {
        "tick_ts": row["tick_ts"], "decision_ts": row["decision_ts"], "equity": row["equity"],
        "peak_equity": row["peak_equity"], "positions": json.loads(row["positions"]),
        "n_submitted": row["n_submitted"], "reconcile_ok": bool(row["reconcile_ok"]),
    }


def recent_orders(conn: sqlite3.Connection, strategy: str, limit: int = 10) -> list[dict]:
    """The most recent paper_orders rows for a strategy, newest first."""
    rows = conn.execute(
        "SELECT symbol, side, status, broker_order_id, submitted_ts FROM paper_orders "
        "WHERE strategy = ? ORDER BY id DESC LIMIT ?", (strategy, limit),
    ).fetchall()
    return [dict(r) for r in rows]
