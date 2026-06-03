from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pandas as pd

from algua.live.paper_loop import PaperRunResult


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
        status = "filled" if executed else "rejected"
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
