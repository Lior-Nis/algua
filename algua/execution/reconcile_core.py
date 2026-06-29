"""Lane-agnostic account reconcile: compare books'-expected net to broker's net per symbol.

Classifies mismatches with a grace window backed by a persisted per-symbol state table.
Both the live and paper lanes delegate here; each supplies its own `expected` net and
table names."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

DEFAULT_TOLERANCE = 1e-6      # absolute share tolerance for rounding / fractional shares
DEFAULT_GRACE_CYCLES = 3      # cycles a mismatch may persist before it escalates to unexplained


@dataclass(frozen=True)
class ReconcileResult:
    clean: bool              # True == no mismatch at all -> safe to trade this cycle
    halt: bool               # True == a mismatch persisted past the grace window -> halt
    mismatches: list[dict]   # per-symbol {symbol, expected, broker, status}


def next_cycle(conn: sqlite3.Connection, *, table: str) -> int:
    """Monotonic, persisted cycle counter (survives restarts so the grace window is
    stable)."""
    conn.execute(
        f"INSERT INTO {table}(id, n) VALUES (1, 1) ON CONFLICT(id) "
        "DO UPDATE SET n = n + 1"
    )
    conn.commit()
    return int(conn.execute(f"SELECT n FROM {table} WHERE id = 1").fetchone()["n"])


def reconcile_account(
    conn: sqlite3.Connection,
    broker_net: dict[str, float],
    expected: dict[str, float],
    cycle: int,
    *,
    state_table: str,
    tolerance: float = DEFAULT_TOLERANCE,
    grace_cycles: int = DEFAULT_GRACE_CYCLES,
) -> ReconcileResult:
    """Compare `expected` (the caller's books-net) to `broker_net` per symbol. Within
    tolerance -> clear any pending row. Otherwise record/keep a row keyed by
    first_seen_cycle; once it has persisted `grace_cycles`, mark it unexplained and
    signal halt. `clean` is True only when nothing mismatches (the caller trades only on
    a clean cycle; a pending mismatch defers, not halts)."""
    pending = {r["symbol"] for r in conn.execute(f"SELECT symbol FROM {state_table}")}
    symbols = set(expected) | set(broker_net) | pending
    mismatches: list[dict] = []
    halt = False
    for sym in sorted(symbols):
        diff = broker_net.get(sym, 0.0) - expected.get(sym, 0.0)
        if abs(diff) <= tolerance:
            conn.execute(f"DELETE FROM {state_table} WHERE symbol = ?", (sym,))
            continue
        row = conn.execute(
            f"SELECT first_seen_cycle FROM {state_table} WHERE symbol = ?", (sym,)
        ).fetchone()
        first_seen = int(row["first_seen_cycle"]) if row is not None else cycle
        status = (
            "unexplained"
            if cycle - first_seen >= grace_cycles
            else "pending"
        )
        conn.execute(
            f"INSERT INTO {state_table}"
            "(symbol, expected_qty, broker_qty, first_seen_cycle, status) "
            "VALUES (?,?,?,?,?)"
            " ON CONFLICT(symbol) DO UPDATE SET expected_qty = "
            "excluded.expected_qty,"
            "  broker_qty = excluded.broker_qty, status = excluded.status",
            (sym, expected.get(sym, 0.0), broker_net.get(sym, 0.0),
             first_seen, status),
        )
        mismatches.append({
            "symbol": sym,
            "expected": expected.get(sym, 0.0),
            "broker": broker_net.get(sym, 0.0),
            "status": status,
        })
        if status == "unexplained":
            halt = True
    conn.commit()
    return ReconcileResult(clean=not mismatches, halt=halt, mismatches=mismatches)
