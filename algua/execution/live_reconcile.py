"""Account-level reconcile: the books' expected net position (Σ all strategies' signed fills) must
match the broker's netted book per symbol. Mismatches are CLASSIFIED, not binary-halted — a brief
timing skew (a fill in positions but not yet in the activities feed) is tolerated for a grace window
and only a persistent unexplained gap halts the account. The grace window survives restarts via a
persisted cycle counter."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

_TOLERANCE = 1e-6      # absolute share tolerance for rounding / fractional shares
_GRACE_CYCLES = 3      # cycles a mismatch may persist before it escalates to unexplained -> halt


@dataclass(frozen=True)
class ReconcileResult:
    clean: bool              # True == no mismatch at all -> safe to trade this cycle
    halt: bool               # True == a mismatch persisted past the grace window -> global halt
    mismatches: list[dict]   # per-symbol {symbol, expected, broker, status}


def account_expected_net(conn: sqlite3.Connection) -> dict[str, float]:
    """The books' belief of the account net position per symbol = Σ all live_fills.qty (signed),
    across every strategy (the account is shared). Zero nets are omitted."""
    rows = conn.execute("SELECT symbol, SUM(qty) AS q FROM live_fills GROUP BY symbol").fetchall()
    return {r["symbol"]: float(r["q"]) for r in rows if float(r["q"]) != 0.0}


def next_cycle(conn: sqlite3.Connection) -> int:
    """Monotonic, persisted cycle counter (survives restarts so the grace window is stable)."""
    conn.execute(
        "INSERT INTO live_cycle(id, n) VALUES (1, 1) ON CONFLICT(id) DO UPDATE SET n = n + 1"
    )
    conn.commit()
    return int(conn.execute("SELECT n FROM live_cycle WHERE id = 1").fetchone()["n"])


def reconcile(
    conn: sqlite3.Connection,
    broker_net: dict[str, float],
    cycle: int,
    tolerance: float = _TOLERANCE,
    grace_cycles: int = _GRACE_CYCLES,
) -> ReconcileResult:
    """Compare the books' expected net to the broker's net per symbol. Within tolerance → clear any
    pending row. Otherwise record/keep a pending row keyed by first_seen_cycle; once it has
    persisted `grace_cycles`, mark it unexplained and signal halt. `clean` is True only when
    NOTHING mismatches (the caller trades only on a clean cycle; a pending mismatch defers
    trading, not halts)."""
    expected = account_expected_net(conn)
    # Include symbols with an existing pending row even if now absent from both expected and broker
    # (a mismatch that resolved to flat-on-both): they must be re-examined so the stale row clears,
    # else a future mismatch on that symbol would read a long-ago first_seen_cycle and mis-halt.
    pending = {r["symbol"] for r in conn.execute("SELECT symbol FROM live_reconcile_state")}
    symbols = set(expected) | set(broker_net) | pending
    mismatches: list[dict] = []
    halt = False
    for sym in sorted(symbols):
        diff = broker_net.get(sym, 0.0) - expected.get(sym, 0.0)
        if abs(diff) <= tolerance:
            conn.execute("DELETE FROM live_reconcile_state WHERE symbol = ?", (sym,))
            continue
        row = conn.execute(
            "SELECT first_seen_cycle FROM live_reconcile_state WHERE symbol = ?", (sym,)
        ).fetchone()
        first_seen = int(row["first_seen_cycle"]) if row is not None else cycle
        status = "unexplained" if cycle - first_seen >= grace_cycles else "pending"
        conn.execute(
            "INSERT INTO live_reconcile_state"
            "(symbol, expected_qty, broker_qty, first_seen_cycle, status) VALUES (?,?,?,?,?)"
            " ON CONFLICT(symbol) DO UPDATE SET expected_qty = excluded.expected_qty,"
            "  broker_qty = excluded.broker_qty, status = excluded.status",
            (sym, expected.get(sym, 0.0), broker_net.get(sym, 0.0), first_seen, status),
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
