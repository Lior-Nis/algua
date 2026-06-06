"""Per-strategy live capital allocations (the fixed sizing denominator). Append-only lifecycle:
the active allocation is the newest non-revoked row for a strategy. Σ(active capital) is capped at
account equity so the book can never over-commit the shared account."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


class AllocationError(ValueError):
    """An allocation request that would over-commit the account, or a deallocation of a
    non-flat strategy."""


def total_allocated(conn: sqlite3.Connection) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(capital), 0.0) AS t FROM strategy_allocations WHERE revoked_ts IS NULL"
    ).fetchone()
    return float(row["t"])


def active_allocation(conn: sqlite3.Connection, strategy_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM strategy_allocations WHERE strategy_id = ? AND revoked_ts IS NULL "
        "ORDER BY id DESC LIMIT 1",
        (strategy_id,),
    ).fetchone()


def allocate(conn: sqlite3.Connection, strategy_id: int, capital: float, actor: str,
             account_equity: float) -> None:
    """Set a strategy's live capital base. Revokes any prior active allocation (re-allocation),
    then enforces Σ(active capital across all strategies) ≤ account_equity. A re-allocation resets
    the strategy's NAV drawdown peak (the operator's deliberate capital change)."""
    if capital <= 0.0:
        raise AllocationError("capital must be positive")
    now = datetime.now(UTC).isoformat()
    existing = active_allocation(conn, strategy_id)
    prior = float(existing["capital"]) if existing is not None else 0.0
    prospective = total_allocated(conn) - prior + capital
    if prospective > account_equity:
        raise AllocationError(
            f"Σ allocations {prospective:.2f} exceeds account equity {account_equity:.2f}"
        )
    if existing is not None:
        conn.execute("UPDATE strategy_allocations SET revoked_ts = ? WHERE id = ?",
                     (now, existing["id"]))
    conn.execute(
        "INSERT INTO strategy_allocations(strategy_id, capital, effective_ts, actor) "
        "VALUES (?,?,?,?)",
        (strategy_id, capital, now, actor),
    )
    conn.commit()


def deallocate(conn: sqlite3.Connection, strategy_id: int, actor: str, is_flat: bool) -> None:
    """Revoke a strategy's active allocation. Requires the strategy flat with no open orders
    (the caller computes `is_flat` from the ledger + broker)."""
    if not is_flat:
        raise AllocationError("cannot deallocate a strategy that is not flat / has open orders")
    existing = active_allocation(conn, strategy_id)
    if existing is None:
        return
    conn.execute("UPDATE strategy_allocations SET revoked_ts = ? WHERE id = ?",
                 (datetime.now(UTC).isoformat(), existing["id"]))
    conn.commit()
