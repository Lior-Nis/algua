"""Per-strategy live capital allocations (the fixed sizing denominator). Append-only lifecycle:
the active allocation is the newest non-revoked row for a strategy. Σ(active capital) is capped at
account equity so the book can never over-commit the shared account."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


class AllocationError(ValueError):
    """An allocation request that would over-commit the account, or a deallocation of a
    non-flat strategy."""


class CountCapReached(AllocationError):
    """An allocation that would admit one more active paper-lane tenant than the operator's
    max-concurrent cap allows. Distinct from the capital bound so intake can tell them apart."""


def total_allocated(conn: sqlite3.Connection) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(capital), 0.0) AS t FROM strategy_allocations WHERE revoked_ts IS NULL"
    ).fetchone()
    return float(row["t"])


def active_paper_lane_count(conn: sqlite3.Connection) -> int:
    """Number of active paper-book tenants: strategies at stage ∈ {paper, forward_tested} that
    hold a non-revoked allocation. This is the concurrency governor intake (#317) re-reads UNDER
    the write lock so the count cap can never be raced past."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM strategies s "
        "WHERE s.stage IN ('paper', 'forward_tested') "
        "AND EXISTS (SELECT 1 FROM strategy_allocations a "
        "WHERE a.strategy_id = s.id AND a.revoked_ts IS NULL)"
    ).fetchone()
    return int(row["n"])


def active_allocation(conn: sqlite3.Connection, strategy_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM strategy_allocations WHERE strategy_id = ? AND revoked_ts IS NULL "
        "ORDER BY id DESC LIMIT 1",
        (strategy_id,),
    ).fetchone()


def allocate_locked(conn: sqlite3.Connection, strategy_id: int, capital: float, actor: str,
                    account_equity: float) -> None:
    """The Σ read-check-revoke-insert body WITHOUT opening a transaction — the caller owns the
    ``BEGIN IMMEDIATE`` scope. Revokes any prior active allocation (re-allocation) then enforces
    Σ(active capital across all strategies) ≤ account_equity. Shared by ``allocate`` and the atomic
    ``intake_candidate_to_paper`` (#317) so the two paths can never drift on the capital check."""
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


def allocate(conn: sqlite3.Connection, strategy_id: int, capital: float, actor: str,
             account_equity: float) -> None:
    """Set a strategy's live capital base. Revokes any prior active allocation (re-allocation),
    then enforces Σ(active capital across all strategies) ≤ account_equity. A re-allocation resets
    the strategy's NAV drawdown peak (the operator's deliberate capital change)."""
    # BEGIN IMMEDIATE takes the write lock up front so the read-check-write (Σ ≤ equity) is atomic:
    # two concurrent allocate calls can't both read the same total and both slip past the cap.
    try:
        conn.execute("BEGIN IMMEDIATE")
        allocate_locked(conn, strategy_id, capital, actor, account_equity)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def revoke_active_locked(conn: sqlite3.Connection, strategy_id: int) -> None:
    """Revoke a strategy's active allocation WITHOUT committing — the caller owns the transaction
    (e.g. the stage-change txn in `_apply_transition_locked`). No-op if nothing is active."""
    existing = active_allocation(conn, strategy_id)
    if existing is None:
        return
    conn.execute("UPDATE strategy_allocations SET revoked_ts = ? WHERE id = ?",
                 (datetime.now(UTC).isoformat(), existing["id"]))


def deallocate(conn: sqlite3.Connection, strategy_id: int, actor: str, is_flat: bool) -> None:
    """Revoke a strategy's active allocation. Requires the strategy flat with no open orders
    (the caller computes `is_flat` from the ledger + broker)."""
    if not is_flat:
        raise AllocationError("cannot deallocate a strategy that is not flat / has open orders")
    revoke_active_locked(conn, strategy_id)
    conn.commit()
