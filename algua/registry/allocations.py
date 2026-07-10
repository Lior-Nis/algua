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
    Σ(active capital across all strategies) ≤ account_equity. Shared by ``allocate_in_lane`` and
    the atomic ``intake_candidate_to_paper`` (#317) so the two can't drift on the capital check."""
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


def allocate_in_lane(conn: sqlite3.Connection, strategy_id: int, capital: float, actor: str,
                     account_equity: float, allowed_stages: frozenset[str],
                     max_concurrent: int | None = None) -> None:
    """Lane-scoped transactional allocator: under ONE top-level ``BEGIN IMMEDIATE`` write lock,
    re-read the strategy's CURRENT stage, refuse the allocation unless that stage is in
    ``allowed_stages``, optionally enforce the paper-lane count cap for a count-INCREASING
    allocation, then delegate to the shared ``allocate_locked`` (Σ ≤ equity) and commit.

    Reading the stage UNDER the write lock is what closes the go-live TOCTOU: a strategy that leaves
    the allowed lane (e.g. ``live -> dormant``) between the caller's friendly early check and this
    write can never be allocated. ``allowed_stages`` is a frozenset of ``Stage.value`` strings (NOT
    ``Stage`` members) so this module stays free of a ``Stage()`` parse of an unexpected DB string
    and of the ``registry -> lifecycle`` import edge.

    A resize of an already-allocated tenant is EXEMPT from the count cap (it admits no new tenant);
    only a count-increasing allocation (no active row yet) is cap-checked. TOP-LEVEL ONLY (mirrors
    ``intake_candidate_to_paper``): a manual ``BEGIN`` inside an open transaction raises, and the
    blanket ``BaseException`` rollback must own the whole txn."""
    if conn.in_transaction:
        raise RuntimeError(
            "allocate_in_lane must run at top level, not inside an open transaction")
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT stage FROM strategies WHERE id = ?", (strategy_id,)).fetchone()
        if row is None:
            raise AllocationError(f"strategy {strategy_id} not found")
        if row["stage"] not in allowed_stages:
            raise AllocationError(
                f"cannot allocate to strategy {strategy_id} at stage {row['stage']!r}; "
                f"allowed {sorted(allowed_stages)}")
        if max_concurrent is not None and active_allocation(conn, strategy_id) is None:
            n = active_paper_lane_count(conn)
            if n >= max_concurrent:
                raise CountCapReached(
                    f"paper book at capacity ({n}/{max_concurrent} active tenants)")
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
