"""Book-level capital rollup (`algua book status`) — who holds a slice, and who should but doesn't.

Allocation had no read-only surface at all: ``paper allocate`` / ``live allocate`` write it,
``fleet status`` never reported it, and the only equity read (``paper account``) calls the broker,
which the monitor forbids. So a strategy could sit in an operational stage holding NO slice, be
skipped by the operator loop forever, and surface only as an unexplained ``idle``.

That is not hypothetical. On 2026-08-15 ``liquidity_stable_quality_momentum`` was exactly this:
``paper -> dormant`` atomically releases the slice, ``dormant -> paper`` restores the STAGE but not
the capital, and nothing re-allocates on the way back. It had never ticked and never would.
:func:`book_status` generalises that into a standing condition rather than a one-off.

Pure reads: SELECTs against the registry DB plus persisted tick snapshots. NO broker call, no
writes, no locks — the same discipline as ``fleet_health``. Lives in ``algua.execution`` (not a cli
module) because ``execution -> registry`` is permitted by the import-linter contracts while a
``cli -> cli`` sibling import is not.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from algua.contracts.lifecycle import Stage
from algua.execution.fleet_health import OPERATIONAL_STAGES
from algua.execution.order_state import latest_tick_snapshot
from algua.registry.allocations import total_allocated

__all__ = ["book_status"]


def _safe_last_equity(conn: sqlite3.Connection, name: str) -> tuple[float | None, str | None]:
    """Newest tick equity, or ``(None, error)``.

    ``latest_tick_snapshot`` eagerly parses the persisted positions JSON, so ONE corrupt row would
    otherwise crash the whole book view and hide every other slice — the same fail-soft boundary
    ``fleet_health._safe_latest_tick`` draws.
    """
    try:
        row = latest_tick_snapshot(conn, name)
    except Exception as exc:  # noqa: BLE001 — a corrupt row degrades one slice, never the book
        return None, f"{type(exc).__name__}: {exc}"
    if row is None:
        return None, None
    equity = row["equity"]
    return (float(equity) if equity is not None else None), None


def book_status(
    conn: sqlite3.Connection, *, capacity: int, operational_stages: frozenset[str] | None = None
) -> dict[str, Any]:
    """Every active allocation, plus every operational strategy MISSING one.

    ``capacity`` is the operator's max-concurrent book cap (``settings.paper_book_capacity``).

    Note on headroom: only the COUNT headroom (``capacity - allocated``) is reported. Capital
    headroom would require the account equity, and the only path to that is a broker call — which
    this view is explicitly forbidden to make. Reporting a capital headroom derived from
    tick-snapshot equity would be inventing a number that is not the account's, so it is omitted
    rather than approximated.
    """
    stages = OPERATIONAL_STAGES if operational_stages is None else operational_stages

    rows = conn.execute(
        "SELECT s.name AS name, s.stage AS stage, a.capital AS capital, "
        "       a.effective_ts AS effective_ts, a.actor AS actor "
        "FROM strategies s "
        "JOIN strategy_allocations a ON a.strategy_id = s.id AND a.revoked_ts IS NULL "
        "ORDER BY a.capital DESC, s.name"
    ).fetchall()

    slices: list[dict[str, Any]] = []
    for row in rows:
        equity, error = _safe_last_equity(conn, row["name"])
        slices.append({
            "strategy": row["name"],
            "stage": row["stage"],
            "capital": float(row["capital"]),
            "last_equity": equity,
            "effective_ts": row["effective_ts"],
            "actor": row["actor"],
            "equity_error": error,
        })

    # Operational stage + no active allocation = a strategy the operator loop will skip forever.
    # Ordered oldest-first by the transition that put it in this stage, so the longest-stranded
    # strategy leads — that is the one bleeding the most opportunity.
    placeholders = ",".join("?" for _ in stages)
    stranded = conn.execute(
        f"SELECT s.name AS name, s.stage AS stage, "  # noqa: S608 — placeholders are ?-bound
        "       (SELECT MAX(t.created_at) FROM stage_transitions t "
        "        WHERE t.strategy_id = s.id AND t.to_stage = s.stage) AS since "
        f"FROM strategies s WHERE s.stage IN ({placeholders}) "
        "AND NOT EXISTS (SELECT 1 FROM strategy_allocations a "
        "                WHERE a.strategy_id = s.id AND a.revoked_ts IS NULL) "
        "ORDER BY since, s.name",
        tuple(sorted(stages)),
    ).fetchall()

    unallocated: list[dict[str, Any]] = []
    for row in stranded:
        has_ticked, _ = _safe_last_equity(conn, row["name"])
        unallocated.append({
            "strategy": row["name"],
            "stage": row["stage"],
            "since": row["since"],
            "ever_ticked": has_ticked is not None,
        })

    allocated = len(slices)
    return {
        "ok": not unallocated,
        "capacity": capacity,
        "allocated": allocated,
        "count_headroom": max(capacity - allocated, 0),
        "sum_allocations": total_allocated(conn),
        "unallocated_operational": unallocated,
        "slices": slices,
        "operational_stages": sorted(stages),
        "live_allocated": sum(1 for s in slices if s["stage"] == Stage.LIVE.value),
    }
