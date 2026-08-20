"""Holdout context: ``holdout_evaluations`` and ``holdout_returns``.

The single-use out-of-sample burn ledger and the one OOS return vector persisted per burn
(operated by ``algua/registry/store/holdout.py``), plus the v23 interval backfill.
"""
from __future__ import annotations

import sqlite3

SCHEMA = """
-- holdout_evaluations burns a walk-forward holdout window on use, so it can be evaluated ONCE.
-- `research promote` carves the last holdout_frac of the period into an out-of-sample holdout and
-- gates on it; the promotion guarantee rests on that holdout being seen once. Each row records a
-- holdout that was looked at (regardless of gate pass/fail — looking consumes it).
-- Single-use key: (strategy_id, OOS interval [holdout_start, holdout_end]) — PROVENANCE-INDEPENDENT
-- (#205). data_source/snapshot_id are recorded as EVIDENCE only, never matched on. A row is REFUSED
-- if its OOS interval overlaps a prior reservation/burn for the strategy (the exact bars, #192),
-- regardless of how the bars were reached, unless the operator passes --allow-holdout-reuse (writes
-- reused=1, auditable). A NULL interval matches unconditionally (fail closed). period_* and
-- holdout_frac are recorded as evidence only; matching is on the INTERVAL, not on config_hash
-- (re-gating the same OOS window with a tweaked config is exactly the leak being closed). FK into
-- strategies(id) — relational state, not an audit snapshot.
CREATE TABLE IF NOT EXISTS holdout_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    data_source TEXT NOT NULL,
    snapshot_id TEXT,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    holdout_frac REAL NOT NULL,
    config_hash TEXT NOT NULL,   -- '' while in-flight (placeholder); real hash written at finalize.
    reused INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    committed_at TEXT,           -- NULL = in-flight reservation (or a legacy burn predating this
                                 -- column); non-NULL = committed burn. Either way an overlapping
                                 -- row blocks fail-closed. Orphaned reservations (pending rows from
                                 -- a crashed run) are listable via WHERE committed_at IS NULL and
                                 -- are cleared only by a deliberate human --allow-holdout-reuse.
    holdout_start TEXT,          -- ISO date; OOS tail start (the matched single-use window, #192)
    holdout_end TEXT             -- ISO date; OOS tail end (last actual bar date)
);
CREATE INDEX IF NOT EXISTS ix_holdout_evaluations_strategy
    ON holdout_evaluations(strategy_id);
-- holdout_returns persists EXACTLY ONE out-of-sample per-period return vector per holdout burn
-- (#221 Slice 1) — the heavy shared prerequisite for Phase-3 Slices 2/3/4 (bootstrap, N_eff,
-- multi-regime). Grain is per-strategy-holdout, NOT per-combo: persisting per-combo vectors would
-- re-open the single-use best-of-N surface sweep() is built to prevent. The FK ties each vector to
-- the burn that produced it; UNIQUE(holdout_evaluation_id) prevents double-writes and makes a
-- reconciliation job (re-running the deterministic walk-forward) safe. SENSITIVE: no CLI
-- accessor and no "get my own vector" API may read returns_blob — sibling-only cross-strategy.
CREATE TABLE IF NOT EXISTS holdout_returns (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    holdout_evaluation_id INTEGER NOT NULL REFERENCES holdout_evaluations(id),
    strategy_id           INTEGER NOT NULL REFERENCES strategies(id),
    holdout_start         TEXT    NOT NULL,   -- OOS interval identity (mirrors #192 / #205)
    holdout_end           TEXT    NOT NULL,
    n_bars                INTEGER NOT NULL,   -- length of stored vector; == holdout_metrics n_bars
    returns_blob          BLOB    NOT NULL,   -- float64 per-period OOS returns, np.tobytes()
    bar_dates_blob        BLOB    NOT NULL,   -- ISO-8601 bar dates, UTF-8 newline-delimited
    created_at            TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_holdout_returns_eval ON holdout_returns(holdout_evaluation_id);
CREATE INDEX IF NOT EXISTS ix_holdout_returns_strategy  ON holdout_returns(strategy_id);
CREATE INDEX IF NOT EXISTS ix_holdout_returns_interval
    ON holdout_returns(holdout_start, holdout_end);
"""


def _backfill_holdout_intervals(conn: sqlite3.Connection) -> None:
    """Backfill v23 holdout_start/holdout_end on legacy rows to the CONSERVATIVE full period
    [period_start, period_end]. The exact OOS tail cannot be recomputed at migration time (no data
    provider here), and the full period is a guaranteed superset of any real tail -> fail closed
    (may over-block a new run overlapping a legacy burn's period, the acceptable direction). Only
    touches rows missing an interval, so a row written by the new reserve path (interval already
    set) is never overwritten; deterministic, so concurrent/repeat runs converge. Idempotent.

    MUST run after migrate()'s holdout_evaluations ALTER (it reads holdout_start); see the
    ordering guard in tests/test_registry_db.py."""
    conn.execute(
        "UPDATE holdout_evaluations SET holdout_start = period_start, holdout_end = period_end"
        " WHERE holdout_start IS NULL OR holdout_end IS NULL"
    )
    leftover = conn.execute(
        "SELECT COUNT(*) AS c FROM holdout_evaluations"
        " WHERE holdout_start IS NULL OR holdout_end IS NULL"
    ).fetchone()["c"]
    if leftover:
        raise RuntimeError(
            f"holdout interval backfill left {leftover} NULL-interval row(s); refusing to stamp v23"
        )
