"""Guarded per-tick provenance persistence."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from algua.registry.deployment import DeploymentError

_VALID_LANES = frozenset({"paper", "live"})
_VALID_CLOCK_SOURCES = frozenset({"broker", "local"})


def record_tick_snapshot(  # noqa: PLR0913
    conn: sqlite3.Connection, strategy: str, *, tick_ts: str, decision_ts: str | None,
    equity: float, peak_equity: float | None, positions: dict[str, float], n_submitted: int,
    reconcile_ok: bool, lane: str, strategy_id: int, code_hash: str, config_hash: str,
    dependency_hash: str | None, account_id: str, cash: float, clock_source: str,
    snapshot_id: str | None = None, deployment_id: int | None = None,
) -> None:
    """Append a snapshot only when its deployment provenance is currently valid."""
    if lane not in _VALID_LANES:
        raise ValueError(f"lane must be one of {sorted(_VALID_LANES)!r}, got {lane!r}")
    if clock_source not in _VALID_CLOCK_SOURCES:
        raise ValueError(
            f"clock_source must be one of {sorted(_VALID_CLOCK_SOURCES)!r}, got {clock_source!r}"
        )
    values = (
        strategy, tick_ts, decision_ts, equity, peak_equity, json.dumps(positions), n_submitted,
        1 if reconcile_ok else 0, lane, strategy_id, code_hash, config_hash, dependency_hash,
        account_id, cash, clock_source, datetime.now(UTC).isoformat(), snapshot_id,
    )
    columns = (
        "strategy, tick_ts, decision_ts, equity, peak_equity, positions, n_submitted,"
        " reconcile_ok, lane, strategy_id, code_hash, config_hash, dependency_hash, account_id,"
        " cash, clock_source, recorded_at, snapshot_id, deployment_id"
    )
    if deployment_id is None:
        legacy = conn.execute(
            "SELECT 1 FROM legacy_deployment_strategies l"
            " JOIN strategies s ON s.id=l.strategy_id"
            " WHERE l.strategy_id=? AND s.name=?"
            " AND ((?='paper' AND s.stage IN ('paper','forward_tested'))"
            " OR (?='live' AND s.stage='live'))",
            (strategy_id, strategy, lane, lane),
        ).fetchone()
        if legacy is None:
            raise DeploymentError(
                "NULL deployment tick is permitted only for the fixed legacy cohort")
        conn.execute(
            f"INSERT INTO tick_snapshots({columns})"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
            values,
        )
    else:
        cur = conn.execute(
            f"INSERT INTO tick_snapshots({columns})"
            " SELECT ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, d.id"
            " FROM strategy_deployments d"
            " JOIN deployment_artifacts a ON a.id=d.artifact_id"
            " JOIN strategies s ON s.id=d.strategy_id"
            " WHERE d.id=? AND d.strategy_id=? AND d.retired_at IS NULL AND s.name=?"
            " AND a.code_hash=? AND a.config_hash=? AND a.dependency_hash=?"
            " AND ((?='paper' AND s.stage IN ('paper','forward_tested'))"
            " OR (?='live' AND s.stage='live'))",
            (*values, deployment_id, strategy_id, strategy, code_hash, config_hash,
             dependency_hash, lane, lane),
        )
        if cur.rowcount != 1:
            raise DeploymentError(
                "tick does not match the strategy's active deployment and artifact identity")
    conn.commit()


def latest_tick_snapshot(conn: sqlite3.Connection, strategy: str) -> dict | None:
    """The most recent tick snapshot for a strategy, or None."""
    row = conn.execute(
        "SELECT tick_ts, decision_ts, equity, peak_equity, positions, n_submitted, reconcile_ok,"
        " lane, strategy_id, code_hash, config_hash, dependency_hash, account_id, cash,"
        " clock_source, recorded_at, snapshot_id, deployment_id"
        " FROM tick_snapshots WHERE strategy = ? ORDER BY id DESC LIMIT 1", (strategy,),
    ).fetchone()
    if row is None:
        return None
    return {
        "tick_ts": row["tick_ts"], "decision_ts": row["decision_ts"], "equity": row["equity"],
        "peak_equity": row["peak_equity"], "positions": json.loads(row["positions"]),
        "n_submitted": row["n_submitted"], "reconcile_ok": bool(row["reconcile_ok"]),
        "lane": row["lane"], "strategy_id": row["strategy_id"],
        "code_hash": row["code_hash"], "config_hash": row["config_hash"],
        "dependency_hash": row["dependency_hash"], "account_id": row["account_id"],
        "cash": row["cash"], "clock_source": row["clock_source"],
        "recorded_at": row["recorded_at"], "snapshot_id": row["snapshot_id"],
        "deployment_id": row["deployment_id"],
    }
