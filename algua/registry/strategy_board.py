"""The research strategy board projection (`algua runs board`).

Exactly one row per REGISTERED strategy identity, each joined to that identity's latest gate run.
A pure read: no writes, no locks, no subprocess — it takes the ``SqliteStrategyRepository`` alone
and reaches sqlite through ``repo.connection``, like ``run_views``.

Display surface only. Nothing here may feed a promotion decision, family classification or any
other selection path (the caveat in the run-tracking design, §5): the board orders by lifecycle
and never ranks, scores or recommends.

Contract: ``docs/development/specs/spec-strategy-board/strategy-board-contract.md``.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime
from typing import Any

from algua.contracts.lifecycle import Stage
from algua.registry.store import SqliteStrategyRepository

__all__ = ["BOARD_METRICS", "SCHEMA_VERSION", "strategy_board_payload"]

SCHEMA_VERSION = "strategy-board/v1"

#: The gate-run scalars the board exposes — a subset of the fixed run vocabulary, selected from
#: the run row as stored. Nothing is derived, imputed or normalized.
BOARD_METRICS: tuple[str, ...] = (
    "sharpe_oos", "total_return_oos", "max_drawdown_oos", "n_obs_oos",
    "mean_window_sharpe", "min_window_sharpe", "pct_positive_windows",
)

_INT_METRICS = frozenset({"n_obs_oos"})
_LIFECYCLE_ORDER = {stage.value: i for i, stage in enumerate(Stage)}

# The latest gate run is chosen per strategy IDENTITY (`runs.strategy_id = strategies.id`), never
# by reusable name, and by recency (`created_at DESC, id DESC`), never by metric.
_BOARD_SQL = f"""
SELECT s.id AS strategy_id, s.name, s.family, s.stage,
       r.id AS run_id, r.created_at, r.passed, {", ".join(f"r.{m}" for m in BOARD_METRICS)}
FROM strategies s
LEFT JOIN runs r ON r.id = (
    SELECT g.id FROM runs g
    WHERE g.kind = 'gate' AND g.strategy_id = s.id
    ORDER BY g.created_at DESC, g.id DESC
    LIMIT 1)
"""


def _scalar(value: Any, *, integer: bool) -> float | int | None:
    """A stored metric as a finite number, or None. SQLite is dynamically typed, so a REAL column
    can hold text or infinity; malformed evidence stays unavailable rather than becoming zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    if integer:
        return int(value) if float(value).is_integer() else None
    return float(value)


def _passed(value: Any) -> bool | None:
    if isinstance(value, bool) or value not in (0, 1):
        return None
    return bool(value)


def _valid_created_at(value: Any) -> str | None:
    """Return an offset-aware ISO timestamp, or ``None`` for unusable recency evidence."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return value if parsed.tzinfo is not None else None


def _gate_entry(row: sqlite3.Row) -> dict[str, Any] | None:
    if row["run_id"] is None:
        return None
    created_at = _valid_created_at(row["created_at"])
    if created_at is None:
        return None
    entry: dict[str, Any] = {
        "run_id": row["run_id"],
        "created_at": created_at,
        "passed": _passed(row["passed"]),
    }
    for metric in BOARD_METRICS:
        value = _scalar(row[metric], integer=metric in _INT_METRICS)
        entry[metric] = None if metric == "n_obs_oos" and value is not None and value < 0 else value
    return entry


def strategy_board_payload(repo: SqliteStrategyRepository) -> dict[str, Any]:
    """`runs board` — every registered strategy once, in lifecycle order, with its latest gate run
    (or ``None``). An empty registry is a successful empty payload."""
    rows = repo.connection.execute(_BOARD_SQL).fetchall()
    rows.sort(key=lambda r: (
        _LIFECYCLE_ORDER.get(r["stage"], len(_LIFECYCLE_ORDER)),
        str(r["name"]).lower(),
        r["strategy_id"],
    ))
    board = [
        {
            "strategy_id": r["strategy_id"],
            "name": r["name"],
            "family": r["family"],
            "stage": r["stage"],
            "latest_gate": _gate_entry(r),
        }
        for r in rows
    ]
    return {"schema_version": SCHEMA_VERSION, "count": len(board), "rows": board}
