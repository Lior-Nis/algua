from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from algua.registry.store import get_strategy


def _now() -> str:
    return datetime.now(UTC).isoformat()


def record_approval(
    conn: sqlite3.Connection, name: str, code_hash: str, config_hash: str, approved_by: str
) -> int:
    _require_non_empty("code_hash", code_hash)
    _require_non_empty("config_hash", config_hash)
    _require_non_empty("approved_by", approved_by)
    rec = get_strategy(conn, name)
    cur = conn.execute(
        "INSERT INTO approvals(strategy_id, code_hash, config_hash, approved_by, created_at)"
        " VALUES (?,?,?,?,?)",
        (rec.id, code_hash, config_hash, approved_by, _now()),
    )
    conn.commit()
    rowid = cur.lastrowid
    assert rowid is not None  # a successful INSERT always sets lastrowid
    return rowid


def has_valid_approval(
    conn: sqlite3.Connection, strategy_id: int, code_hash: str, config_hash: str
) -> bool:
    row = conn.execute(
        "SELECT 1 FROM approvals WHERE strategy_id=? AND code_hash=? AND config_hash=?"
        " AND revoked_at IS NULL LIMIT 1",
        (strategy_id, code_hash, config_hash),
    ).fetchone()
    return row is not None


def _require_non_empty(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
