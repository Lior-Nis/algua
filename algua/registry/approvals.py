from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from algua.registry.store import get_strategy


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_approval(
    conn: sqlite3.Connection, name: str, code_hash: str, config_hash: str, approved_by: str
) -> int:
    rec = get_strategy(conn, name)
    cur = conn.execute(
        "INSERT INTO approvals(strategy_id, code_hash, config_hash, approved_by, created_at)"
        " VALUES (?,?,?,?,?)",
        (rec.id, code_hash, config_hash, approved_by, _now()),
    )
    conn.commit()
    return int(cur.lastrowid)


def has_valid_approval(
    conn: sqlite3.Connection, strategy_id: int, code_hash: str, config_hash: str
) -> bool:
    row = conn.execute(
        "SELECT 1 FROM approvals WHERE strategy_id=? AND code_hash=? AND config_hash=?"
        " AND revoked_at IS NULL LIMIT 1",
        (strategy_id, code_hash, config_hash),
    ).fetchone()
    return row is not None
