from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


def engage(conn: sqlite3.Connection, *, reason: str, actor: str) -> None:
    """Engage the account-wide halt (single row id=1).

    Idempotent: re-engaging updates reason/actor.
    """
    conn.execute(
        "INSERT INTO global_halt(id, reason, actor, created_at) VALUES (1,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET "
        "reason=excluded.reason, actor=excluded.actor, created_at=excluded.created_at",
        (reason, actor, datetime.now(UTC).isoformat()),
    )
    conn.commit()


def is_engaged(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM global_halt WHERE id = 1").fetchone() is not None


def clear(conn: sqlite3.Connection) -> bool:
    """Clear the halt. Returns whether a row was actually removed."""
    cur = conn.execute("DELETE FROM global_halt")
    conn.commit()
    return cur.rowcount > 0


def get(conn: sqlite3.Connection) -> dict[str, str] | None:
    row = conn.execute(
        "SELECT reason, actor, created_at FROM global_halt WHERE id = 1"
    ).fetchone()
    if row is None:
        return None
    return {"reason": row["reason"], "actor": row["actor"], "created_at": row["created_at"]}
