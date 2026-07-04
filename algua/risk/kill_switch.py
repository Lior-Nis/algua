from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


def trip(conn: sqlite3.Connection, strategy: str, *, reason: str, actor: str) -> None:
    conn.execute(
        "INSERT INTO kill_switches(strategy, reason, actor, created_at) VALUES (?,?,?,?) "
        "ON CONFLICT(strategy) DO UPDATE SET "
        "reason=excluded.reason, actor=excluded.actor, created_at=excluded.created_at",
        (strategy, reason, actor, datetime.now(UTC).isoformat()),
    )
    conn.commit()


def is_tripped(conn: sqlite3.Connection, strategy: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM kill_switches WHERE strategy = ?", (strategy,)
    ).fetchone()
    return row is not None


def reset(conn: sqlite3.Connection, strategy: str) -> bool:
    cur = conn.execute("DELETE FROM kill_switches WHERE strategy = ?", (strategy,))
    conn.commit()
    return cur.rowcount > 0


def list_tripped(conn: sqlite3.Connection) -> list[str]:
    """Sorted list of strategy names with a tripped kill switch (empty if none)."""
    rows = conn.execute("SELECT strategy FROM kill_switches ORDER BY strategy").fetchall()
    return [r["strategy"] for r in rows]


def get(conn: sqlite3.Connection, strategy: str) -> dict[str, str] | None:
    row = conn.execute(
        "SELECT strategy, reason, actor, created_at FROM kill_switches WHERE strategy = ?",
        (strategy,),
    ).fetchone()
    if row is None:
        return None
    return {"strategy": row["strategy"], "reason": row["reason"],
            "actor": row["actor"], "created_at": row["created_at"]}
