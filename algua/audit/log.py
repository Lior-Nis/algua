from __future__ import annotations

import sqlite3
from datetime import UTC, datetime


def append(
    conn: sqlite3.Connection, *, actor: str, action: str, reason: str | None = None,
    strategy: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO audit_log(ts, actor, action, reason, strategy) VALUES (?,?,?,?,?)",
        (datetime.now(UTC).isoformat(), actor, action, reason, strategy),
    )
    conn.commit()


def read(conn: sqlite3.Connection, *, strategy: str | None = None) -> list[sqlite3.Row]:
    if strategy is None:
        return conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
    return conn.execute(
        "SELECT * FROM audit_log WHERE strategy = ? ORDER BY id", (strategy,)
    ).fetchall()
