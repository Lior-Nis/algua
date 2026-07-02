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


def read(
    conn: sqlite3.Connection,
    *,
    strategy: str | None = None,
    actor: str | None = None,
    action: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[sqlite3.Row]:
    """Return audit rows, most-recent-first.

    Args:
        conn: open SQLite connection.
        strategy: when given, filter to rows for that strategy only.
        actor: when given, filter to rows for that actor only.
        action: when given, filter to rows for that action only.
        since: inclusive lower bound on ``ts`` (``ts >= since``). Must be a
            canonical UTC ISO-8601 timestamp — the same format ``append`` writes
            (``datetime.now(UTC).isoformat()``) — so the text comparison is
            chronologically correct; callers normalize before passing.
        until: exclusive upper bound on ``ts`` (``ts < until``); same format
            requirement as *since*.
        limit: maximum rows to return.  ``None`` (default) returns all rows.
        offset: number of rows to skip before returning results; used for
            pagination together with *limit*.
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit!r}")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset!r}")
    # Each fragment is a hardcoded "column op ?" string; only the VALUES are
    # bound. No user input is ever interpolated into the SQL text.
    clauses: list[str] = []
    params: list[object] = []
    for fragment, value in (
        ("strategy = ?", strategy),
        ("actor = ?", actor),
        ("action = ?", action),
        ("ts >= ?", since),
        ("ts < ?", until),
    ):
        if value is not None:
            clauses.append(fragment)
            params.append(value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"SELECT * FROM audit_log {where} ORDER BY id DESC"
    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params += [limit, offset]
    elif offset:
        # offset without limit requires SQLite's -1 sentinel for unlimited
        query += " LIMIT -1 OFFSET ?"
        params.append(offset)
    return conn.execute(query, params).fetchall()
