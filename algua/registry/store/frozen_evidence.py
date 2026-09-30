"""Append-only persistence for frozen planner attempts (Story 1.3d §3).

The CLI binds the live frozen port's injected ``record`` callable to
:func:`record_frozen_invocation`, which writes one ``frozen_invocations`` row per attempt and
commits it in its own transaction, so a recorded attempt survives whatever the tick does next.
The schema (``algua/registry/db/frozen_evidence.py``) makes the rows immutable and checks the
Phase B and tick-link invariants; this module adds no rule of its own. SQLite errors propagate
unchanged: a recorder failure is systemic (Story 1.3c §8).
"""
from __future__ import annotations

import sqlite3
from dataclasses import fields
from typing import Any

from algua.contracts.frozen_evidence import FrozenAttempt

_COLUMNS = tuple(field.name for field in fields(FrozenAttempt))
_FLAGS = frozenset({"timed_out", "stdout_exceeded", "stderr_truncated"})
_INSERT = (
    f"INSERT INTO frozen_invocations({', '.join(_COLUMNS)})"
    f" VALUES ({', '.join('?' for _ in _COLUMNS)})"
)
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM frozen_invocations WHERE id=?"


def record_frozen_invocation(conn: sqlite3.Connection, attempt: FrozenAttempt) -> int:
    """Insert ``attempt`` as one row, commit it, and return its id.

    Refuses a connection that already has a transaction open: committing here would silently
    commit the caller's unrelated work with the evidence row.
    """
    if conn.in_transaction:
        raise RuntimeError(
            "record_frozen_invocation commits its own transaction; the connection already has"
            " an open transaction")
    values = tuple(
        int(getattr(attempt, name)) if name in _FLAGS else getattr(attempt, name)
        for name in _COLUMNS
    )
    try:
        cur = conn.execute(_INSERT, values)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    if cur.lastrowid is None:
        raise sqlite3.DatabaseError("frozen invocation insert produced no row id")
    return int(cur.lastrowid)


def frozen_invocation(conn: sqlite3.Connection, invocation_id: int) -> FrozenAttempt | None:
    """The recorded attempt with ``invocation_id``, or ``None``."""
    row = conn.execute(_SELECT, (invocation_id,)).fetchone()
    if row is None:
        return None
    values: dict[str, Any] = dict(zip(_COLUMNS, tuple(row), strict=True))
    for name in _FLAGS:
        values[name] = bool(values[name])
    return FrozenAttempt(**values)
