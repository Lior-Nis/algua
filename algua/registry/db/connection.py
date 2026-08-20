"""``connect()`` -- the sqlite handle every registry caller opens, and its pragma posture.

Deliberately does NOT call ``migrate()``: callers pair the two (see ``algua/cli/_common.py``).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000;")  # WAL + busy_timeout = deliberate concurrency posture
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    # recursive_triggers is OFF by default, and with it OFF the implicit row-delete SQLite performs
    # to resolve an `INSERT/REPLACE ... OR REPLACE` conflict does NOT fire BEFORE DELETE triggers.
    # The v37 append-only triggers on families/family_members/family_parents/family_events/
    # backtest_returns (#524) would therefore be silently bypassed by a REPLACE — an in-place row
    # rewrite masquerading as an append. Turn it ON so the append-only invariant also covers the
    # implicit-delete path. NB: this is a PER-CONNECTION pragma, not schema-resident: a raw
    # sqlite3.connect() that skips this helper does NOT inherit it (see the narrowed trigger comment
    # in family.py). Explicit DELETE/UPDATE stay aborted through ANY connection regardless.
    conn.execute("PRAGMA recursive_triggers=ON;")
    return conn
