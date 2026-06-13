import sqlite3

import pytest

from algua.registry.db import _add_missing_columns


def test_add_missing_columns_tolerates_lost_alter_race(tmp_path):
    """A process that loses the concurrent-ALTER race holds a stale 'column absent' snapshot,
    then its own ALTER raises 'duplicate column name'. _add_missing_columns must swallow that
    and treat the column as present (idempotent), not propagate the error."""
    conn = sqlite3.connect(tmp_path / "r.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.execute("ALTER TABLE t ADD COLUMN c TEXT")  # the 'winner' already added it
    conn.commit()

    class _StaleColumnsConn:
        """Wraps a real connection but hides column ``hidden`` from PRAGMA table_info,
        reproducing the stale schema snapshot a race-loser holds."""
        def __init__(self, real, table, hidden):
            self._real, self._table, self._hidden = real, table, hidden

        def execute(self, sql, *args):
            cur = self._real.execute(sql, *args)
            if sql.startswith(f"PRAGMA table_info({self._table})"):
                return [r for r in cur.fetchall() if r["name"] != self._hidden]
            return cur

        def __getattr__(self, name):
            return getattr(self._real, name)

    stale = _StaleColumnsConn(conn, "t", "c")
    # Column 'c' really exists, but `stale` reports it absent -> _add_missing_columns will ALTER
    # and hit 'duplicate column name'. It must NOT raise.
    _add_missing_columns(stale, "t", {"c": "TEXT"})  # no exception == pass
    assert {r["name"] for r in conn.execute("PRAGMA table_info(t)")} == {"id", "c"}


def test_add_missing_columns_reraises_non_duplicate_errors(tmp_path):
    """A non-duplicate OperationalError (here: ALTER on a missing table) must propagate, so a
    future over-broad except can't silently swallow real migration failures."""
    conn = sqlite3.connect(tmp_path / "r.db")
    conn.row_factory = sqlite3.Row
    # No table 't' exists -> the ALTER raises 'no such table', which must NOT be swallowed.
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        _add_missing_columns(conn, "t", {"c": "TEXT"})
