import sqlite3

import pytest

from algua.registry.db import SCHEMA_VERSION, _add_missing_columns, connect, migrate


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


def test_fresh_db_has_committed_at_column(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(holdout_evaluations)")}
    assert "committed_at" in cols
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_v21_db_gains_committed_at_on_migrate(tmp_path):
    """A holdout_evaluations table created WITHOUT committed_at (pre-v22) gains it via migrate,
    legacy rows keep committed_at NULL, and user_version stamps to the new version."""
    conn = connect(tmp_path / "r.db")
    conn.execute(
        "CREATE TABLE holdout_evaluations ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_id INTEGER NOT NULL,"
        " data_source TEXT NOT NULL, snapshot_id TEXT, period_start TEXT NOT NULL,"
        " period_end TEXT NOT NULL, holdout_frac REAL NOT NULL, config_hash TEXT NOT NULL,"
        " reused INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)"
    )
    conn.execute("PRAGMA user_version=21")
    conn.execute(
        "INSERT INTO holdout_evaluations"
        "(strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,"
        " config_hash, reused, created_at) VALUES (1,'demo',NULL,'2022-01-01','2022-12-31',"
        " 0.2,'abc',0,'2022-01-01')"
    )
    conn.commit()
    migrate(conn)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(holdout_evaluations)")}
    assert "committed_at" in cols
    row = conn.execute("SELECT committed_at FROM holdout_evaluations").fetchone()
    assert row["committed_at"] is None
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_pre_v23_holdout_rows_backfill_to_full_period(tmp_path):
    """A holdout_evaluations row created WITHOUT holdout_start/holdout_end (pre-v23) gains them via
    migrate, backfilled to the conservative full period [period_start, period_end]."""
    from algua.registry.db import connect, migrate

    db = tmp_path / "r.db"
    conn = connect(db)
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS strategies (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT);"
        "INSERT INTO strategies(id, name) VALUES (1, 's');"
        "CREATE TABLE holdout_evaluations ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_id INTEGER NOT NULL,"
        " data_source TEXT NOT NULL, snapshot_id TEXT, period_start TEXT NOT NULL,"
        " period_end TEXT NOT NULL, holdout_frac REAL NOT NULL, config_hash TEXT NOT NULL,"
        " reused INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, committed_at TEXT);"
    )
    conn.execute(
        "INSERT INTO holdout_evaluations"
        "(strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,"
        " config_hash, reused, created_at, committed_at)"
        " VALUES (1,'demo',NULL,'2022-01-01','2023-12-31',0.2,'h',0,'2022-01-01T00:00:00+00:00',"
        " '2022-02-01T00:00:00+00:00')",
    )
    conn.commit()

    migrate(conn)

    cols = {row["name"] for row in conn.execute("PRAGMA table_info(holdout_evaluations)")}
    assert {"holdout_start", "holdout_end"} <= cols
    row = conn.execute(
        "SELECT holdout_start, holdout_end FROM holdout_evaluations"
    ).fetchone()
    assert (row["holdout_start"], row["holdout_end"]) == ("2022-01-01", "2023-12-31")
    n_null = conn.execute(
        "SELECT COUNT(*) AS c FROM holdout_evaluations"
        " WHERE holdout_start IS NULL OR holdout_end IS NULL"
    ).fetchone()["c"]
    assert n_null == 0
    migrate(conn)  # idempotent
    conn.close()
