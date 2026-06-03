from algua.registry.db import SCHEMA_VERSION, connect, migrate


def _tables(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r["name"] for r in rows}


def test_migrate_creates_paper_tables(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    tables = _tables(conn)
    assert {"paper_orders", "paper_fills", "audit_log", "strategy_peaks"} <= tables


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    migrate(conn)  # second run must not raise
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION


def test_migrate_creates_kill_switches_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert "kill_switches" in _tables(conn)
