from algua.registry import store
from algua.registry.db import connect, migrate


def _tables(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r["name"] for r in rows}


def test_migrate_creates_paper_tables(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    tables = _tables(conn)
    assert {"paper_orders", "paper_fills", "audit_log"} <= tables


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    migrate(conn)  # second run must not raise
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == 4


def test_migrate_creates_kill_switches_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert "kill_switches" in _tables(conn)


def test_migrate_creates_live_equity_peak_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert "live_equity_peak" in _tables(conn)


def test_equity_peak_roundtrip_and_upsert(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    assert store.get_equity_peak(conn, "s") is None      # no row yet
    store.set_equity_peak(conn, "s", 100.0)
    assert store.get_equity_peak(conn, "s") == 100.0
    store.set_equity_peak(conn, "s", 125.5)              # UPSERT overwrites
    assert store.get_equity_peak(conn, "s") == 125.5


def test_clear_equity_peak(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    store.set_equity_peak(conn, "s", 100.0)
    store.clear_equity_peak(conn, "s")
    assert store.get_equity_peak(conn, "s") is None
