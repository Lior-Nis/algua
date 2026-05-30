from algua.registry.db import connect, migrate


def test_migrate_creates_tables_and_sets_version(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"strategies", "stage_transitions", "approvals"} <= tables
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == 1


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    migrate(conn)  # second run must not raise
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == 1


def test_wal_mode_enabled(tmp_path):
    conn = connect(tmp_path / "r.db")
    mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert mode.lower() == "wal"
