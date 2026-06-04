from algua.registry.db import SCHEMA_VERSION, connect, migrate


def test_migrate_creates_tables_and_sets_version(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"strategies", "stage_transitions", "approvals"} <= tables
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION


def test_migrate_is_idempotent(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    migrate(conn)  # second run must not raise
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION


def test_wal_mode_enabled(tmp_path):
    conn = connect(tmp_path / "r.db")
    mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert mode.lower() == "wal"


def test_bootstrap_runs_even_when_version_already_current(tmp_path):
    """migrate() is an idempotent bootstrap, not a version-gated migrator.

    A stale/pre-stamped user_version must NOT cause the schema to be skipped:
    the CREATE TABLE IF NOT EXISTS script always runs and brings the DB up to
    the full current schema regardless of the recorded version.
    """
    conn = connect(tmp_path / "r.db")
    # Pre-stamp the version without creating any tables, simulating a DB whose
    # recorded version says "current" but whose schema is empty.
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION};")
    conn.commit()
    migrate(conn)
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"strategies", "stage_transitions", "approvals"} <= tables
