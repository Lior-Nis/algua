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


def test_migrate_adds_dependency_hash_column_to_legacy_approvals(tmp_path):
    """A pre-existing, populated approvals table without dependency_hash gets the column added
    in place (ALTER), with existing rows defaulting to NULL — fail-closed for the live gate."""
    conn = connect(tmp_path / "r.db")
    # Build a legacy approvals table lacking dependency_hash, with one row already in it.
    conn.executescript(
        """
        CREATE TABLE strategies (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE approvals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL,
            code_hash TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            approved_by TEXT NOT NULL,
            created_at TEXT NOT NULL,
            revoked_at TEXT
        );
        INSERT INTO strategies(id, name) VALUES (1, 's');
        INSERT INTO approvals(strategy_id, code_hash, config_hash, approved_by, created_at)
            VALUES (1, 'c', 'cfg', 'legacy', '2026-01-01T00:00:00+00:00');
        """
    )
    conn.commit()

    migrate(conn)

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(approvals)")}
    assert "dependency_hash" in cols
    legacy = conn.execute("SELECT dependency_hash FROM approvals WHERE id=1").fetchone()
    assert legacy["dependency_hash"] is None  # existing row fails closed

    migrate(conn)  # re-running the ALTER path must stay idempotent
    cols_again = {r["name"] for r in conn.execute("PRAGMA table_info(approvals)")}
    assert "dependency_hash" in cols_again


def test_migrate_adds_dependency_hash_column_to_legacy_stage_transitions(tmp_path):
    """A pre-existing, populated stage_transitions table without dependency_hash gets the column
    added in place (ALTER), with existing rows defaulting to NULL — same idempotent mechanism as
    the approvals migration."""
    conn = connect(tmp_path / "r.db")
    # Build a legacy stage_transitions table lacking dependency_hash, with one row already in it.
    conn.executescript(
        """
        CREATE TABLE strategies (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE stage_transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL,
            from_stage TEXT,
            to_stage TEXT NOT NULL,
            actor TEXT NOT NULL,
            reason TEXT,
            code_hash TEXT,
            config_hash TEXT,
            created_at TEXT NOT NULL
        );
        INSERT INTO strategies(id, name) VALUES (1, 's');
        INSERT INTO stage_transitions(strategy_id, to_stage, actor, created_at)
            VALUES (1, 'idea', 'system', '2026-01-01T00:00:00+00:00');
        """
    )
    conn.commit()

    migrate(conn)

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(stage_transitions)")}
    assert "dependency_hash" in cols
    legacy = conn.execute(
        "SELECT dependency_hash FROM stage_transitions WHERE id=1"
    ).fetchone()
    assert legacy["dependency_hash"] is None  # existing row fails closed

    migrate(conn)  # re-running the ALTER path must stay idempotent
    cols_again = {r["name"] for r in conn.execute("PRAGMA table_info(stage_transitions)")}
    assert "dependency_hash" in cols_again


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
