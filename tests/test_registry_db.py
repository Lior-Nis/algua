import sqlite3

from algua.registry.db import SCHEMA_VERSION, connect, migrate

_META_COLS = {"family", "tags", "author", "hypothesis_status", "derived_from", "description"}


def test_schema_version_is_21():
    assert SCHEMA_VERSION == 21


def test_v21_adds_tick_provenance_and_forward_gate_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(tick_snapshots)")}
    assert {"lane", "code_hash", "config_hash", "dependency_hash", "strategy_id",
            "account_id", "cash", "clock_source", "recorded_at"} <= cols
    ocols = {r["name"] for r in conn.execute("PRAGMA table_info(paper_orders)")}
    assert "strategy_id" in ocols
    fcols = {r["name"] for r in conn.execute("PRAGMA table_info(forward_gate_evaluations)")}
    assert {"strategy_id", "passed", "realized_sharpe", "holdout_sharpe", "first_tick_id",
            "last_tick_id", "n_concurrent_forward", "consumed", "created_at"} <= fcols
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 21


def test_v21_new_columns_are_nullable_on_existing_rows(tmp_path):
    """New tick_snapshots / paper_orders columns are nullable: pre-v21 rows survive and read back
    with NULL for all new fields (legacy rows are fail-closed — NULL is inadmissible gate evidence;
    no backfill is intended)."""
    db = tmp_path / "r.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    # Create the pre-v21 tick_snapshots table shape (only the original columns).
    conn.executescript(
        """
        CREATE TABLE tick_snapshots (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy     TEXT NOT NULL,
            tick_ts      TEXT NOT NULL,
            decision_ts  TEXT,
            equity       REAL NOT NULL,
            peak_equity  REAL,
            positions    TEXT NOT NULL,
            n_submitted  INTEGER NOT NULL,
            reconcile_ok INTEGER NOT NULL
        );
        INSERT INTO tick_snapshots(strategy, tick_ts, equity, positions, n_submitted, reconcile_ok)
            VALUES ('legacy_strat', '2026-01-01T09:30:00', 100000.0, '{}', 0, 1);
        CREATE TABLE paper_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            target_weight REAL NOT NULL,
            decision_ts TEXT NOT NULL,
            submitted_ts TEXT NOT NULL,
            status TEXT NOT NULL,
            broker_order_id TEXT NOT NULL
        );
        INSERT INTO paper_orders(strategy, symbol, side, target_weight, decision_ts,
                                 submitted_ts, status, broker_order_id)
            VALUES ('legacy_strat', 'AAPL', 'buy', 0.1, '2026-01-01T09:30:00',
                    '2026-01-01T09:30:01', 'filled', 'ord-001');
        """
    )
    conn.commit()
    migrate(conn)

    tick_row = conn.execute(
        "SELECT * FROM tick_snapshots WHERE strategy='legacy_strat'"
    ).fetchone()
    for col in ("lane", "code_hash", "config_hash", "dependency_hash",
                "strategy_id", "account_id", "cash", "clock_source", "recorded_at"):
        assert tick_row[col] is None, f"tick_snapshots.{col} should be NULL on legacy row"

    order_row = conn.execute(
        "SELECT * FROM paper_orders WHERE strategy='legacy_strat'"
    ).fetchone()
    assert order_row["strategy_id"] is None, "paper_orders.strategy_id should be NULL on legacy row"


def test_strategies_has_metadata_columns(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(strategies)")}
    assert _META_COLS <= cols


def test_metadata_columns_are_null_on_existing_rows(tmp_path):
    # Simulate a pre-v17 DB: create the old-shaped table, insert a row, then migrate.
    db = tmp_path / "r.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE strategies (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL "
        "UNIQUE, stage TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
        "('legacy', 'idea', '2026-01-01', '2026-01-01')"
    )
    conn.commit()
    migrate(conn)
    row = conn.execute("SELECT * FROM strategies WHERE name='legacy'").fetchone()
    for col in _META_COLS:
        assert row[col] is None, f"{col} should be NULL on a pre-existing row, got {row[col]!r}"


def test_migrate_is_idempotent_at_v20(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    migrate(conn)  # second run must be a no-op, not an error
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


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


def test_migrate_creates_search_trials_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "search_trials" in tables
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(search_trials)")}
    # Keyed by strategy NAME (not the registry FK id) so pre-registration sweeps still count.
    assert {"id", "strategy_name", "n_combos", "grid_json", "created_at"} <= cols
    assert "strategy_id" not in cols


def test_migrate_adds_search_trials_to_legacy_db(tmp_path):
    """A legacy populated DB that predates search_trials gains the whole table via the
    CREATE TABLE IF NOT EXISTS bootstrap; re-running stays idempotent."""
    conn = connect(tmp_path / "r.db")
    # A legacy DB with the older core tables populated but no search_trials at all.
    conn.executescript(
        """
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
            stage TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        INSERT INTO strategies(name, stage, created_at, updated_at)
            VALUES ('s', 'idea', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');
        """
    )
    conn.commit()
    assert "search_trials" not in {
        r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }

    migrate(conn)
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "search_trials" in tables
    # The legacy strategy row survived and the new (name-keyed) table is usable.
    conn.execute(
        "INSERT INTO search_trials(strategy_name, n_combos, grid_json, created_at)"
        " VALUES (?,?,?,?)",
        ("s", 4, "{}", "2026-01-02T00:00:00+00:00"),
    )
    conn.commit()

    migrate(conn)  # idempotent re-run must not drop the row or raise
    assert conn.execute("SELECT COUNT(*) FROM search_trials").fetchone()[0] == 1


def test_migrate_rekeys_legacy_id_keyed_search_trials_to_name(tmp_path):
    """A dev DB with the OLD id-keyed search_trials is forward-migrated to the name-keyed table,
    carrying each row's breadth across by resolving strategy_id -> strategies.name."""
    conn = connect(tmp_path / "r.db")
    conn.executescript(
        """
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
            stage TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE search_trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL REFERENCES strategies(id),
            n_combos INTEGER NOT NULL,
            grid_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        INSERT INTO strategies(name, stage, created_at, updated_at)
            VALUES ('s', 'idea', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');
        INSERT INTO search_trials(strategy_id, n_combos, grid_json, created_at)
            VALUES (1, 7, '{}', '2026-01-02T00:00:00+00:00');
        """
    )
    conn.commit()

    migrate(conn)

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(search_trials)")}
    assert "strategy_name" in cols
    assert "strategy_id" not in cols
    rows = conn.execute(
        "SELECT strategy_name, n_combos FROM search_trials"
    ).fetchall()
    assert [(r["strategy_name"], r["n_combos"]) for r in rows] == [("s", 7)]

    migrate(conn)  # idempotent re-run must not duplicate or drop the row
    assert conn.execute("SELECT COUNT(*) FROM search_trials").fetchone()[0] == 1


def test_migrate_creates_holdout_evaluations_table(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "holdout_evaluations" in tables
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(holdout_evaluations)")}
    assert {"id", "strategy_id", "data_source", "snapshot_id", "period_start", "period_end",
            "holdout_frac", "config_hash", "reused", "created_at"} <= cols


def test_migrate_adds_holdout_evaluations_to_legacy_db(tmp_path):
    """A legacy populated DB that predates holdout_evaluations gains the whole table via the
    CREATE TABLE IF NOT EXISTS bootstrap; re-running stays idempotent."""
    conn = connect(tmp_path / "r.db")
    conn.executescript(
        """
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
            stage TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        INSERT INTO strategies(name, stage, created_at, updated_at)
            VALUES ('s', 'idea', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');
        """
    )
    conn.commit()
    assert "holdout_evaluations" not in {
        r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }

    migrate(conn)
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "holdout_evaluations" in tables
    sid = conn.execute("SELECT id FROM strategies WHERE name='s'").fetchone()["id"]
    conn.execute(
        "INSERT INTO holdout_evaluations(strategy_id, data_source, snapshot_id, period_start,"
        " period_end, holdout_frac, config_hash, reused, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (sid, "SyntheticProvider", None, "2022-01-01", "2023-12-31", 0.2, "cfg", 0,
         "2026-01-02T00:00:00+00:00"),
    )
    conn.commit()

    migrate(conn)  # idempotent re-run must not drop the row or raise
    assert conn.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone()[0] == 1


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


def test_gate_evaluations_table_exists(tmp_path):
    from algua.registry.db import connect, migrate
    conn = connect(tmp_path / "g.db")
    migrate(conn)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(gate_evaluations)").fetchall()}
    assert {
        "id", "strategy_id", "passed", "n_funnel", "own_lifetime_combos",
        "windowed_total_combos", "funnel_window_days", "breadth_provenance", "pit_ok",
        "pit_override", "holdout_n_bars", "min_holdout_observations", "code_hash", "config_hash",
        "dependency_hash", "data_source", "snapshot_id", "period_start", "period_end",
        "holdout_frac", "actor", "decision_json", "consumed", "created_at",
    } <= cols
    conn.close()


def test_ideas_table_created_with_expected_columns(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(ideas)")}
    assert {
        "id", "title", "hypothesis", "family", "tags", "source_type", "source_ref",
        "source_date", "source_note", "required_data", "status", "signature",
        "authored_strategy_id", "duplicate_of_idea_id", "override_reason",
        "created_at", "updated_at",
    } <= cols
    # FK to strategies(id) is declared
    fks = {row["table"] for row in conn.execute("PRAGMA foreign_key_list(ideas)")}
    assert "strategies" in fks
