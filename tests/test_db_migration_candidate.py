import sqlite3

from algua.registry.db import SCHEMA_VERSION, migrate


def _v19_with_rows(conn: sqlite3.Connection) -> None:
    """Minimal pre-rename shape: strategies + stage_transitions holding 'shortlisted'."""
    conn.executescript(
        """
        CREATE TABLE strategies (id INTEGER PRIMARY KEY, name TEXT NOT NULL, stage TEXT NOT NULL);
        CREATE TABLE stage_transitions (
            id INTEGER PRIMARY KEY, strategy_id INTEGER NOT NULL,
            from_stage TEXT, to_stage TEXT NOT NULL
        );
        INSERT INTO strategies(name, stage) VALUES ('s1', 'shortlisted'), ('s2', 'paper');
        INSERT INTO stage_transitions(strategy_id, from_stage, to_stage)
            VALUES (1, 'backtested', 'shortlisted'), (1, 'shortlisted', 'paper');
        """
    )
    conn.commit()


def test_migration_rewrites_shortlisted_rows():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _v19_with_rows(conn)
    migrate(conn)
    stages = {r["name"]: r["stage"] for r in conn.execute("SELECT name, stage FROM strategies")}
    assert stages == {"s1": "candidate", "s2": "paper"}
    froms = [r["from_stage"] for r in conn.execute("SELECT from_stage FROM stage_transitions")]
    tos = [r["to_stage"] for r in conn.execute("SELECT to_stage FROM stage_transitions")]
    assert "shortlisted" not in froms and "shortlisted" not in tos
    assert "candidate" in froms and "candidate" in tos
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_migration_fresh_empty_db_is_clean():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrate(conn)  # no tables yet — must not raise "no such table"
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_migration_strategies_only_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE strategies (id INTEGER PRIMARY KEY, name TEXT NOT NULL, stage TEXT NOT NULL);"
        "INSERT INTO strategies(name, stage) VALUES ('s1', 'shortlisted');"
    )
    conn.commit()
    migrate(conn)  # stage_transitions absent — per-table guard must skip it, not raise
    assert conn.execute("SELECT stage FROM strategies WHERE name='s1'").fetchone()[0] == "candidate"


def test_migration_runs_even_when_already_stamped():
    """No user_version gating: a DB stamped at the new version but still holding old rows is fixed."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _v19_with_rows(conn)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()
    migrate(conn)
    assert conn.execute("SELECT stage FROM strategies WHERE name='s1'").fetchone()[0] == "candidate"
