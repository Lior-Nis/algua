"""v42 schema: the runs ledger tables exist, are correctly shaped, and migrate idempotently."""
from __future__ import annotations

import sqlite3

from algua.registry.db import SCHEMA_VERSION
from algua.registry.db.migrate import migrate

# Every metric column must name its sample class. Spec §3.1: a bare `sharpe` column would let the
# UI sort the most overfit number in the system to the top.
_SAMPLE_SUFFIXES = ("_is", "_oos", "_realized", "_window_sharpe", "_positive_windows")
_METRIC_PREFIXES = ("sharpe", "sortino", "total_return", "max_drawdown", "ann_vol",
                    "cagr", "calmar", "n_obs")


def _fresh() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrate(conn)
    return conn


def test_runs_tables_exist() -> None:
    conn = _fresh()
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "runs" in names
    assert "run_metrics" in names


def test_schema_version_is_42() -> None:
    conn = _fresh()
    assert SCHEMA_VERSION == 42
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 42


def test_no_bare_sharpe_column() -> None:
    """No metric column may omit its sample class (spec §3.1)."""
    conn = _fresh()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(runs)")]
    for col in cols:
        if any(col.startswith(p) for p in _METRIC_PREFIXES):
            assert col.endswith(_SAMPLE_SUFFIXES), f"{col} does not name its sample class"


def test_kind_check_rejects_unknown() -> None:
    conn = _fresh()
    try:
        conn.execute(
            "INSERT INTO runs(kind, strategy_name, created_at, metric_schema_version,"
            " derived_from, components, config_json) VALUES ('nonsense','s','t',1,'[]','[]','{}')")
    except sqlite3.IntegrityError:
        return
    raise AssertionError("kind CHECK did not reject an unknown kind")


def test_migrate_is_idempotent() -> None:
    conn = _fresh()
    migrate(conn)
    migrate(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 42


def test_migrates_a_legacy_db_lacking_runs() -> None:
    """A DB stamped at v41 with no runs table gains one without losing data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrate(conn)
    conn.execute("DROP TABLE runs")
    conn.execute("DROP TABLE run_metrics")
    conn.execute("PRAGMA user_version=41")
    migrate(conn)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"runs", "run_metrics"} <= names
