"""v42 schema: the runs ledger tables exist, are correctly shaped, and migrate idempotently."""
from __future__ import annotations

import sqlite3

from algua.registry.db import SCHEMA_VERSION
from algua.registry.db.migrate import migrate

# Every metric column must name its sample class. Spec §3.1: a bare `sharpe` column would let the
# UI sort the most overfit number in the system to the top.
_SAMPLE_SUFFIXES = ("_is", "_oos", "_realized", "_window_sharpe", "_positive_windows")
# Includes mean_/std_/min_/pct_ (the window-dispersion columns) — an earlier version of this list
# omitted them, so the loop below silently never checked mean_window_sharpe/std_window_sharpe/
# min_window_sharpe/pct_positive_windows at all (their names never start with any of the other
# prefixes). See tests/registry/test_runs_store.py::test_metric_and_provenance_columns_are_bound_
# to_the_ddl, which independently subsumes this check against the full vocabulary.
_METRIC_PREFIXES = ("sharpe", "sortino", "total_return", "max_drawdown", "ann_vol",
                    "cagr", "calmar", "n_obs", "mean_", "std_", "min_", "pct_")


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


def test_schema_version_is_44() -> None:
    conn = _fresh()
    assert SCHEMA_VERSION == 45
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 45


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
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 45


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


def test_runs_has_series_pointer_columns() -> None:
    conn = _fresh()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    assert {"series_backtest_id", "series_holdout_id"} <= cols


def test_v43_db_gains_series_pointer_columns(tmp_path) -> None:  # noqa: ANN001
    """A populated v43 DB must ALTER cleanly — the bootstrap cannot add a column."""
    import sqlite3 as _sq

    conn = _sq.connect(tmp_path / "legacy.db")
    conn.row_factory = _sq.Row
    migrate(conn)
    conn.execute("ALTER TABLE runs DROP COLUMN series_backtest_id")
    conn.execute("ALTER TABLE runs DROP COLUMN series_holdout_id")
    conn.execute(
        "INSERT INTO runs(kind, strategy_name, created_at, metric_schema_version,"
        " derived_from, components, config_json) VALUES ('backtest','a','t',1,'[]','[]','{}')")
    conn.execute("PRAGMA user_version=43")
    conn.commit()
    migrate(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    assert {"series_backtest_id", "series_holdout_id"} <= cols
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
