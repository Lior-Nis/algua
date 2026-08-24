"""RunLedgerMixin: insert + read back, JSON-list lineage, overflow metrics, trial capping."""
from __future__ import annotations

import json
import sqlite3

import pytest

from algua.registry.db.migrate import migrate
from algua.registry.store import SqliteStrategyRepository
from algua.registry.store.runs import METRIC_COLUMNS, PROVENANCE_COLUMNS


@pytest.fixture()
def repo() -> SqliteStrategyRepository:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrate(conn)
    return SqliteStrategyRepository(conn)


def test_record_run_round_trips(repo: SqliteStrategyRepository) -> None:
    run_id = repo.record_run(
        "backtest", "alpha",
        provenance={"code_hash": "abc", "period_start": "2020-01-01"},
        metrics={"sharpe_is": 1.25, "n_obs_is": 500},
        config={"lookback": 60},
    )
    row = repo.get_run(run_id)
    assert row is not None
    assert row["kind"] == "backtest"
    assert row["strategy_name"] == "alpha"
    assert row["code_hash"] == "abc"
    assert row["sharpe_is"] == pytest.approx(1.25)
    assert row["n_obs_is"] == 500
    assert json.loads(row["config_json"]) == {"lookback": 60}
    assert row["metric_schema_version"] == 1


def test_lineage_defaults_to_empty_lists(repo: SqliteStrategyRepository) -> None:
    row = repo.get_run(repo.record_run("backtest", "alpha"))
    assert row is not None
    assert json.loads(row["derived_from"]) == []
    assert json.loads(row["components"]) == []


def test_lineage_round_trips_as_lists(repo: SqliteStrategyRepository) -> None:
    parent = repo.record_run("walk_forward", "alpha")
    child = repo.record_run(
        "gate", "alpha",
        derived_from=[parent],
        components=[{"name": "sentiment", "version": 3, "digest": "d0"}],
    )
    row = repo.get_run(child)
    assert row is not None
    assert json.loads(row["derived_from"]) == [parent]
    assert json.loads(row["components"])[0]["name"] == "sentiment"


def test_unknown_metric_key_is_rejected(repo: SqliteStrategyRepository) -> None:
    """A typo'd metric must not vanish silently into nowhere."""
    with pytest.raises(ValueError, match="not in the fixed metric vocabulary"):
        repo.record_run("backtest", "alpha", metrics={"sharpe": 1.0})


def test_extra_metrics_land_in_the_overflow_table(repo: SqliteStrategyRepository) -> None:
    run_id = repo.record_run(
        "gate", "alpha", extra_metrics={"dsr_confidence": 0.096, "market_beta": 0.21})
    rows = {r["key"]: r["value"] for r in repo.connection.execute(
        "SELECT key, value FROM run_metrics WHERE run_id=?", (run_id,))}
    assert rows["dsr_confidence"] == pytest.approx(0.096)
    assert rows["market_beta"] == pytest.approx(0.21)


def test_non_finite_extra_metric_is_stored_as_null(repo: SqliteStrategyRepository) -> None:
    run_id = repo.record_run("gate", "alpha", extra_metrics={"dsr_n_eff": float("nan")})
    row = repo.connection.execute(
        "SELECT value FROM run_metrics WHERE run_id=? AND key='dsr_n_eff'", (run_id,)).fetchone()
    assert row["value"] is None


def test_record_sweep_trials_writes_children(repo: SqliteStrategyRepository) -> None:
    parent = repo.record_run("sweep", "alpha")
    trials = [
        {"config": {"lookback": lb}, "metrics": {"mean_window_sharpe": float(i)}}
        for i, lb in enumerate([60, 90, 120])
    ]
    n, truncated = repo.record_sweep_trials(parent, "alpha", trials)
    assert (n, truncated) == (3, None)
    kids = repo.list_runs(kind="sweep_trial", strategy_name="alpha")
    assert len(kids) == 3
    assert all(json.loads(k["derived_from"]) == [parent] for k in kids)


def test_record_sweep_trials_caps_and_reports(
    repo: SqliteStrategyRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Beyond the cap: stop writing rows and REPORT the truncation, never silently."""
    monkeypatch.setattr("algua.registry.store.runs.MAX_PERSISTED_TRIALS", 2)
    parent = repo.record_run("sweep", "alpha")
    trials = [{"config": {"i": i}, "metrics": {}} for i in range(5)]
    n, truncated = repo.record_sweep_trials(parent, "alpha", trials)
    assert (n, truncated) == (2, 2)
    assert len(repo.list_runs(kind="sweep_trial", strategy_name="alpha")) == 2


def test_list_runs_sorts_by_metric_descending_nulls_last(
    repo: SqliteStrategyRepository,
) -> None:
    repo.record_run("gate", "a", metrics={"sharpe_oos": 0.1})
    repo.record_run("gate", "b", metrics={"sharpe_oos": 2.0})
    repo.record_run("gate", "c")  # NULL sharpe_oos
    names = [r["strategy_name"] for r in repo.list_runs(kind="gate", sort="sharpe_oos")]
    assert names == ["b", "a", "c"]


def test_list_runs_rejects_a_non_vocabulary_sort_key(repo: SqliteStrategyRepository) -> None:
    """The sort key is interpolated into SQL, so it MUST be allow-listed."""
    with pytest.raises(ValueError, match="not a sortable metric"):
        repo.list_runs(sort="1; DROP TABLE runs")


def test_sweep_trial_rejects_a_metric_outside_the_trial_vocabulary(
    repo: SqliteStrategyRepository,
) -> None:
    """A trial has no out-of-sample segment: a full-vocabulary-but-not-trial-scoped key like
    `sharpe_is` must raise, not silently vanish."""
    parent = repo.record_run("sweep", "alpha")
    trials = [{"config": {"lookback": 60}, "metrics": {"sharpe_is": 1.2}}]
    with pytest.raises(ValueError, match="not a sweep-trial metric"):
        repo.record_sweep_trials(parent, "alpha", trials)
    assert repo.list_runs(kind="sweep_trial", strategy_name="alpha") == []


def test_metric_and_provenance_columns_are_bound_to_the_ddl() -> None:
    """Nothing else ties METRIC_COLUMNS / PROVENANCE_COLUMNS to the actual `runs` DDL — a column
    renamed or dropped in algua/registry/db/runs.py without a matching update here would silently
    desync the fixed vocabulary from what SQLite actually stores.

    Also SUBSUMES an earlier deferred finding: tests/registry/test_runs_schema.py's
    `test_no_bare_sharpe_column` checks column naming via a `_METRIC_PREFIXES` allowlist that did
    not include `mean_`/`std_`/`min_`/`pct_`, so it silently never checked
    mean_window_sharpe/std_window_sharpe/min_window_sharpe/pct_positive_windows at all. This test
    instead partitions every real `runs` column into "known non-metric" (identity/lineage/
    administrative) + PROVENANCE_COLUMNS + METRIC_COLUMNS and asserts nothing is left over — so a
    column that is metric-shaped but NOT in METRIC_COLUMNS (whatever its name) fails loudly.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrate(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}

    for name in METRIC_COLUMNS:
        assert name in cols, f"{name!r} in METRIC_COLUMNS is not a real `runs` column"
    for name in PROVENANCE_COLUMNS:
        assert name in cols, f"{name!r} in PROVENANCE_COLUMNS is not a real `runs` column"

    # Identity, lineage, and administrative columns — everything in the DDL that is neither
    # provenance nor a measurement.
    _NON_METRIC = {
        "id", "kind", "strategy_name", "strategy_id", "created_at", "metric_schema_version",
        "derived_from", "components", "config_json", "passed", "trials_truncated_at", "gate_id",
    } | set(PROVENANCE_COLUMNS)
    metric_shaped = cols - _NON_METRIC
    assert metric_shaped == set(METRIC_COLUMNS), (
        "a `runs` column exists that is neither a known non-metric column nor in the fixed "
        f"metric vocabulary: {metric_shaped - set(METRIC_COLUMNS)}"
    )
