"""strategy_board: one row per registered strategy, joined to its latest identity-bound gate run."""
from __future__ import annotations

import json
import sqlite3

import pytest

from algua.contracts.lifecycle import Stage
from algua.registry.db.migrate import migrate
from algua.registry.store import SqliteStrategyRepository
from algua.registry.strategy_board import (
    BOARD_METRICS,
    SCHEMA_VERSION,
    strategy_board_payload,
)


@pytest.fixture()
def repo() -> SqliteStrategyRepository:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrate(conn)
    return SqliteStrategyRepository(conn)


def _gate(repo: SqliteStrategyRepository, name: str, strategy_id: int | None, *,
          created_at: str | None = None, passed: bool | None = None, **metrics: float) -> int:
    run_id = repo.record_run(
        "gate", name, strategy_id=strategy_id, passed=passed, metrics=metrics or None)
    if created_at is not None:
        with repo.connection:
            repo.connection.execute(
                "UPDATE runs SET created_at=? WHERE id=?", (created_at, run_id))
    return run_id


def _set_stage(repo: SqliteStrategyRepository, name: str, stage: Stage) -> None:
    with repo.connection:
        repo.connection.execute("UPDATE strategies SET stage=? WHERE name=?", (stage.value, name))


def test_empty_registry_is_a_successful_empty_payload(repo: SqliteStrategyRepository) -> None:
    assert strategy_board_payload(repo) == {
        "schema_version": SCHEMA_VERSION, "count": 0, "rows": []}
    assert SCHEMA_VERSION == "strategy-board/v1"


def test_every_registered_strategy_appears_once_even_without_a_gate(
        repo: SqliteStrategyRepository) -> None:
    gated = repo.add("gated")
    repo.add("bare", family="trend")
    _gate(repo, "gated", gated.id, sharpe_oos=0.5)
    _gate(repo, "gated", gated.id, sharpe_oos=0.7)
    payload = strategy_board_payload(repo)
    assert payload["count"] == 2
    assert sorted(r["name"] for r in payload["rows"]) == ["bare", "gated"]
    bare = next(r for r in payload["rows"] if r["name"] == "bare")
    assert bare["latest_gate"] is None
    assert bare["family"] == "trend"
    assert bare["stage"] == "idea"
    assert isinstance(bare["strategy_id"], int)


def test_latest_gate_is_newest_not_best(repo: SqliteStrategyRepository) -> None:
    rec = repo.add("alpha")
    _gate(repo, "alpha", rec.id, created_at="2026-01-01T00:00:00+00:00", sharpe_oos=3.0)
    newest = _gate(repo, "alpha", rec.id, created_at="2026-02-01T00:00:00+00:00", sharpe_oos=-1.0)
    _gate(repo, "alpha", rec.id, created_at="2026-01-15T00:00:00+00:00", sharpe_oos=2.0)
    (row,) = strategy_board_payload(repo)["rows"]
    assert row["latest_gate"]["run_id"] == newest
    assert row["latest_gate"]["sharpe_oos"] == -1.0


def test_equal_timestamps_break_ties_by_higher_run_id(repo: SqliteStrategyRepository) -> None:
    rec = repo.add("alpha")
    stamp = "2026-03-01T00:00:00+00:00"
    _gate(repo, "alpha", rec.id, created_at=stamp, sharpe_oos=9.0)
    higher = _gate(repo, "alpha", rec.id, created_at=stamp, sharpe_oos=0.1)
    first = strategy_board_payload(repo)["rows"][0]["latest_gate"]["run_id"]
    second = strategy_board_payload(repo)["rows"][0]["latest_gate"]["run_id"]
    assert first == second == higher


def test_name_only_and_foreign_identity_runs_are_not_attached(
        repo: SqliteStrategyRepository) -> None:
    rec = repo.add("reused")
    foreign = repo.add("other_identity")
    _gate(repo, "reused", None, sharpe_oos=5.0)          # legacy name-only run
    _gate(repo, "reused", foreign.id, sharpe_oos=6.0)     # a different valid identity
    row = next(r for r in strategy_board_payload(repo)["rows"] if r["strategy_id"] == rec.id)
    assert row["latest_gate"] is None


def test_only_gate_kind_runs_represent_a_strategy(repo: SqliteStrategyRepository) -> None:
    rec = repo.add("alpha")
    repo.record_run("backtest", "alpha", strategy_id=rec.id, metrics={"sharpe_oos": 4.0})
    repo.record_run("walk_forward", "alpha", strategy_id=rec.id, metrics={"sharpe_oos": 4.0})
    (row,) = strategy_board_payload(repo)["rows"]
    assert row["latest_gate"] is None


def test_missing_metrics_stay_null_never_zero(repo: SqliteStrategyRepository) -> None:
    rec = repo.add("alpha")
    _gate(repo, "alpha", rec.id, sharpe_oos=0.4)
    (row,) = strategy_board_payload(repo)["rows"]
    gate = row["latest_gate"]
    assert gate["sharpe_oos"] == 0.4
    for key in BOARD_METRICS:
        if key != "sharpe_oos":
            assert gate[key] is None
    assert gate["passed"] is None


def test_malformed_stored_values_become_unavailable(repo: SqliteStrategyRepository) -> None:
    rec = repo.add("alpha")
    run_id = _gate(repo, "alpha", rec.id, sharpe_oos=0.4, passed=True)
    with repo.connection:
        # SQLite is dynamically typed: a TEXT / infinity can sit in a REAL column.
        repo.connection.execute(
            "UPDATE runs SET sharpe_oos='oops', total_return_oos=?, n_obs_oos=-2, passed=7"
            " WHERE id=?", (float("inf"), run_id))
    gate = strategy_board_payload(repo)["rows"][0]["latest_gate"]
    assert gate["run_id"] == run_id
    assert gate["sharpe_oos"] is None
    assert gate["total_return_oos"] is None
    assert gate["n_obs_oos"] is None
    assert gate["passed"] is None


def test_malformed_latest_timestamp_makes_representative_evidence_unavailable(
        repo: SqliteStrategyRepository) -> None:
    rec = repo.add("alpha")
    _gate(repo, "alpha", rec.id, created_at="2026-01-01T00:00:00+00:00", sharpe_oos=0.4)
    _gate(repo, "alpha", rec.id, created_at="zzz", sharpe_oos=9.0)
    (row,) = strategy_board_payload(repo)["rows"]
    assert row["latest_gate"] is None


def test_passed_is_a_real_boolean(repo: SqliteStrategyRepository) -> None:
    a, b = repo.add("a"), repo.add("b")
    _gate(repo, "a", a.id, passed=True)
    _gate(repo, "b", b.id, passed=False)
    rows = {r["name"]: r for r in strategy_board_payload(repo)["rows"]}
    assert rows["a"]["latest_gate"]["passed"] is True
    assert rows["b"]["latest_gate"]["passed"] is False


def test_rows_follow_lifecycle_then_case_insensitive_name_then_id(
        repo: SqliteStrategyRepository) -> None:
    for name, stage in [("zeta", Stage.PAPER), ("Beta", Stage.IDEA), ("alpha", Stage.IDEA),
                        ("live_one", Stage.LIVE), ("old", Stage.RETIRED),
                        ("cand", Stage.CANDIDATE), ("rest", Stage.DORMANT)]:
        repo.add(name)
        _set_stage(repo, name, stage)
    names = [r["name"] for r in strategy_board_payload(repo)["rows"]]
    assert names == ["alpha", "Beta", "cand", "zeta", "live_one", "rest", "old"]


def test_payload_is_scalar_only_and_json_serialisable(repo: SqliteStrategyRepository) -> None:
    rec = repo.add("alpha")
    gate_id = repo.record_run(
        "gate", "alpha", strategy_id=rec.id, derived_from=[1], components=[{"name": "m"}],
        config={"secret": 1}, extra_metrics={"dsr_t": 1.0}, passed=True,
        provenance={"code_hash": "abc"}, metrics={"sharpe_oos": 0.2})
    payload = strategy_board_payload(repo)
    assert set(payload) == {"schema_version", "count", "rows"}
    (row,) = payload["rows"]
    assert set(row) == {"strategy_id", "name", "family", "stage", "latest_gate"}
    assert set(row["latest_gate"]) == {"run_id", "created_at", "passed", *BOARD_METRICS}
    assert row["latest_gate"]["run_id"] == gate_id
    for value in row["latest_gate"].values():
        assert not isinstance(value, (list, dict))
    blob = json.dumps(payload)
    for leaked in ("derived_from", "components", "config", "hash", "decision", "returns", "abc"):
        assert leaked not in blob


def test_board_metrics_are_a_subset_of_the_fixed_vocabulary() -> None:
    from algua.registry.store.runs import METRIC_COLUMNS
    assert set(BOARD_METRICS) <= set(METRIC_COLUMNS)
    assert BOARD_METRICS == (
        "sharpe_oos", "total_return_oos", "max_drawdown_oos", "n_obs_oos",
        "mean_window_sharpe", "min_window_sharpe", "pct_positive_windows")


def test_projection_is_a_pure_read(repo: SqliteStrategyRepository) -> None:
    rec = repo.add("alpha")
    _gate(repo, "alpha", rec.id, sharpe_oos=0.4)
    before = repo.connection.total_changes
    strategy_board_payload(repo)
    assert repo.connection.total_changes == before
