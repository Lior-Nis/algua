"""Every evaluation lands a run row — including for an UNREGISTERED strategy."""
from __future__ import annotations

import pytest

from algua.cli._common import registry_conn
from algua.cli.backtest_cmd import run_backtest_task
from algua.registry.store import SqliteStrategyRepository

STRATEGY = "cross_sectional_momentum"


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    """The established DB-isolation idiom (see tests/test_cli_backtest.py)."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def test_backtest_records_a_run() -> None:
    run_backtest_task(STRATEGY, demo=True)
    with registry_conn() as conn:
        rows = SqliteStrategyRepository(conn).list_runs(kind="backtest")
    assert len(rows) == 1
    assert rows[0]["strategy_name"] == STRATEGY
    assert rows[0]["sharpe_is"] is not None
    assert rows[0]["code_hash"] is not None


def test_backtest_records_a_run_for_an_unregistered_strategy() -> None:
    """Exploration precedes registration — that evidence must not be discarded."""
    run_backtest_task(STRATEGY, demo=True, register=False)
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        assert STRATEGY not in {s.name for s in repo.list_strategies()}
        assert len(repo.list_runs(kind="backtest")) == 1


def test_walk_forward_records_oos_and_window_metrics() -> None:
    from typer.testing import CliRunner

    from algua.cli.main import app

    res = CliRunner().invoke(app, ["backtest", "walk-forward", STRATEGY, "--demo"])
    assert res.exit_code == 0, res.output
    with registry_conn() as conn:
        rows = SqliteStrategyRepository(conn).list_runs(kind="walk_forward")
    assert len(rows) == 1
    row = rows[0]
    assert row["sharpe_oos"] is not None
    assert row["n_obs_oos"] is not None
    assert row["mean_window_sharpe"] is not None
    # A walk-forward measures no full-period in-sample figure; it must not invent one.
    assert row["sharpe_is"] is None


def _run_sweep() -> None:
    from typer.testing import CliRunner

    from algua.cli.main import app

    res = CliRunner().invoke(
        app, ["backtest", "sweep", STRATEGY, "--demo", "--param", "lookback=20,40,60"])
    assert res.exit_code == 0, res.output


def test_sweep_records_a_parent_and_one_child_per_combo() -> None:
    import json

    _run_sweep()
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        parents = repo.list_runs(kind="sweep")
        children = repo.list_runs(kind="sweep_trial")
    assert len(parents) == 1
    assert len(children) == 3
    assert parents[0]["trials_truncated_at"] is None
    assert all(json.loads(c["derived_from"]) == [parents[0]["id"]] for c in children)
    assert all(c["mean_window_sharpe"] is not None for c in children)


def test_sweep_trial_count_matches_the_recorded_breadth() -> None:
    """The point of the slice: n_combos stops being an assertion and becomes a countable set."""
    _run_sweep()
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        n_children = len(repo.list_runs(kind="sweep_trial"))
        declared = repo.total_search_combos(STRATEGY)
    assert n_children == declared
