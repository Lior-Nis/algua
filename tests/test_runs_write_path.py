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
