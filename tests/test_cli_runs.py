"""`algua runs list` — the CLI seam over `algua.registry.run_views.run_list_payload`."""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.registry.db import registry_conn
from algua.registry.store import SqliteStrategyRepository

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def _json(result):
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def _record(kind: str, strategy: str, **metrics: float | None) -> int:
    with registry_conn() as conn:
        return SqliteStrategyRepository(conn).record_run(kind, strategy, metrics=metrics)


def test_empty_ledger_is_ok_with_zero_rows():
    payload = _json(runner.invoke(app, ["runs", "list"]))
    assert payload["ok"] is True
    assert payload["runs"] == []
    assert payload["count"] == 0


def test_kind_filters():
    _record("backtest", "alpha")
    _record("walk_forward", "alpha")
    payload = _json(runner.invoke(app, ["runs", "list", "--kind", "backtest"]))
    assert payload["count"] == 1
    assert payload["runs"][0]["kind"] == "backtest"


def test_sort_orders_best_first_with_nulls_last():
    _record("walk_forward", "alpha", sharpe_oos=None)
    _record("walk_forward", "beta", sharpe_oos=0.5)
    _record("walk_forward", "gamma", sharpe_oos=1.5)
    payload = _json(runner.invoke(app, ["runs", "list", "--sort", "sharpe_oos"]))
    names = [row["strategy_name"] for row in payload["runs"]]
    assert names == ["gamma", "beta", "alpha"]


def test_limit_caps():
    for i in range(5):
        _record("backtest", f"strat{i}")
    payload = _json(runner.invoke(app, ["runs", "list", "--limit", "2"]))
    assert payload["count"] == 2
    assert len(payload["runs"]) == 2


def test_bad_sort_exits_nonzero_with_json_error_envelope():
    result = runner.invoke(app, ["runs", "list", "--sort", "1; DROP TABLE runs"])
    assert result.exit_code == 1
    out = json.loads(result.stdout)
    assert out["ok"] is False
    assert "error" in out
    assert "code" in out


def test_strategy_filters():
    _record("backtest", "alpha")
    _record("backtest", "beta")
    payload = _json(runner.invoke(app, ["runs", "list", "--strategy", "alpha"]))
    assert payload["count"] == 1
    assert payload["runs"][0]["strategy_name"] == "alpha"


def test_family_filters():
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        repo.add("alpha", family="momentum")
        repo.add("beta", family="mean_reversion")
        repo.record_run("backtest", "alpha")
        repo.record_run("backtest", "beta")
    payload = _json(runner.invoke(app, ["runs", "list", "--family", "momentum"]))
    assert payload["count"] == 1
    assert payload["runs"][0]["strategy_name"] == "alpha"
