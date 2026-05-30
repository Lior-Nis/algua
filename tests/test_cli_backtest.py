import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def test_backtest_run_demo_emits_metrics():
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                                 "--demo", "--start", "2023-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["strategy"] == "cross_sectional_momentum"
    assert "sharpe" in payload["metrics"]


def test_backtest_run_register_advances_registry():
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                                 "--demo", "--start", "2023-01-01", "--end", "2023-12-31",
                                 "--register"])
    assert result.exit_code == 0, result.stdout
    show = runner.invoke(app, ["registry", "show", "cross_sectional_momentum"])
    assert json.loads(show.stdout)["stage"] == "backtested"


def test_unknown_strategy_is_json_error():
    result = runner.invoke(app, ["backtest", "run", "nope", "--demo"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
