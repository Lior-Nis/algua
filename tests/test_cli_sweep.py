import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def test_sweep_demo_emits_ranked():
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--param", "lookback=20,40", "--param", "top_k=1,3",
                                 "--top", "2"])
    assert result.exit_code == 0, result.stdout
    d = json.loads(result.stdout)
    assert d["n_combos"] == 4
    assert len(d["ranked"]) == 2
    assert d["best"]["params"]
    assert d["rank_by"] == "mean_sharpe"


def test_sweep_requires_param():
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_sweep_malformed_param_is_json_error():
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                                 "--param", "lookback"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_sweep_requires_data_source():
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum",
                                 "--param", "lookback=20,40"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_sweep_top_zero_is_json_error():
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                                 "--param", "lookback=20,40", "--top", "0"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
