import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def _stage(name="cross_sectional_momentum"):
    show = runner.invoke(app, ["registry", "show", name])
    return json.loads(show.stdout)["stage"]


def _backtest_to_backtested():
    return runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                               "--start", "2022-01-01", "--end", "2023-12-31", "--register"])


def test_promote_passes_and_shortlists():
    assert _backtest_to_backtested().exit_code == 0
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                 "--min-pct-positive", "0", "--min-window-sharpe", "-100",
                                 "--n-combos", "9"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["promoted"] is True
    assert payload["n_combos"] == 9
    assert _stage() == "shortlisted"


def test_promote_fails_does_not_transition():
    assert _backtest_to_backtested().exit_code == 0
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "999"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] is False
    assert payload["promoted"] is False
    assert _stage() == "backtested"


def test_promote_from_idea_is_json_error():
    runner.invoke(app, ["registry", "add", "cross_sectional_momentum"])
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                 "--min-pct-positive", "0", "--min-window-sharpe", "-100"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
