import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def test_walk_forward_demo_emits_windows_and_stability_but_no_holdout():
    result = runner.invoke(app, ["backtest", "walk-forward", "cross_sectional_momentum",
                                 "--demo", "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--windows", "4", "--holdout-frac", "0.2"])
    assert result.exit_code == 0, result.stdout
    d = json.loads(result.stdout)
    assert len(d["window_metrics"]) == 4
    assert "mean_sharpe" in d["stability"]
    # The OOS holdout is WITHHELD here: it is revealed (and burned) only by `research promote`.
    # Emitting it from walk-forward would defeat the single-use guarantee.
    assert "holdout_metrics" not in d


def test_walk_forward_requires_a_data_source():
    result = runner.invoke(app, ["backtest", "walk-forward", "cross_sectional_momentum"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_walk_forward_too_few_bars_is_json_error():
    result = runner.invoke(app, ["backtest", "walk-forward", "cross_sectional_momentum",
                                 "--demo", "--start", "2023-12-01", "--end", "2023-12-10",
                                 "--windows", "4"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
