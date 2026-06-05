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


def _sweep():
    return runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                               "--start", "2022-01-01", "--end", "2023-12-31",
                               "--param", "lookback=20,40", "--param", "top_k=1,3"])


def test_promote_passes_and_shortlists():
    assert _backtest_to_backtested().exit_code == 0
    # No sweep recorded yet, so declare breadth explicitly via --n-combos.
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
    assert payload["breadth_provenance"] == "declared"
    # A declared breadth still raises the bar: effective > base.
    assert payload["effective_min_holdout_sharpe"] > payload["base_min_holdout_sharpe"]
    assert _stage() == "shortlisted"


def test_promote_uses_measured_breadth_from_sweep():
    assert _backtest_to_backtested().exit_code == 0
    assert _sweep().exit_code == 0  # records a 4-combo search_trial
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                 "--min-pct-positive", "0", "--min-window-sharpe", "-100"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["breadth_provenance"] == "measured"
    assert payload["n_combos"] == 4


def test_measured_breadth_wins_over_declaration():
    assert _backtest_to_backtested().exit_code == 0
    assert _sweep().exit_code == 0  # 4 combos measured
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                 "--min-pct-positive", "0", "--min-window-sharpe", "-100",
                                 "--n-combos", "999"])  # declaration ignored
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["breadth_provenance"] == "measured"
    assert payload["n_combos"] == 4


def test_two_sweeps_accumulate_breadth():
    assert _backtest_to_backtested().exit_code == 0
    assert _sweep().exit_code == 0
    assert _sweep().exit_code == 0  # second sweep accumulates
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                 "--min-pct-positive", "0", "--min-window-sharpe", "-100"])
    payload = json.loads(result.stdout)
    assert payload["n_combos"] == 8  # 4 + 4


def test_promote_refuses_with_no_breadth():
    assert _backtest_to_backtested().exit_code == 0
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                 "--min-pct-positive", "0", "--min-window-sharpe", "-100"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "search breadth" in payload["error"]
    assert _stage() == "backtested"  # not transitioned


def test_promote_fails_does_not_transition():
    assert _backtest_to_backtested().exit_code == 0
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "999", "--n-combos", "1"])
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


def test_promote_rejects_bad_n_combos():
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum",
                                 "--demo", "--n-combos", "0"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_promote_rejects_out_of_range_pct_positive():
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum",
                                 "--demo", "--min-pct-positive", "1.5"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
