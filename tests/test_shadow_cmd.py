import json
import sys
from contextlib import closing
from pathlib import Path

import pytest
from typer.testing import CliRunner

import algua.strategies.momentum as _mom_pkg
from algua.backtest._sample import SyntheticProvider
from algua.cli.main import app
from algua.config.settings import get_settings
from algua.registry.db import connect, migrate

runner = CliRunner()

_START, _END = "2022-01-01", "2023-06-30"

_MOM_DIR = Path(_mom_pkg.__file__).parent

# A second plain, tradable strategy authored on the fly so `compare` has two distinct strategies.
# The loader rebuilds its filesystem index per call, so a module written here is discoverable.
_CHALLENGER_NAME = "shadow_test_challenger"
_CHALLENGER_SRC = (
    "import pandas as pd\n"
    "from algua.contracts.types import ExecutionContract\n"
    "from algua.strategies.base import StrategyConfig\n"
    "\n"
    "CONFIG = StrategyConfig(\n"
    "    name='shadow_test_challenger',\n"
    "    universe=['AAA', 'BBB'],\n"
    "    execution=ExecutionContract(rebalance_frequency='1d', decision_lag_bars=1),\n"
    "    construction='top_k_equal_weight', construction_params={'top_k': 1},\n"
    ")\n"
    "\n"
    "def signal(view, params):\n"
    "    # Momentum-ish: last close per symbol as the score.\n"
    "    closes = view.reset_index().pivot(index='timestamp', columns='symbol', "
    "values='adj_close')\n"
    "    return closes.iloc[-1]\n"
)


@pytest.fixture
def _challenger_module():
    path = _MOM_DIR / f"{_CHALLENGER_NAME}.py"
    path.write_text(_CHALLENGER_SRC)
    try:
        yield _CHALLENGER_NAME
    finally:
        path.unlink(missing_ok=True)
        sys.modules.pop(f"algua.strategies.momentum.{_CHALLENGER_NAME}", None)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    # Shadow reads a bars SNAPSHOT via _select_provider; patch it to the synthetic provider so the
    # test needs no ingested snapshot (mirrors the paper-lane tests).
    monkeypatch.setattr("algua.cli.shadow_cmd._select_provider",
                        lambda demo, snapshot: SyntheticProvider())


def _register(name: str) -> None:
    assert runner.invoke(app, ["backtest", "run", name, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0


def _shadow_rows(name: str) -> int:
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        return conn.execute(
            "SELECT COUNT(*) FROM shadow_evaluations WHERE challenger = ?", (name,)
        ).fetchone()[0]


def test_shadow_run_records_and_emits(monkeypatch):
    name = "cross_sectional_momentum"
    _register(name)
    result = runner.invoke(app, ["shadow", "run", name, "--snapshot", "snap1",
                                 "--start", _START, "--end", _END])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["challenger"] == name
    assert "sharpe" in payload and "final_equity" in payload
    # A row was recorded on the advisory ledger (never on any live/paper table).
    assert _shadow_rows(name) == 1


def test_shadow_run_rejects_unknown_strategy():
    result = runner.invoke(app, ["shadow", "run", "no_such_strategy", "--snapshot", "snap1",
                                 "--start", _START, "--end", _END])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_shadow_run_rejects_non_positive_cash():
    name = "cross_sectional_momentum"
    _register(name)
    result = runner.invoke(app, ["shadow", "run", name, "--snapshot", "snap1",
                                 "--start", _START, "--end", _END, "--cash", "0"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_shadow_compare_evaluates_both_on_same_surface(_challenger_module):
    champ, chal = "cross_sectional_momentum", _challenger_module
    _register(champ)
    _register(chal)
    result = runner.invoke(app, ["shadow", "compare", "--champion", champ, "--challenger", chal,
                                 "--snapshot", "snap1", "--start", _START, "--end", _END])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["champion"]["strategy"] == champ
    assert payload["challenger"]["strategy"] == chal
    # Both sides scored on the IDENTICAL surface (fair comparison).
    assert payload["surface"]["snapshot"] == "snap1"
    assert payload["surface"]["timeframe"] == "1d"
    assert "challenger_leads" in payload and isinstance(payload["challenger_leads"], bool)
    assert "sharpe" in payload["delta"]
    # Both rows recorded.
    assert _shadow_rows(champ) == 1 and _shadow_rows(chal) == 1


def test_shadow_compare_rejects_same_strategy():
    name = "cross_sectional_momentum"
    _register(name)
    result = runner.invoke(app, ["shadow", "compare", "--champion", name, "--challenger", name,
                                 "--snapshot", "snap1", "--start", _START, "--end", _END])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_shadow_compare_does_not_transition_stage(_challenger_module):
    champ, chal = "cross_sectional_momentum", _challenger_module
    _register(champ)
    _register(chal)
    runner.invoke(app, ["shadow", "compare", "--champion", champ, "--challenger", chal,
                        "--snapshot", "snap1", "--start", _START, "--end", _END])
    # Neither strategy advanced beyond its registered stage: shadow is ADVISORY, never promotes.
    for name in (champ, chal):
        show = json.loads(runner.invoke(app, ["registry", "show", name]).stdout)
        assert show["stage"] == "backtested"


def test_shadow_show_returns_latest_eval():
    name = "cross_sectional_momentum"
    _register(name)
    runner.invoke(app, ["shadow", "run", name, "--snapshot", "snap1",
                        "--start", _START, "--end", _END])
    result = runner.invoke(app, ["shadow", "show", name])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["latest_shadow_evaluation"]["challenger"] == name
    assert payload["latest_shadow_evaluation"]["snapshot_id"] == "snap1"


def test_shadow_show_unknown_strategy_fails():
    result = runner.invoke(app, ["shadow", "show", "no_such_strategy"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
