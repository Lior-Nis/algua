# tests/test_cli_factor.py
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import algua.strategies.momentum as _momfam
from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def _json(result):
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_list_includes_seeded_factors():
    rows = _json(runner.invoke(app, ["factor", "list"]))
    names = {r["name"] for r in rows}
    assert {"momentum", "zscore"} <= names
    mom = next(r for r in rows if r["name"] == "momentum")
    assert mom["import_path"] == "algua.features.indicators:momentum"
    assert mom["platform_supported"] is True
    assert mom["data_needs"] == ["ohlcv"]


def test_list_filters_by_kind():
    rows = _json(runner.invoke(app, ["factor", "list", "--kind", "momentum"]))
    names = {r["name"] for r in rows}
    assert "momentum" in names
    assert "zscore" not in names


def test_invalid_kind_uses_error_envelope():
    result = runner.invoke(app, ["factor", "list", "--kind", "bogus"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "bogus" in payload["error"]


def test_show_full_spec():
    out = _json(runner.invoke(app, ["factor", "show", "momentum"]))
    assert out["ok"] is True
    assert out["module"] == "algua.features.indicators"
    assert out["doc"]


def test_show_unknown_uses_error_envelope():
    result = runner.invoke(app, ["factor", "show", "nope"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_uses_reports_no_catalogued_factor_for_bundled():
    out = _json(runner.invoke(app, ["factor", "uses", "cross_sectional_momentum"]))
    assert out["factors"] == []


def _write_dep_strategy(stem: str) -> Path:
    path = Path(_momfam.__path__[0]) / f"{stem}.py"
    path.write_text(
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "from algua.features.indicators import momentum\n"
        f"CONFIG = StrategyConfig(name='{stem}', universe=['AAPL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:\n"
        "    wide = view.reset_index().pivot(index='timestamp', columns='symbol',\n"
        "        values='adj_close')\n"
        "    return momentum(wide.iloc[-1], 1).dropna()\n"
    )
    return path


def test_dependents_lists_registered_importer():
    path = _write_dep_strategy("tmp_cli_dep")
    try:
        assert runner.invoke(app, ["registry", "add", "tmp_cli_dep"]).exit_code == 0
        out = _json(runner.invoke(app, ["factor", "dependents", "momentum"]))
        assert "tmp_cli_dep" in out["dependents"]
        assert out["unloadable"] == []
    finally:
        path.unlink(missing_ok=True)
        sys.modules.pop("algua.strategies.momentum.tmp_cli_dep", None)


def test_dependents_nonzero_exit_on_unloadable_without_allow_partial():
    assert runner.invoke(app, ["registry", "add", "ghost_cli"]).exit_code == 0
    result = runner.invoke(app, ["factor", "dependents", "momentum"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert any(u["name"] == "ghost_cli" for u in payload["unloadable"])
    # with --allow-partial it exits 0 but still reports
    ok_result = runner.invoke(app, ["factor", "dependents", "momentum", "--allow-partial"])
    assert ok_result.exit_code == 0


def test_factor_eval_emits_backtest_and_ic():
    payload = _json(runner.invoke(app, [
        "factor", "eval", "xs_trailing_return",
        "--demo", "--start", "2023-01-01", "--end", "2023-06-30",
        "--symbols", "AAA,BBB,CCC",
        "--construction", "top_k_equal_weight", "--construction-param", "top_k=1",
        "--param", "lookback=10",
    ]))
    assert payload["ok"] is True
    assert payload["factor"] == "xs_trailing_return"
    assert payload["ic"]["method"] == "spearman"
    assert payload["ic"]["fdr_corrected"] is False
    assert "metrics" in payload["backtest"]


def test_factor_eval_requires_construction():
    result = runner.invoke(app, [
        "factor", "eval", "xs_trailing_return", "--demo",
        "--symbols", "AAA,BBB", "--param", "lookback=10",
    ])
    assert result.exit_code != 0  # typer: missing required --construction


def test_factor_eval_rejects_non_standalone_factor():
    result = runner.invoke(app, [
        "factor", "eval", "momentum", "--demo", "--symbols", "AAA,BBB",
        "--construction", "equal_weight_positive",
    ])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "standalone" in payload["error"].lower()
