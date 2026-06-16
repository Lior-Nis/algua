# tests/test_cli_factor.py
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import algua.strategies.momentum as _momfam
from algua.backtest.engine import BacktestError
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


def test_uses_reports_composed_factor_for_bundled():
    # After composition (issue #140) the bundled strategy delegates to xs_trailing_return, whose
    # module imports indicators.py — so module-granular lineage reports all three.
    out = _json(runner.invoke(app, ["factor", "uses", "cross_sectional_momentum"]))
    assert {"xs_trailing_return", "momentum", "zscore"} <= set(out["factors"])


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
    # CLI applies FDR correction — ic.fdr_corrected is now True (#219 slice E)
    assert payload["ic"]["fdr_corrected"] is True
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


# ---------------------------------------------------------------------------
# Task 5: FDR wiring tests (#219)
# ---------------------------------------------------------------------------

def test_factor_eval_emits_fdr_block_with_corrected_flag():
    """After CLI wiring: factor eval emits an fdr block with fdr_corrected: True
    and ic.fdr_corrected is also overridden to True."""
    payload = _json(runner.invoke(app, [
        "factor", "eval", "xs_trailing_return",
        "--demo", "--start", "2023-01-01", "--end", "2023-06-30",
        "--symbols", "AAA,BBB,CCC",
        "--construction", "top_k_equal_weight", "--construction-param", "top_k=1",
        "--param", "lookback=10",
    ]))
    assert payload["ok"] is True
    # fdr block present
    assert "fdr" in payload, "missing top-level fdr block"
    fdr = payload["fdr"]
    assert fdr["fdr_corrected"] is True
    # ic.fdr_corrected is overridden to True by the CLI
    assert payload["ic"]["fdr_corrected"] is True
    # required fdr keys
    for key in ("n_hypotheses", "breadth_benchmark_t", "breadth_significant",
                 "dsr_binding", "significant"):
        assert key in fdr, f"missing fdr key: {key}"
    # blast radius present
    assert "n_dependents" in payload


def test_factor_eval_fdr_n_hypotheses_is_one_on_fresh_db():
    """First eval on an empty DB: n_hypotheses = 1 (this is the first hypothesis)."""
    payload = _json(runner.invoke(app, [
        "factor", "eval", "xs_trailing_return",
        "--demo", "--start", "2023-01-01", "--end", "2023-06-30",
        "--symbols", "AAA,BBB,CCC",
        "--construction", "top_k_equal_weight", "--construction-param", "top_k=1",
        "--param", "lookback=10",
    ]))
    assert payload["fdr"]["n_hypotheses"] == 1


def test_factor_eval_repeated_run_does_not_inflate_n_hypotheses():
    """Re-running the same factor/params/window: n_hypotheses stays 1 (dedup)."""
    _args = [
        "factor", "eval", "xs_trailing_return",
        "--demo", "--start", "2023-01-01", "--end", "2023-06-30",
        "--symbols", "AAA,BBB,CCC",
        "--construction", "top_k_equal_weight", "--construction-param", "top_k=1",
        "--param", "lookback=10",
    ]
    _json(runner.invoke(app, _args))   # first run
    payload2 = _json(runner.invoke(app, _args))  # identical re-run
    assert payload2["fdr"]["n_hypotheses"] == 1  # deduped


def test_factor_eval_different_params_inflate_n_hypotheses():
    """Two evals with different params → n_hypotheses = 2."""
    base = [
        "factor", "eval", "xs_trailing_return", "--demo",
        "--start", "2023-01-01", "--end", "2023-06-30",
        "--symbols", "AAA,BBB,CCC",
        "--construction", "top_k_equal_weight", "--construction-param", "top_k=1",
    ]
    _json(runner.invoke(app, base + ["--param", "lookback=10"]))
    payload2 = _json(runner.invoke(app, base + ["--param", "lookback=20"]))
    assert payload2["fdr"]["n_hypotheses"] == 2


def test_factor_eval_dsr_binding_flips_true_after_two_distinct_hypotheses():
    """DSR starts not binding (N=1); after a distinct second hypothesis DSR binds."""
    base = [
        "factor", "eval", "xs_trailing_return", "--demo",
        "--start", "2023-01-01", "--end", "2023-06-30",
        "--symbols", "AAA,BBB,CCC",
        "--construction", "top_k_equal_weight", "--construction-param", "top_k=1",
    ]
    p1 = _json(runner.invoke(app, base + ["--param", "lookback=10"]))
    assert p1["fdr"]["dsr_binding"] is False  # N=1

    p2 = _json(runner.invoke(app, base + ["--param", "lookback=20"]))
    # N=2 and trial_ir_var is now available (≥2 distinct IRs in window)
    # dsr_binding should be True (both conditions met)
    assert p2["fdr"]["dsr_binding"] is True


def test_factor_renders_backtest_error_as_json(monkeypatch):

    # evaluate_factor runs the backtest engine, which raises BacktestError on ordinary operational
    # failures (empty data, risk breach, ...). The command must keep those inside the JSON contract.
    import algua.cli.factor_cmd as fc

    def _boom(*a, **k):
        raise BacktestError("no bars in window")

    monkeypatch.setattr(fc, "evaluate_factor", _boom)
    result = runner.invoke(app, [
        "factor", "eval", "xs_trailing_return", "--demo", "--symbols", "AAA,BBB",
        "--construction", "equal_weight_positive", "--param", "lookback=5",
    ])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "no bars in window" in payload["error"]
