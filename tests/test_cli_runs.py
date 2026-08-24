"""`algua runs list`/`show`/`series` — the CLI seam over `algua.registry.run_views`."""
from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.registry.db import registry_conn
from algua.registry.gate_history import GATE_DECISION_ALLOWLIST
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


def test_list_output_contains_no_series():
    """The two commands share a module — guard against widening `runs list`'s payload to carry a
    series (that is `runs series`'s job alone, see #349)."""
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        returns = pd.Series(
            [0.01, -0.02], index=pd.to_datetime(["2024-01-02", "2024-01-03"]), dtype=float)
        returns_id = repo.persist_backtest_returns("alpha", "2024-01-01", "2024-01-05", returns)
        repo.record_run("backtest", "alpha", series_backtest_id=returns_id)
    payload = _json(runner.invoke(app, ["runs", "list"]))
    (row,) = payload["runs"]
    assert "returns" not in row
    assert "returns_json" not in row
    assert row["series_backtest_id"] == returns_id


# -- runs show --------------------------------------------------------------------------------


def _record_gate(repo: SqliteStrategyRepository, decision_json: str) -> int:
    rec = repo.add("alpha")
    gate_id = repo.record_gate_evaluation(
        rec.id, passed=True, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=90, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=63, min_holdout_observations=63, code_hash="c", config_hash="cfg",
        dependency_hash="d", data_source="SyntheticProvider", snapshot_id=None,
        period_start="2024-01-01", period_end="2024-06-01", holdout_frac=0.2, actor="agent",
        decision_json=decision_json,
    )
    return repo.record_run("gate", "alpha", strategy_id=rec.id, gate_id=gate_id)


def test_show_returns_one_run():
    run_id = _record("backtest", "alpha", sharpe_is=1.0)
    payload = _json(runner.invoke(app, ["runs", "show", str(run_id)]))
    assert payload["ok"] is True
    assert payload["id"] == run_id
    assert payload["strategy_name"] == "alpha"


def test_show_missing_run_id_is_json_error_envelope():
    result = runner.invoke(app, ["runs", "show", "9999"])
    assert result.exit_code == 1
    out = json.loads(result.stdout)
    assert out["ok"] is False
    assert "error" in out
    assert "code" in out


def test_show_gate_run_includes_checks_and_stays_inside_allowlist():
    decision_json = json.dumps({
        "passed": True,
        "checks": [{
            "name": "holdout_sharpe", "op": ">", "threshold": 0.0, "value": 1.2,
            "passed": True, "advisory": False,
        }],
        # Deliberately outside GATE_DECISION_ALLOWLIST — must never survive projection.
        "per_regime_sharpes": [0.1, 0.2, 0.3],
        "not_a_real_field": "smuggled",
    })
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        run_id = _record_gate(repo, decision_json)
    payload = _json(runner.invoke(app, ["runs", "show", str(run_id)]))
    decision = payload["gate_decision"]
    assert decision["checks"][0]["name"] == "holdout_sharpe"
    # The test that matters most: assert against the REAL allowlist constant, not a hand-copied
    # list — a hand-copied list silently rots the moment the allowlist changes.
    assert set(decision) <= GATE_DECISION_ALLOWLIST
    assert "per_regime_sharpes" not in decision
    assert "not_a_real_field" not in decision


# -- runs series --------------------------------------------------------------------------------


def test_series_single_run_returns_its_series():
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        returns = pd.Series(
            [0.01, -0.02], index=pd.to_datetime(["2024-01-02", "2024-01-03"]), dtype=float)
        series_id = repo.persist_backtest_returns("alpha", "2024-01-01", "2024-01-05", returns)
        run_id = repo.record_run("backtest", "alpha", series_backtest_id=series_id)
    payload = _json(runner.invoke(app, ["runs", "series", str(run_id)]))
    entry = payload["series"][str(run_id)]
    assert entry["kind"] == "backtest"
    assert len(entry["returns"]) == 2


def test_series_several_runs_returns_several():
    id_a = _record("backtest", "alpha")
    id_b = _record("backtest", "beta")
    payload = _json(
        runner.invoke(app, ["runs", "series", str(id_a), "--run-id", str(id_b)]))
    assert set(payload["series"]) == {str(id_a), str(id_b)}


def test_series_run_with_no_pointer_is_null_not_omitted():
    run_id = _record("backtest", "alpha")
    payload = _json(runner.invoke(app, ["runs", "series", str(run_id)]))
    assert str(run_id) in payload["series"]
    assert payload["series"][str(run_id)] is None


def test_series_dedupes_repeated_ids():
    run_id = _record("backtest", "alpha")
    payload = _json(
        runner.invoke(app, ["runs", "series", str(run_id), "--run-id", str(run_id)]))
    assert list(payload["series"]) == [str(run_id)]


def test_series_caps_at_16_ids():
    ids = [_record("backtest", f"strat{i}") for i in range(17)]
    args = ["runs", "series", str(ids[0])]
    for run_id in ids[1:]:
        args += ["--run-id", str(run_id)]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    out = json.loads(result.stdout)
    assert out["ok"] is False
    assert "16" in out["error"]


def test_series_at_16_ids_succeeds():
    ids = [_record("backtest", f"strat{i}") for i in range(16)]
    args = ["runs", "series", str(ids[0])]
    for run_id in ids[1:]:
        args += ["--run-id", str(run_id)]
    payload = _json(runner.invoke(app, args))
    assert len(payload["series"]) == 16


def test_series_missing_run_id_is_json_error_envelope():
    result = runner.invoke(app, ["runs", "series", "9999"])
    assert result.exit_code == 1
    out = json.loads(result.stdout)
    assert out["ok"] is False
    assert "error" in out
    assert "code" in out
