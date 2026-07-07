"""Regression: the trials x windows OOS-Sharpe matrix NEVER reaches any SweepResult surface (#467
Gate-2 item 5 / R2-2).

Because the matrix is NOT a field on SweepResult — it rides only as the second element of
sweep_with_matrix()'s tuple — nothing that merely holds a result object (attribute access, to_dict
/ JSON, the --summary projection, the sweep_task payload, or the MLflow log_sweep serialization)
can reach the per-combo per-window Sharpes.
"""

import json

import pytest
from typer.testing import CliRunner

from algua.backtest._sample import SyntheticProvider
from algua.backtest.sweep import SweepResult, sweep, sweep_with_matrix
from algua.cli.backtest_cmd import _SWEEP_SUMMARY_KEYS
from algua.cli.main import app
from tests.test_sweep import END, START, _momentum

runner = CliRunner()

# Any key/attr that would leak the per-combo per-window matrix.
_MATRIX_NAMES = ("matrix", "trial_window_sharpes", "window_sharpes", "trial_matrix")


def _no_matrix_key(d: dict) -> None:
    for k in _MATRIX_NAMES:
        assert k not in d, f"matrix leaked via key {k!r}: {sorted(d)}"


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def _result() -> SweepResult:
    return sweep(
        _momentum(), SyntheticProvider(seed=3), START, END,
        grid={"lookback": [20, 40]}, windows=4, holdout_frac=0.2,
    )


def test_sweepresult_has_no_matrix_field_or_dict_key():
    res = _result()
    for attr in _MATRIX_NAMES:
        assert not hasattr(res, attr)
    _no_matrix_key(res.to_dict())


def test_summary_projection_has_no_matrix_key():
    # The --summary keep-list can never surface a matrix key.
    for k in _MATRIX_NAMES:
        assert k not in _SWEEP_SUMMARY_KEYS


def test_log_sweep_serialization_source_has_no_matrix_key():
    # log_sweep draws every logged field from the SweepResult; its serializable view is to_dict(),
    # which carries no matrix key, so nothing the tracker persists can leak it.
    _no_matrix_key(_result().to_dict())


def test_sweep_task_cli_payload_has_no_matrix_key():
    # A real `backtest sweep` (= sweep_task) payload.
    out = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                              "--start", "2022-01-01", "--end", "2023-12-31",
                              "--param", "lookback=20,40", "--windows", "4"])
    assert out.exit_code == 0, out.stdout
    payload = json.loads(out.stdout)
    _no_matrix_key(payload)
    # The ranked list rows carry params/score/stability but never a raw per-window matrix row.
    for row in payload.get("ranked", []):
        _no_matrix_key(row)


def test_sweep_with_matrix_returns_matrix_only_as_second_tuple_element():
    res, matrix = sweep_with_matrix(
        _momentum(), SyntheticProvider(seed=3), START, END,
        grid={"lookback": [20, 40]}, windows=4, holdout_frac=0.2,
    )
    assert isinstance(matrix, list) and len(matrix) == 2 and all(matrix)
    # ...and the SweepResult in slot 0 still exposes no way to reach it.
    for attr in _MATRIX_NAMES:
        assert not hasattr(res, attr)
