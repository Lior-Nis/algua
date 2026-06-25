# tests/test_cli_backtest_series.py
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from algua.backtest.engine import BacktestError
from algua.backtest.result import BacktestResult
from algua.cli.backtest_cmd import emit_series_file
from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def _result(returns):
    return BacktestResult(
        strategy="mom", metrics={"sharpe": 1.0}, config_hash="cfg",
        data_source="SyntheticProvider", timeframe="1d",
        period={"start": "2023-01-01", "end": "2023-01-02"},
        seed=0, code_hash="abc", dependency_hash="dep", returns=returns,
    )


def test_emit_series_file_fails_closed_on_none(tmp_path):
    with pytest.raises(BacktestError):
        emit_series_file(_result(None), tmp_path / "s.parquet")
    assert not (tmp_path / "s.parquet").exists()


def test_emit_series_file_fails_closed_on_empty(tmp_path):
    empty = pd.Series([], dtype=float, index=pd.to_datetime([]))
    with pytest.raises(BacktestError):
        emit_series_file(_result(empty), tmp_path / "s.parquet")
    assert not (tmp_path / "s.parquet").exists()


def test_emit_series_file_fails_closed_on_nan(tmp_path):
    """A series containing NaN must be rejected as non-finite (finding #1, GATE-2)."""
    idx = pd.to_datetime(["2023-01-01", "2023-01-02"])
    nan_series = pd.Series([0.01, np.nan], index=idx)
    with pytest.raises(BacktestError, match="no finite return series"):
        emit_series_file(_result(nan_series), tmp_path / "s.parquet")
    assert not (tmp_path / "s.parquet").exists()


def test_emit_series_file_fails_closed_on_inf(tmp_path):
    """A series containing inf must be rejected as non-finite (finding #1, GATE-2)."""
    idx = pd.to_datetime(["2023-01-01", "2023-01-02"])
    inf_series = pd.Series([0.01, np.inf], index=idx)
    with pytest.raises(BacktestError, match="no finite return series"):
        emit_series_file(_result(inf_series), tmp_path / "s.parquet")
    assert not (tmp_path / "s.parquet").exists()


def test_emit_series_file_bad_path_json_contract(tmp_path):
    """An unwritable path must produce a BacktestError (stays in JSON contract, finding #3)."""
    import os
    idx = pd.to_datetime(["2023-01-01", "2023-01-02"])
    good_series = pd.Series([0.01, 0.02], index=idx)
    # Make the parent directory read-only so mkstemp inside write_bytes_atomic fails.
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    os.chmod(ro_dir, 0o555)
    try:
        with pytest.raises(BacktestError, match="failed to write series"):
            emit_series_file(_result(good_series), ro_dir / "s.parquet")
    finally:
        os.chmod(ro_dir, 0o755)  # restore so tmp_path cleanup succeeds


def test_emit_series_file_writes_and_descriptor(tmp_path):
    idx = pd.to_datetime(["2023-01-01", "2023-01-02"])
    desc = emit_series_file(_result(pd.Series([0.01, 0.02], index=idx)), tmp_path / "s.parquet")
    assert desc["n"] == 2
    assert desc["config_hash"] == "cfg" and desc["start"] == "2023-01-01"
    # Finding #7: all identity fields must be present in the stdout descriptor
    for key in ("universe_name", "fundamentals_snapshot", "news_snapshot", "delisting_snapshot"):
        assert key in desc, f"descriptor missing key: {key}"
    meta = pq.read_schema(pa.BufferReader((tmp_path / "s.parquet").read_bytes())).metadata
    assert b"algua.result_json" in meta


def test_cli_emit_series_demo(tmp_path):
    out = tmp_path / "series.parquet"
    res = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                              "--start", "2023-01-01", "--end", "2023-12-31",
                              "--emit-series", str(out)])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["series"]["path"] == str(out)
    df = pd.read_parquet(out)
    assert list(df.columns) == ["date", "ret"]
    assert payload["series"]["n"] == len(df)
    assert len(df) > 0


def test_cli_emit_series_deterministic(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    for out in (a, b):
        r = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                                "--start", "2023-01-01", "--end", "2023-12-31",
                                "--emit-series", str(out)])
        assert r.exit_code == 0, r.stdout
    assert a.read_bytes() == b.read_bytes()


def test_cli_no_emit_series_has_no_series_key(tmp_path):
    res = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                              "--start", "2023-01-01", "--end", "2023-12-31"])
    assert res.exit_code == 0, res.stdout
    assert "series" not in json.loads(res.stdout)
