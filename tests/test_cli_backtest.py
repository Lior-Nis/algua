import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def test_backtest_run_demo_emits_metrics():
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                                 "--demo", "--start", "2023-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["strategy"] == "cross_sectional_momentum"
    assert "sharpe" in payload["metrics"]


def test_backtest_run_register_advances_registry():
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                                 "--demo", "--start", "2023-01-01", "--end", "2023-12-31",
                                 "--register"])
    assert result.exit_code == 0, result.stdout
    show = runner.invoke(app, ["registry", "show", "cross_sectional_momentum"])
    assert json.loads(show.stdout)["stage"] == "backtested"


def test_unknown_strategy_is_json_error():
    result = runner.invoke(app, ["backtest", "run", "nope", "--demo"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def _ingest_snapshot(tmp_path):
    """Ingest synthetic momentum-universe bars; return the snapshot id."""
    from datetime import UTC, datetime

    from algua.backtest._sample import SyntheticProvider
    from algua.data.store import DataStore

    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
    start, end = datetime(2022, 1, 1, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC)
    bars = SyntheticProvider(seed=0).get_bars(symbols, start, end, "1d")
    frame = bars.reset_index().rename(columns={"timestamp": "ts"})
    rec = DataStore(tmp_path).ingest_bars(
        provider="synthetic", symbols=symbols, start="2022-01-01", end="2023-12-31",
        as_of="2024-01-01", source="test", frame=frame, timeframe="1d", adjustment="none",
    )
    return rec.snapshot_id


def test_backtest_run_on_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    snap = _ingest_snapshot(tmp_path)
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                                 "--snapshot", snap,
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["snapshot_id"] == snap
    assert "sharpe" in payload["metrics"]


def test_backtest_run_requires_a_data_source(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_backtest_run_rejects_both_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                                 "--demo", "--snapshot", "x"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
