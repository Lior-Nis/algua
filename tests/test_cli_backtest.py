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
    assert payload["ok"] is True  # success envelope discriminator
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


def test_backtest_run_unknown_snapshot_is_json_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    # valid-format but absent snapshot id -> JSON error, not a traceback
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                                 "--snapshot", "deadbeefdeadbeef"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False


def _ingest_pit_universe(tmp_path):
    """Ingest a time-varying universe `pit_core`: AAPL/MSFT from 2022, NVDA added 2023.

    Returns the two snapshot ids in effective-date order.
    """
    from algua.data.store import DataStore

    store = DataStore(tmp_path)
    first = store.ingest_universe(
        universe="pit_core", symbols=["AAPL", "MSFT"], effective_date="2022-01-01",
        as_of="2022-01-02T00:00:00+00:00", source="test",
    )
    second = store.ingest_universe(
        universe="pit_core", symbols=["AAPL", "MSFT", "NVDA"], effective_date="2023-01-01",
        as_of="2023-01-02T00:00:00+00:00", source="test",
    )
    return first.snapshot_id, second.snapshot_id


def test_backtest_run_with_universe_threads_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    snap = _ingest_snapshot(tmp_path)
    first_u, second_u = _ingest_pit_universe(tmp_path)

    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                                 "--snapshot", snap, "--universe", "pit_core",
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["universe_name"] == "pit_core"
    eff = [s["effective_date"] for s in payload["universe_snapshots"]]
    assert eff == ["2022-01-01", "2023-01-01"]
    assert {s["snapshot_id"] for s in payload["universe_snapshots"]} == {first_u, second_u}
    # The bars snapshot_id is a SEPARATE provenance dimension — still the bars snapshot.
    assert payload["snapshot_id"] == snap


def test_backtest_run_without_universe_has_null_universe_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    snap = _ingest_snapshot(tmp_path)
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                                 "--snapshot", snap,
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["universe_name"] is None
    assert payload["universe_snapshots"] is None


def test_backtest_walk_forward_with_universe_threads_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    snap = _ingest_snapshot(tmp_path)
    _ingest_pit_universe(tmp_path)
    result = runner.invoke(app, ["backtest", "walk-forward", "cross_sectional_momentum",
                                 "--snapshot", snap, "--universe", "pit_core",
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["universe_name"] == "pit_core"
    assert [s["effective_date"] for s in payload["universe_snapshots"]] == \
        ["2022-01-01", "2023-01-01"]


def test_backtest_sweep_with_universe_threads_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    snap = _ingest_snapshot(tmp_path)
    _ingest_pit_universe(tmp_path)
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum",
                                 "--snapshot", snap, "--universe", "pit_core",
                                 "--param", "lookback=20,40",
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["universe_name"] == "pit_core"
    assert [s["effective_date"] for s in payload["universe_snapshots"]] == \
        ["2022-01-01", "2023-01-01"]


def test_backtest_run_unknown_universe_is_json_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    snap = _ingest_snapshot(tmp_path)
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                                 "--snapshot", snap, "--universe", "does_not_exist",
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
