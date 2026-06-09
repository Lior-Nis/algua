import json

import pandas as pd
from typer.testing import CliRunner

from algua.cli.main import app
from algua.data.store import DataStore

runner = CliRunner()


def _seed(data_dir):
    store = DataStore(data_dir)
    idx = pd.date_range("2025-01-01", periods=9, freq="D", tz="UTC")
    rows = [[t, s, 10.0, 10.0, 10.0, 10.0, 10.0, 1000.0]
            for s in ["AAPL", "MSFT", "NVDA"] for t in idx]
    bars = pd.DataFrame(rows, columns=["ts", "symbol", "open", "high", "low", "close",
                                       "adj_close", "volume"])
    brec = store.ingest_bars(provider="t", symbols=["AAPL", "MSFT", "NVDA"], start="2025-01-01",
                             end="2025-01-10", as_of="2025-02-01T00:00:00Z", source="t", frame=bars)
    funds = pd.DataFrame(
        [["AAPL", "2024-12-31", "eps_diluted", 5.0, "2024-12-31T00:00:00Z", "v"]],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"])
    frec = store.ingest_fundamentals(provider="v", symbols=["AAPL", "MSFT", "NVDA"],
                                     as_of="2025-01-01T00:00:00Z", source="v", frame=funds)
    return brec.snapshot_id, frec.snapshot_id


def test_backtest_run_with_fundamentals_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    bid, fid = _seed(tmp_path)
    res = runner.invoke(app, ["backtest", "run", "fundamentals_earnings_tilt",
                              "--snapshot", bid, "--fundamentals-snapshot", fid,
                              "--start", "2025-01-01", "--end", "2025-01-10"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["ok"] is True
    assert payload["fundamentals_snapshot"] == fid
