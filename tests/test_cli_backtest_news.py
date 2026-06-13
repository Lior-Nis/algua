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
    # News articles within the backtest window so the signal can count coverage.
    # published_at within [2025-01-05, 2025-01-09] => within 5-day window_days of last bar
    # knowable_at == published_at (immediate publication), as_of after all knowable_at
    news = pd.DataFrame([
        {
            "source": "reuters",
            "article_id": "art-001",
            "symbols": ["AAPL", "MSFT"],
            "published_at": pd.Timestamp("2025-01-06T12:00:00Z"),
            "knowable_at": pd.Timestamp("2025-01-06T12:00:00Z"),
            "headline": "AAPL and MSFT report strong quarter",
        },
        {
            "source": "reuters",
            "article_id": "art-002",
            "symbols": ["NVDA"],
            "published_at": pd.Timestamp("2025-01-07T09:00:00Z"),
            "knowable_at": pd.Timestamp("2025-01-07T09:00:00Z"),
            "headline": "NVDA sets new GPU record",
        },
        {
            "source": "reuters",
            "article_id": "art-003",
            "symbols": ["AAPL"],
            "published_at": pd.Timestamp("2025-01-08T15:00:00Z"),
            "knowable_at": pd.Timestamp("2025-01-08T15:00:00Z"),
            "headline": "AAPL unveils new product line",
        },
    ])
    nrec = store.ingest_news(provider="reuters", as_of="2025-02-01T00:00:00Z", frame=news)
    return brec.snapshot_id, nrec.snapshot_id


def test_backtest_run_with_news_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    bid, nid = _seed(tmp_path)
    res = runner.invoke(app, ["backtest", "run", "news_coverage_tilt",
                              "--snapshot", bid, "--news-snapshot", nid,
                              "--start", "2025-01-01", "--end", "2025-01-10"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["ok"] is True
    assert payload["news_snapshot"] == nid


def test_news_snapshot_on_non_news_strategy_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    bid, nid = _seed(tmp_path)
    res = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                              "--snapshot", bid, "--news-snapshot", nid,
                              "--start", "2025-01-01", "--end", "2025-01-10"])
    assert res.exit_code != 0
    assert "needs_news" in res.output


def test_fundamentals_snapshot_on_non_fundamentals_strategy_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    bid, nid = _seed(tmp_path)  # reuse the news seed; we only need a bars snapshot id here
    res = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                              "--snapshot", bid, "--fundamentals-snapshot", nid,
                              "--start", "2025-01-01", "--end", "2025-01-10"])
    assert res.exit_code != 0
    assert "needs_fundamentals" in res.output
