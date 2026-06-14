"""Tests: backtest sweep + walk-forward CLI accept --fundamentals/--news-snapshot (Task 5)."""
from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from algua.cli.main import app
from algua.data.store import DataStore

runner = CliRunner()


def _seed(data_dir):
    """Seed bars + news into a temp DataStore; return (bars_snapshot_id, news_snapshot_id)."""
    store = DataStore(data_dir)
    # 20 days so walk-forward with --windows 2 has enough bars (train=16, 8/window > 5)
    idx = pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC")
    rows = [[t, s, 10.0, 10.0, 10.0, 10.0, 10.0, 1000.0]
            for s in ["AAPL", "MSFT", "NVDA"] for t in idx]
    bars = pd.DataFrame(rows, columns=["ts", "symbol", "open", "high", "low", "close",
                                       "adj_close", "volume"])
    brec = store.ingest_bars(provider="t", symbols=["AAPL", "MSFT", "NVDA"], start="2025-01-01",
                             end="2025-01-21", as_of="2025-02-01T00:00:00Z", source="t", frame=bars)
    # News articles within the backtest window so the signal can count coverage.
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


def test_backtest_sweep_with_news_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    bid, nid = _seed(tmp_path)
    res = runner.invoke(app, ["backtest", "sweep", "news_coverage_tilt",
                              "--snapshot", bid, "--news-snapshot", nid,
                              "--start", "2025-01-01", "--end", "2025-01-21",
                              "--windows", "2", "--param", "window_days=3,5"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["ok"] is True
    assert payload["news_snapshot"] == nid


def test_backtest_walkforward_with_news_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    bid, nid = _seed(tmp_path)
    res = runner.invoke(app, ["backtest", "walk-forward", "news_coverage_tilt",
                              "--snapshot", bid, "--news-snapshot", nid,
                              "--start", "2025-01-01", "--end", "2025-01-21",
                              "--windows", "2"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["ok"] is True


def test_sweep_news_snapshot_on_non_news_strategy_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    bid, nid = _seed(tmp_path)
    res = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum",
                              "--snapshot", bid, "--news-snapshot", nid,
                              "--start", "2025-01-01", "--end", "2025-01-21",
                              "--windows", "2"])
    assert res.exit_code != 0
    assert "needs_news" in res.output
