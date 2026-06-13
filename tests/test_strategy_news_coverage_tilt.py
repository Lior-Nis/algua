from datetime import datetime

import pandas as pd

from algua.backtest.engine import run
from algua.data.serve import StoreBackedNewsProvider, StoreBackedProvider
from algua.data.store import DataStore
from algua.strategies.loader import list_strategies, load_strategy
from algua.strategies.news import news_coverage_tilt


def test_load_news_coverage_tilt_binds_news_lane():
    strat = load_strategy("news_coverage_tilt")
    assert strat.config.needs_news is True
    assert strat.news_signal_fn is not None
    assert strat.signal_fn is None
    assert "news_coverage_tilt" in list_strategies()


def _view(asof: str):
    """A minimal bars view: tz-aware UTC `timestamp` index + symbol/adj_close columns, mirroring the
    real engine slice (algua.data.schema.to_bar_schema sets this index)."""
    idx = pd.date_range(end=asof, periods=3, freq="D", tz="UTC")
    rows = []
    for s in ["AAPL", "MSFT", "NVDA"]:
        for t in idx:
            rows.append([t, s, 10.0])
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "adj_close"]).set_index("timestamp")


def _news_asof():
    """An as-of news frame (the canonical NEWS_COLUMNS), as the engine's mask hands it: AAPL has
    2 in-window articles, MSFT has 1, NVDA has 1 article published OUTSIDE the window."""
    from algua.data.news_schema import to_news_schema

    def row(article_id, symbol, published_at):
        return {
            "source": "vendor",
            "article_id": article_id,
            "symbol": symbol,
            "published_at": pd.Timestamp(published_at, tz="UTC"),
            "knowable_at": pd.Timestamp(published_at, tz="UTC"),
            "headline": f"h-{article_id}",
            "url": pd.NA,
            "body": pd.NA,
            "retracted": False,
        }

    frame = pd.DataFrame(
        [
            row("a1", "AAPL", "2025-01-09"),
            row("a2", "AAPL", "2025-01-08"),
            row("a3", "MSFT", "2025-01-09"),
            row("a4", "NVDA", "2025-01-01"),  # older than a 5-day window of the 2025-01-10 bar
        ]
    )
    return to_news_schema(frame)


def test_signal_counts_distinct_in_window_articles():
    view = _view("2025-01-10")
    news = _news_asof()
    scores = news_coverage_tilt.signal(view, {"window_days": 5}, news)
    assert scores.loc["AAPL"] == 2.0
    assert scores.loc["MSFT"] == 1.0
    # NVDA's only article is older than the window -> not counted (absent or zero).
    assert "NVDA" not in scores.index


def test_signal_empty_news_returns_empty():
    view = _view("2025-01-10")
    empty = _news_asof().iloc[0:0]
    scores = news_coverage_tilt.signal(view, {"window_days": 5}, empty)
    assert scores.empty


def _toy_bars():
    idx = pd.date_range("2025-01-01", periods=9, freq="D", tz="UTC")
    rows = []
    for s in ["AAPL", "MSFT", "NVDA"]:
        for t in idx:
            rows.append([t, s, 10.0, 11.0, 9.0, 10.0, 10.0, 1000.0])
    return pd.DataFrame(rows, columns=["ts", "symbol", "open", "high", "low", "close",
                                       "adj_close", "volume"])


def _e2e_news():
    """Per-article ingest input (explode_news_symbols form): a symbols list per article."""
    def art(article_id, symbols, knowable):
        return {
            "source": "vendor",
            "article_id": article_id,
            "symbols": symbols,
            "published_at": pd.Timestamp(knowable, tz="UTC"),
            "knowable_at": pd.Timestamp(knowable, tz="UTC"),
            "headline": f"h-{article_id}",
        }

    return pd.DataFrame(
        [
            art("a1", ["AAPL"], "2025-01-03T00:00:00Z"),
            art("a2", ["AAPL"], "2025-01-05T00:00:00Z"),
            art("a3", ["MSFT"], "2025-01-06T00:00:00Z"),
        ]
    )


def test_run_with_news_stamps_snapshot_and_finite_sharpe(tmp_path):
    store = DataStore(tmp_path)
    bars = _toy_bars()
    brec = store.ingest_bars(provider="t", symbols=["AAPL", "MSFT", "NVDA"], start="2025-01-01",
                             end="2025-01-10", as_of="2025-02-01T00:00:00Z", source="t", frame=bars)
    nrec = store.ingest_news(provider="vendor", as_of="2025-01-31T00:00:00Z", frame=_e2e_news())
    strat = load_strategy("news_coverage_tilt")
    result = run(strat, StoreBackedProvider(store, brec.snapshot_id),
                 datetime(2025, 1, 1), datetime(2025, 1, 10),
                 news_provider=StoreBackedNewsProvider(store, nrec.snapshot_id))
    assert result.news_snapshot == nrec.snapshot_id
    assert "sharpe" in result.metrics
    import math
    assert math.isfinite(result.metrics["sharpe"])
