from datetime import UTC, datetime
from typing import Any

import pandas as pd

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import _news_as_of, run
from algua.contracts.types import ExecutionContract
from algua.data.news_schema import explode_news_symbols, to_news_schema
from algua.data.serve import StoreBackedNewsProvider
from algua.data.store import DataStore
from algua.strategies.base import LoadedStrategy, StrategyConfig


def _news():
    raw = pd.DataFrame([
        {"source": "r", "article_id": "a1", "symbols": ["AAPL", "MSFT"],
         "published_at": "2023-01-01T00:00:00Z", "knowable_at": "2023-01-01T00:00:00Z",
         "headline": "h1"},
        {"source": "r", "article_id": "a1", "symbols": ["AAPL"],
         "published_at": "2023-01-01T00:00:00Z", "knowable_at": "2023-01-10T00:00:00Z",
         "headline": "h2"},
    ])
    return to_news_schema(explode_news_symbols(raw))


def test_as_of_before_drop_shows_both():
    out = _news_as_of(_news(), pd.Timestamp("2023-01-05T00:00:00Z"))
    assert set(out["symbol"]) == {"AAPL", "MSFT"}
    assert not out["retracted"].any()  # tombstones never surface in the as-of view


def test_as_of_after_drop_excludes_retracted_symbol():
    out = _news_as_of(_news(), pd.Timestamp("2023-01-15T00:00:00Z"))
    assert set(out["symbol"]) == {"AAPL"}  # MSFT's latest revision is a tombstone -> dropped


def test_as_of_excludes_future_knowable():
    out = _news_as_of(_news(), pd.Timestamp("2022-12-31T00:00:00Z"))
    assert out.empty


def test_as_of_empty_preserves_columns_and_dtypes():
    out = _news_as_of(_news(), pd.Timestamp("2022-01-01T00:00:00Z"))
    assert list(out.columns) == list(_news().columns)
    assert out["retracted"].dtype == _news()["retracted"].dtype


def _news_readd():
    # MSFT: live@Jan01, dropped(tombstone)@Jan10, re-added@Jan20
    raw = pd.DataFrame([
        {"source": "r", "article_id": "a1", "symbols": ["AAPL", "MSFT"],
         "published_at": "2023-01-01T00:00:00Z", "knowable_at": "2023-01-01T00:00:00Z",
         "headline": "h1"},
        {"source": "r", "article_id": "a1", "symbols": ["AAPL"],
         "published_at": "2023-01-01T00:00:00Z", "knowable_at": "2023-01-10T00:00:00Z",
         "headline": "h2"},
        {"source": "r", "article_id": "a1", "symbols": ["AAPL", "MSFT"],
         "published_at": "2023-01-01T00:00:00Z", "knowable_at": "2023-01-20T00:00:00Z",
         "headline": "h3"},
    ])
    return to_news_schema(explode_news_symbols(raw))


def test_as_of_readd_after_tombstone_resurfaces_symbol():
    frame = _news_readd()
    # between the tombstone (Jan10) and the re-add (Jan20): MSFT EXCLUDED
    assert set(_news_as_of(frame, pd.Timestamp("2023-01-15T00:00:00Z"))["symbol"]) == {"AAPL"}
    # after the re-add: MSFT is live again
    after = _news_as_of(frame, pd.Timestamp("2023-01-25T00:00:00Z"))
    assert set(after["symbol"]) == {"AAPL", "MSFT"}
    assert not after["retracted"].any()


START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 4, 1, tzinfo=UTC)


def _passthrough(scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Test construction policy: identity (see test_backtest_engine.py)."""
    return scores


def _ew_signal(view: pd.DataFrame, params: dict) -> pd.Series:
    syms = view["symbol"].unique()
    return pd.Series(1.0 / len(syms), index=sorted(syms))


def _news_signal(view: pd.DataFrame, params: dict, news: pd.DataFrame) -> pd.Series:
    """Minimal needs_news signal: ignores `news`, returns an empty series (no positions).
    The test only checks provenance stamping, not weights."""
    return pd.Series(dtype="float64")


def _ingest_news_snapshot(tmp_path):
    store = DataStore(tmp_path)
    raw = pd.DataFrame([
        {"source": "r", "article_id": "a1", "symbols": ["AAA", "BBB"],
         "published_at": "2024-01-02T00:00:00Z", "knowable_at": "2024-01-02T00:00:00Z",
         "headline": "h1"},
    ])
    rec = store.ingest_news(provider="t", as_of="2024-04-01T00:00:00Z", frame=raw)
    return store, rec


def test_run_stamps_news_snapshot_when_needs_news(tmp_path):
    store, rec = _ingest_news_snapshot(tmp_path)
    news_provider = StoreBackedNewsProvider(store, rec.snapshot_id)
    cfg = StrategyConfig(
        name="news_strat",
        universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={},
        construction="passthrough",
        needs_news=True,
    )
    strategy = LoadedStrategy(config=cfg, news_signal_fn=_news_signal, construct_fn=_passthrough)
    result = run(strategy, SyntheticProvider(), START, END, news_provider=news_provider)
    assert result.news_snapshot == rec.snapshot_id


def test_run_does_not_stamp_news_snapshot_for_non_news_strategy(tmp_path):
    store, rec = _ingest_news_snapshot(tmp_path)
    news_provider = StoreBackedNewsProvider(store, rec.snapshot_id)
    cfg = StrategyConfig(
        name="plain_strat",
        universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={},
        construction="passthrough",
    )
    strategy = LoadedStrategy(config=cfg, signal_fn=_ew_signal, construct_fn=_passthrough)
    # Engine ignores news_provider for a non-needs_news strategy — but it must not be stamped.
    result = run(strategy, SyntheticProvider(), START, END, news_provider=news_provider)
    assert result.news_snapshot is None
