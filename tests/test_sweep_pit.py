import pickle
from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import BacktestError
from algua.backtest.sweep import sweep
from algua.contracts.types import ExecutionContract
from algua.data.serve import StoreBackedFundamentalsProvider, StoreBackedNewsProvider
from algua.data.store import DataStore
from algua.portfolio.construction import get_construction_policy
from algua.strategies.base import LoadedStrategy, StrategyConfig

START, END = datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC)


def _news_signal(view, params, news):
    # Module-level (picklable) so the parallel sweep workers can receive it.
    return pd.Series(dtype="float64")


def _news_strategy():
    return LoadedStrategy(
        config=StrategyConfig(
            name="news_sweep", universe=["AAPL", "MSFT", "NVDA"],
            execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
            params={"window_days": 5}, construction="equal_weight_positive", needs_news=True),
        news_signal_fn=_news_signal,
        construct_fn=get_construction_policy("equal_weight_positive"))


def _news_provider(tmp_path):
    store = DataStore(tmp_path)
    raw = pd.DataFrame([{"source": "r", "article_id": "a1", "symbols": ["AAPL"],
                         "published_at": "2023-02-01T00:00:00Z",
                         "knowable_at": "2023-02-01T00:00:00Z", "headline": "h"}])
    rec = store.ingest_news(provider="t", as_of="2023-03-01T00:00:00Z", frame=raw)
    return StoreBackedNewsProvider(store, rec.snapshot_id), rec.snapshot_id


def test_provider_is_picklable(tmp_path):
    prov, _ = _news_provider(tmp_path)
    pickle.loads(pickle.dumps(prov))
    fprov = StoreBackedFundamentalsProvider(DataStore(tmp_path), "x")
    pickle.loads(pickle.dumps(fprov))


def test_sweep_parallel_with_news_provider_stamps_snapshot(tmp_path):
    prov, sid = _news_provider(tmp_path)
    res = sweep(_news_strategy(), SyntheticProvider(seed=1), START, END,
                grid={"window_days": [3, 5]}, news_provider=prov)
    assert res.news_snapshot == sid
    assert res.fundamentals_snapshot is None
    assert res.n_combos == 2


def test_sweep_without_provider_fails_closed():
    with pytest.raises(BacktestError, match="needs_news"):
        sweep(_news_strategy(), SyntheticProvider(seed=1), START, END,
              grid={"window_days": [3, 5]})
