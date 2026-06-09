from datetime import datetime

import pandas as pd
import pytest

from algua.backtest.engine import BacktestError, _fundamentals_as_of, run
from algua.data.serve import StoreBackedFundamentalsProvider, StoreBackedProvider
from algua.data.store import DataStore
from algua.strategies.loader import load_strategy


def _funds():
    return pd.DataFrame(
        [
            # original report, then a restatement that flips the sign later
            ["AAPL", "2025-03-31", "eps_diluted", 1.0, "2025-05-01T13:00:00Z", "v"],
            ["AAPL", "2025-03-31", "eps_diluted", -1.0, "2025-08-01T13:00:00Z", "v"],
        ],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"],
    )


def test_as_of_mask_picks_latest_knowable_and_restatement_flips():
    frame = __import__("algua.data.fundamentals_schema", fromlist=["to_fundamentals_schema"]) \
        .to_fundamentals_schema(_funds())
    before = _fundamentals_as_of(frame, pd.Timestamp("2025-06-01", tz="UTC"))
    assert before["value"].iloc[0] == 1.0  # original, restatement not yet knowable
    after = _fundamentals_as_of(frame, pd.Timestamp("2025-09-01", tz="UTC"))
    assert after["value"].iloc[0] == -1.0  # restated value now knowable


def test_as_of_mask_rejects_naive_t():
    frame = __import__("algua.data.fundamentals_schema", fromlist=["to_fundamentals_schema"]) \
        .to_fundamentals_schema(_funds())
    with pytest.raises(BacktestError, match="tz-aware"):
        _fundamentals_as_of(frame, pd.Timestamp("2025-06-01"))


def test_as_of_mask_empty_before_any_knowable():
    frame = __import__("algua.data.fundamentals_schema", fromlist=["to_fundamentals_schema"]) \
        .to_fundamentals_schema(_funds())
    out = _fundamentals_as_of(frame, pd.Timestamp("2025-01-01", tz="UTC"))
    assert len(out) == 0


def test_run_fails_closed_when_fundamentals_strategy_lacks_provider(tmp_path):
    # bars snapshot for the universe
    store = DataStore(tmp_path)
    bars = _toy_bars()
    brec = store.ingest_bars(provider="t", symbols=["AAPL", "MSFT", "NVDA"], start="2025-01-01",
                             end="2025-01-10", as_of="2025-02-01T00:00:00Z", source="t", frame=bars)
    strat = load_strategy("fundamentals_earnings_tilt")
    with pytest.raises(BacktestError, match="fundamentals"):
        run(strat, StoreBackedProvider(store, brec.snapshot_id),
            datetime(2025, 1, 1), datetime(2025, 1, 10))


def _toy_bars():
    """Return a flat bars frame with a 'ts' column suitable for store.ingest_bars."""
    idx = pd.date_range("2025-01-01", periods=9, freq="D", tz="UTC")
    rows = []
    for s in ["AAPL", "MSFT", "NVDA"]:
        for t in idx:
            rows.append([t, s, 10.0, 10.0, 10.0, 10.0, 10.0, 1000.0])
    return pd.DataFrame(rows, columns=["ts", "symbol", "open", "high", "low", "close",
                                       "adj_close", "volume"])


def test_run_with_fundamentals_stamps_snapshot(tmp_path):
    store = DataStore(tmp_path)
    bars = _toy_bars()
    brec = store.ingest_bars(provider="t", symbols=["AAPL", "MSFT", "NVDA"], start="2025-01-01",
                             end="2025-01-10", as_of="2025-02-01T00:00:00Z", source="t", frame=bars)
    funds = pd.DataFrame(
        [["AAPL", "2024-12-31", "eps_diluted", 5.0, "2024-12-31T00:00:00Z", "v"]],
        columns=["symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source"])
    frec = store.ingest_fundamentals(provider="v", symbols=["AAPL", "MSFT", "NVDA"],
                                     as_of="2025-01-01T00:00:00Z", source="v", frame=funds)
    strat = load_strategy("fundamentals_earnings_tilt")
    result = run(strat, StoreBackedProvider(store, brec.snapshot_id),
                 datetime(2025, 1, 1), datetime(2025, 1, 10),
                 fundamentals_provider=StoreBackedFundamentalsProvider(store, frec.snapshot_id))
    assert result.fundamentals_snapshot == frec.snapshot_id
