"""FIX E: strengthen _assert_fundamentals_shape — rejects bad provider frames."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from algua.backtest.engine import BacktestError, run
from algua.data.serve import StoreBackedProvider
from algua.data.store import DataStore
from algua.strategies.loader import load_strategy


def _toy_bars_raw():
    idx = pd.date_range("2025-01-01", periods=9, freq="D", tz="UTC")
    rows = []
    for s in ["AAPL", "MSFT", "NVDA"]:
        for t in idx:
            rows.append([t, s, 10.0, 10.0, 10.0, 10.0, 10.0, 1000.0])
    return pd.DataFrame(rows, columns=["ts", "symbol", "open", "high", "low",
                                       "close", "adj_close", "volume"])


def _good_fundamentals_frame():
    """A valid, contract-shaped fundamentals frame."""
    from algua.data.fundamentals_schema import to_fundamentals_schema
    rows = [["AAPL", "2024-12-31", "eps_diluted", 5.0, "2024-12-31T00:00:00Z", "v"]]
    raw = pd.DataFrame(rows, columns=[
        "symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source",
    ])
    return to_fundamentals_schema(raw)


class _BadFundamentalsProvider:
    """A provider that returns a frame with duplicate bitemporal keys."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.snapshot_id = "fake-snapshot"

    def get_fundamentals(self, symbols: list[str], as_of: datetime) -> pd.DataFrame:
        return self._frame


def test_duplicate_bitemporal_key_raises_backtest_error(tmp_path):
    """A provider returning duplicate (symbol, fiscal_period_end, metric, knowable_at) rows
    must be rejected by _assert_fundamentals_shape as BacktestError."""
    store = DataStore(tmp_path)
    brec = store.ingest_bars(
        provider="t", symbols=["AAPL", "MSFT", "NVDA"],
        start="2025-01-01", end="2025-01-10",
        as_of="2025-02-01T00:00:00Z", source="t", frame=_toy_bars_raw(),
    )
    strat = load_strategy("fundamentals_earnings_tilt")

    # Build a frame with a duplicate bitemporal key
    good = _good_fundamentals_frame()
    dup = pd.concat([good, good], ignore_index=True)

    provider = _BadFundamentalsProvider(dup)
    with pytest.raises(BacktestError, match="duplicate"):
        run(
            strat,
            StoreBackedProvider(store, brec.snapshot_id),
            datetime(2025, 1, 1),
            datetime(2025, 1, 10),
            fundamentals_provider=provider,
        )


def test_non_float_value_raises_backtest_error(tmp_path):
    """A provider returning a non-float64 'value' column raises BacktestError."""
    store = DataStore(tmp_path)
    brec = store.ingest_bars(
        provider="t", symbols=["AAPL", "MSFT", "NVDA"],
        start="2025-01-01", end="2025-01-10",
        as_of="2025-02-01T00:00:00Z", source="t", frame=_toy_bars_raw(),
    )
    strat = load_strategy("fundamentals_earnings_tilt")

    # Build a frame with value as int64 instead of float64
    good = _good_fundamentals_frame()
    bad = good.copy()
    bad["value"] = bad["value"].astype("int64")

    provider = _BadFundamentalsProvider(bad)
    with pytest.raises(BacktestError, match="float64"):
        run(
            strat,
            StoreBackedProvider(store, brec.snapshot_id),
            datetime(2025, 1, 1),
            datetime(2025, 1, 10),
            fundamentals_provider=provider,
        )
