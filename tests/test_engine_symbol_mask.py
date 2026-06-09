"""FIX C: engine masks f_asof to known symbols even in static-universe mode."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.backtest.engine import _decision_weights
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig


class _RecordingStrategy:
    """Captures the symbols in the fundamentals frame passed to compute_weights."""

    def __init__(self) -> None:
        self.seen_symbols: list[set[str]] = []

    def __call__(
        self, view: pd.DataFrame, params: dict[str, Any], fundamentals: pd.DataFrame
    ) -> pd.Series:
        self.seen_symbols.append(set(fundamentals["symbol"].unique()))
        # return flat (empty) weights
        return pd.Series(dtype="float64")


def _toy_adj(symbols: list[str], n: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(1.0, index=idx, columns=symbols)


def _toy_bars(symbols: list[str], n: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    rows = []
    for sym in symbols:
        for t in idx:
            rows.append([t, sym, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
    df = pd.DataFrame(rows, columns=["timestamp", "symbol", "open", "high", "low",
                                     "close", "adj_close", "volume"])
    return df.set_index("timestamp").sort_index()


def _toy_fundamentals(symbols: list[str]) -> pd.DataFrame:
    from algua.data.fundamentals_schema import to_fundamentals_schema
    rows = [
        [sym, "2024-12-31", "eps_diluted", 1.0, "2024-12-31T00:00:00Z", "v"]
        for sym in symbols
    ]
    raw = pd.DataFrame(rows, columns=[
        "symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source",
    ])
    return to_fundamentals_schema(raw)


def _make_strategy(fn: Any) -> LoadedStrategy:
    cfg = StrategyConfig(
        name="test_strat",
        universe=["AAPL", "MSFT"],  # only 2 symbols
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        needs_fundamentals=True,
    )
    return LoadedStrategy(config=cfg, fundamentals_fn=fn)


def test_out_of_universe_symbol_masked_in_static_mode():
    """In static mode (no universe_by_date), a fundamentals frame with an extra symbol
    must be masked so the strategy never sees the out-of-universe symbol."""
    recorder = _RecordingStrategy()
    strat = _make_strategy(recorder)

    # bars + adj for AAPL and MSFT only
    universe_symbols = ["AAPL", "MSFT"]
    adj = _toy_adj(universe_symbols, n=3)
    bars = _toy_bars(universe_symbols, n=3)

    # fundamentals include NVDA (not in universe)
    funds = _toy_fundamentals(["AAPL", "MSFT", "NVDA"])

    # No universe_by_date → static mode
    _decision_weights(strat, bars, adj, universe_by_date=None, fundamentals=funds)

    # Every call to compute_weights must NOT have seen NVDA
    for seen in recorder.seen_symbols:
        assert "NVDA" not in seen, (
            f"NVDA (out-of-universe) leaked into fundamentals frame: {seen}"
        )
