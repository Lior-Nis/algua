"""Integration tests for delisting overlay wired into simulate() — Task 5 of #212."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from algua.backtest.engine import BacktestError, simulate
from algua.backtest.delisting import DelistingRecord
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _passthrough(scores: pd.Series, view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Identity construction policy — signal already emits desired raw weights."""
    return scores


def _equal_weight_ab_strategy() -> LoadedStrategy:
    """Equal-weight A+B strategy over a two-symbol universe."""
    cfg = StrategyConfig(
        name="ew_ab",
        universe=["A", "B"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={},
        construction="passthrough",
    )

    def signal(view: pd.DataFrame, params: dict) -> pd.Series:
        syms = view["symbol"].unique()
        return pd.Series(1.0 / len(syms), index=sorted(syms))

    return LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=_passthrough)


def _bars() -> pd.DataFrame:
    """4-bar panel for A (all 4 bars) and B (only first 2 bars, then disappears — simulating a
    delisting after 2020-01-02). Returned in the same long-form format as SyntheticProvider:
    timestamp-indexed, 'symbol' column plus OHLCAV columns."""
    # 4 calendar days — daily backtest doesn't require trading-day calendar
    idx = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    rows = []
    # A: 4 bars
    for ts, px in zip(idx, [10.0, 11.0, 12.0, 13.0], strict=True):
        rows.append({
            "timestamp": ts, "symbol": "A",
            "open": px, "high": px, "low": px,
            "close": px, "adj_close": px, "volume": 100.0,
        })
    # B: only 2 bars (delists after 2020-01-02)
    for ts, px in zip(idx[:2], [10.0, 11.0], strict=True):
        rows.append({
            "timestamp": ts, "symbol": "B",
            "open": px, "high": px, "low": px,
            "close": px, "adj_close": px, "volume": 100.0,
        })
    return pd.DataFrame(rows).set_index("timestamp").sort_values(["timestamp", "symbol"])


class _FakeProvider:
    """Thin wrapper around a pre-built bars frame, matching the DataProvider protocol."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def get_bars(self, symbols, start, end, timeframe):  # noqa: ANN001
        return self._frame[self._frame["symbol"].isin(set(symbols))]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_delisting_exit_realizes_and_keeps_equity_finite() -> None:
    """When a delisting record is supplied for B, simulate() must:
    - include B in forced_exits
    - produce finite returns throughout (no 0*NaN poison)
    - show zero position for B from the exit bar onward
    """
    strat = _equal_weight_ab_strategy()
    provider = _FakeProvider(_bars())
    recs: dict[str, list[DelistingRecord]] = {
        "B": [DelistingRecord(date(2020, 1, 2), 5.0, "vendor")],
    }
    pf, weights, forced = simulate(
        strat, provider,
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 4, tzinfo=UTC),
        delisting_records=recs,
    )

    # forced_exits must mention B
    assert forced, "expected at least one forced exit"
    assert any(fe["symbol"] == "B" for fe in forced)

    # returns must be finite
    returns = pf.returns()
    assert bool(np.isfinite(returns.fillna(0.0)).all()), "non-finite returns after delisting"

    # B must carry zero position from its exit bar onward
    positions = pf.assets()
    exit_bar = pd.Timestamp("2020-01-02", tz="UTC")
    after = positions["B"].loc[positions.index >= exit_bar]
    assert bool((after.abs() < 1e-9).all()), f"residual B position after exit: {after}"


def test_held_into_gap_without_record_raises_backtest_error() -> None:
    """When B disappears mid-panel and no delisting record is supplied, simulate() must
    raise BacktestError (fail-closed: unknown terminal value)."""
    strat = _equal_weight_ab_strategy()
    provider = _FakeProvider(_bars())

    with pytest.raises(BacktestError, match="delisting record|residual"):
        simulate(
            strat, provider,
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2020, 1, 4, tzinfo=UTC),
        )
