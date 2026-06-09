"""Earnings-yield tilt: among the universe, hold (equal-weight) the names whose latest KNOWN
diluted EPS is positive. A minimal demonstration of the as-of fundamentals lane (issue #132)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

CONFIG = StrategyConfig(
    name="fundamentals_earnings_tilt",
    universe=["AAPL", "MSFT", "NVDA"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"metric": "eps_diluted"},
    needs_fundamentals=True,
)


def compute_weights(
    view: pd.DataFrame, params: dict[str, Any], fundamentals: pd.DataFrame
) -> pd.Series:
    metric = str(params["metric"])
    rows = fundamentals[fundamentals["metric"] == metric]
    if rows.empty:
        return pd.Series(dtype="float64")
    # latest known value per symbol (frame is already as-of-masked + canonically sorted by the
    # engine, so the last row per symbol is the most-recently-knowable)
    latest = rows.groupby("symbol")["value"].last()
    winners = latest[latest > 0.0].index
    if len(winners) == 0:
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / len(winners), index=list(winners))
