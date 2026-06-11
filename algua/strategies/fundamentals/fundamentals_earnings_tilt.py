"""Earnings-yield tilt: SIGNAL = latest KNOWN diluted EPS per symbol; CONSTRUCTION = equal-weight
the positive-score names. A minimal demonstration of the as-of fundamentals lane (issue #132)."""
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
    construction="equal_weight_positive",
    needs_fundamentals=True,
)


def signal(view: pd.DataFrame, params: dict[str, Any], fundamentals: pd.DataFrame) -> pd.Series:
    """Latest-known value of `metric` per symbol (the score). equal_weight_positive then holds the
    names with a positive score, equal weight."""
    metric = str(params["metric"])
    rows = fundamentals[fundamentals["metric"] == metric]
    if rows.empty:
        return pd.Series(dtype="float64")
    return rows.groupby("symbol")["value"].last()
