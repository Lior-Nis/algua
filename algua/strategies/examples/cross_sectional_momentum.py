"""Cross-sectional momentum: hold the top-k trailing-return names, equal weight."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

CONFIG = StrategyConfig(
    name="cross_sectional_momentum",
    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 60, "top_k": 3},
)


def compute_weights(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    lookback = int(params["lookback"])
    top_k = int(params["top_k"])
    # Wide adj_close (index=timestamp, columns=symbol) from the point-in-time view.
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")
    # Last-row momentum only — avoid materializing the full momentum matrix per bar.
    scores = (wide.iloc[-1] / wide.iloc[-1 - lookback] - 1.0).dropna()
    winners = scores.sort_values(ascending=False).head(top_k).index
    if len(winners) == 0:
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / len(winners), index=winners)
