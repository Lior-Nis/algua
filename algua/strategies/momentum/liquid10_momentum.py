"""Cross-sectional momentum over the liquid10 basket — test vehicle for the paper loop."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.features.alphas import xs_trailing_return
from algua.strategies.base import StrategyConfig

GENERATED_BY = "agent"

CONFIG = StrategyConfig(
    name="liquid10_momentum",
    universe=["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "XOM", "PG", "KO", "WMT"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 60},
    construction="top_k_equal_weight",
    construction_params={"top_k": 5},
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Trailing `lookback`-bar return per symbol (the alpha score), via the catalogued
    `xs_trailing_return` factor — same signal as cross_sectional_momentum, varied params."""
    return xs_trailing_return(view, params)


def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Vectorized SCORES twin of `signal` (per-bar signal stays canonical). Full trailing-return
    matrix in one shot; rows without `lookback` history are all-NaN (construction drops NaN)."""
    lookback = int(params["lookback"])
    wide = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    return wide / wide.shift(lookback) - 1.0
