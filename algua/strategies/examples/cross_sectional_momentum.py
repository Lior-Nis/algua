"""Cross-sectional momentum: SIGNAL = trailing return per symbol; CONSTRUCTION = top-k equal-wt."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

CONFIG = StrategyConfig(
    name="cross_sectional_momentum",
    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 60},
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Trailing `lookback`-bar return per symbol (the alpha score). Empty until enough history."""
    lookback = int(params["lookback"])
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")
    return (wide.iloc[-1] / wide.iloc[-1 - lookback] - 1.0).dropna()


def signal_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """OPTIONAL vectorized SCORES twin of `signal` (the canonical per-bar signal stays above). The
    full trailing-return matrix in one shot; rows without `lookback` history are all-NaN (the
    construction policy drops NaN, so those bars are flat — matching `signal` returning empty)."""
    lookback = int(params["lookback"])
    wide = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    return wide / wide.shift(lookback) - 1.0
