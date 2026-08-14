"""Rank liquid names by low volatility residual to an equal-weight market proxy."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

GENERATED_BY = "agent"

CONFIG = StrategyConfig(
    name="market_residual_low_volatility",
    universe=["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "KO", "MSFT", "PG", "WMT", "XOM"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 60},
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=121,
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Score lower trailing stock-specific volatility more highly."""
    lookback = int(params["lookback"])
    if lookback < 2:
        return pd.Series(dtype="float64")

    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")

    prices = wide.iloc[-(lookback + 1) :]
    valid_prices = (prices > 0.0) & np.isfinite(prices)
    clean_prices = prices.where(valid_prices)
    returns = clean_prices.pct_change(fill_method=None).iloc[1:]
    returns = returns.where(np.isfinite(returns))

    complete = returns.notna().sum(axis=0).eq(lookback)
    complete_returns = returns.loc[:, complete]
    if complete_returns.empty:
        return pd.Series(dtype="float64")

    market_return = complete_returns.mean(axis=1)
    residual_returns = complete_returns.sub(market_return, axis=0)
    residual_volatility = residual_returns.std(axis=0, ddof=1)
    valid_volatility = (residual_volatility > 0.0) & np.isfinite(residual_volatility)
    return (-residual_volatility.loc[valid_volatility]).astype("float64")
