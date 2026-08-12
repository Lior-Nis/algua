"""Liquidity-10 4-day loser rebound, gated by calm pre-selloff volatility."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

GENERATED_BY = "agent"

CONFIG = StrategyConfig(
    name="calm_four_day_rebound",
    universe=["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "KO", "MSFT", "PG", "WMT", "XOM"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"selloff_lookback": 4, "pre_vol_lookback": 20},
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=25,
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Score sharp 4-day losers when the prior 20-day realized vol is below the peer median."""
    selloff_lookback = int(params["selloff_lookback"])
    pre_vol_lookback = int(params["pre_vol_lookback"])
    required_lookback = selloff_lookback + pre_vol_lookback + 1

    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) <= required_lookback:
        return pd.Series(dtype="float64")

    selloff_return = wide.iloc[-1] / wide.iloc[-1 - selloff_lookback] - 1.0
    pre_selloff_prices = wide.iloc[-1 - required_lookback : -selloff_lookback]
    pre_selloff_returns = pre_selloff_prices.pct_change().iloc[1:]
    if len(pre_selloff_returns) != pre_vol_lookback:
        return pd.Series(dtype="float64")

    complete = pre_selloff_returns.count() == pre_vol_lookback
    realized_vol = pre_selloff_returns.std().where(complete)
    finite_vol = realized_vol.where(np.isfinite(realized_vol))
    median_vol = finite_vol.dropna().median()
    if not np.isfinite(median_vol) or median_vol <= 0.0:
        return pd.Series(dtype="float64")

    selloff_depth = -selloff_return
    calm_discount = 1.0 - (finite_vol / median_vol)
    eligible = (selloff_return < 0.0) & (finite_vol < median_vol)
    scores = (selloff_depth * calm_discount).where(eligible)
    return scores.where(np.isfinite(scores)).dropna()
