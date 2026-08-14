"""Rank peer-relative adjusted-price selloffs while vetoing extreme return shocks."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

GENERATED_BY = "agent"

CONFIG = StrategyConfig(
    name="shock_veto_peer_selloff_rebound",
    universe=["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "KO", "MSFT", "PG", "WMT", "XOM"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"selloff_lookback": 5, "shock_multiple": 3.5, "volatility_lookback": 63},
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=64,
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Score negative peer-relative selloffs that contain no extreme daily shock."""
    selloff_lookback = int(params["selloff_lookback"])
    shock_multiple = float(params["shock_multiple"])
    volatility_lookback = int(params["volatility_lookback"])

    if selloff_lookback < 1 or volatility_lookback < selloff_lookback:
        return pd.Series(dtype="float64")
    if not np.isfinite(shock_multiple) or shock_multiple <= 0.0:
        return pd.Series(dtype="float64")

    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) <= volatility_lookback:
        return pd.Series(dtype="float64")

    returns = wide.pct_change(fill_method=None)
    trailing_returns = returns.iloc[-volatility_lookback:]
    full_volatility_window = trailing_returns.notna().sum() == volatility_lookback

    robust_scale = 1.4826 * trailing_returns.abs().median()
    valid_scale = robust_scale.gt(0.0) & np.isfinite(robust_scale)

    selloff_returns = returns.iloc[-selloff_lookback:]
    full_selloff_window = selloff_returns.notna().sum() == selloff_lookback
    max_abs_return = selloff_returns.abs().max()
    shock = max_abs_return.gt(shock_multiple * robust_scale)

    selloff = wide.iloc[-1] / wide.iloc[-1 - selloff_lookback] - 1.0
    valid_selloff = np.isfinite(selloff)
    peer_selloff = selloff.where(full_selloff_window & valid_selloff).dropna()
    if peer_selloff.empty:
        return pd.Series(dtype="float64")

    peer_relative_selloff = peer_selloff - peer_selloff.median()
    eligible = full_volatility_window & full_selloff_window & valid_scale & valid_selloff & ~shock
    scores = (-peer_relative_selloff).where(eligible & peer_relative_selloff.lt(0.0)).dropna()
    return scores.astype("float64")
