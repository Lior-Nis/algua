"""Peer-relative selloff rebound favoring losses distributed across sessions."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

# Provenance marker (additions-only discipline): every scaffolded module is agent-authored.
# Informational only — read by `algua doctor`'s advisory generated_provenance probe, NOT a trust
# or authorization control.
GENERATED_BY = "agent"

CONFIG = StrategyConfig(
    name="distributed_loss_peer_selloff_rebound",
    universe=["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "KO", "MSFT", "PG", "WMT", "XOM"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"selloff_lookback": 5, "concentration_weight": 1.0},
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=7,
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Score negative peer-relative selloffs, penalizing one-session loss concentration."""
    lookback = int(params["selloff_lookback"])
    concentration_weight = float(params["concentration_weight"])
    if lookback <= 1 or not np.isfinite(concentration_weight) or concentration_weight < 0.0:
        return pd.Series(dtype="float64")

    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) < lookback + 1:
        return pd.Series(dtype="float64")

    prices = wide.iloc[-lookback - 1 :]
    full_price_window = prices.notna().sum(axis=0) == lookback + 1
    finite_prices = pd.Series(np.isfinite(prices).all(axis=0), index=prices.columns)
    nonzero_return_denominators = (prices.iloc[:-1] != 0.0).all(axis=0)

    daily_returns = prices.pct_change(fill_method=None).iloc[1:]
    full_return_window = daily_returns.notna().sum(axis=0) == lookback
    finite_returns = pd.Series(np.isfinite(daily_returns).all(axis=0), index=prices.columns)

    start = prices.iloc[0]
    end = prices.iloc[-1]
    valid_window = (
        full_price_window
        & finite_prices
        & nonzero_return_denominators
        & full_return_window
        & finite_returns
        & (start != 0.0)
    )
    cumulative_return = (end / start - 1.0).where(valid_window)
    valid_cumulative = cumulative_return.dropna()
    if len(valid_cumulative) < 2:
        return pd.Series(dtype="float64")

    peer_median = float(valid_cumulative.median())
    if not np.isfinite(peer_median):
        return pd.Series(dtype="float64")
    relative_selloff = peer_median - cumulative_return

    downside = (-daily_returns).clip(lower=0.0)
    downside_sum = downside.sum(axis=0)
    downside_hhi = downside.pow(2).sum(axis=0) / downside_sum.pow(2)
    minimum_hhi = 1.0 / lookback
    normalized_concentration = (downside_hhi - minimum_hhi) / (1.0 - minimum_hhi)
    normalized_concentration = normalized_concentration.clip(lower=0.0, upper=1.0)

    penalty_denominator = 1.0 + concentration_weight * normalized_concentration
    eligible = (
        valid_window
        & (relative_selloff > 0.0)
        & (downside_sum > 0.0)
        & np.isfinite(relative_selloff)
        & np.isfinite(downside_hhi)
        & np.isfinite(penalty_denominator)
        & (penalty_denominator > 0.0)
    )
    scores = (relative_selloff / penalty_denominator).where(eligible).dropna()
    return scores.astype("float64")
