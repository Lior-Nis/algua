"""Market-adjusted 3-day oversold rebound with volatility-spike filtering."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

# Provenance marker (additions-only discipline): every scaffolded module is agent-authored.
# Informational only — read by `algua doctor`'s advisory generated_provenance probe, NOT a trust
# or authorization control.
GENERATED_BY = "agent"

CONFIG = StrategyConfig(
    name="peer_median_oversold_rebound",
    universe=["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "KO", "MSFT", "PG", "WMT", "XOM"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"return_lookback": 3, "vol_lookback": 30, "spike_z": 2.0},
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=33,  # 30 prior absolute 3-day returns plus the 3-day return window.
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Score current 3-day underperformance versus the peer median, excluding spike outliers."""
    return_lookback = int(params["return_lookback"])
    vol_lookback = int(params["vol_lookback"])
    spike_z = float(params["spike_z"])

    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()

    required_rows = return_lookback + vol_lookback + 1
    if len(wide) < required_rows:
        return pd.Series(dtype="float64")

    returns = wide.pct_change(return_lookback)
    current_return = returns.iloc[-1]
    median_return = current_return.dropna().median()
    if pd.isna(median_return):
        return pd.Series(dtype="float64")

    prior_abs_returns = returns.abs().iloc[-vol_lookback - 1 : -1]
    if prior_abs_returns.count().lt(vol_lookback).all():
        return pd.Series(dtype="float64")

    enough_history = prior_abs_returns.count().eq(vol_lookback)
    abs_mean = prior_abs_returns.mean()
    abs_std = prior_abs_returns.std(ddof=0)
    spike_threshold = abs_mean + spike_z * abs_std
    current_abs_move = current_return.abs()

    scores = median_return - current_return
    selectable = (
        enough_history
        & current_return.notna()
        & scores.gt(0.0)
        & spike_threshold.notna()
        & current_abs_move.le(spike_threshold)
    )
    return scores.where(selectable).dropna()
