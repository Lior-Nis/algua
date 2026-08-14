"""Cadenced relative strength scaled by robust trailing-return dispersion."""
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
    name="cadenced_mad_relative_strength",
    universe=["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "KO", "MSFT", "PG", "WMT", "XOM"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 126, "cadence": 5},
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=136,
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Score trailing return per unit of MAD, frozen between cadence anchors."""
    lookback = int(params["lookback"])
    cadence = int(params["cadence"])
    if lookback <= 0 or cadence <= 0:
        return pd.Series(dtype="float64")

    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()

    anchor_offset = (len(wide) - 1) % cadence
    anchor_position = len(wide) - 1 - anchor_offset
    window_start = anchor_position - lookback
    if window_start < 0:
        return pd.Series(dtype="float64")

    prices = wide.iloc[window_start : anchor_position + 1]
    daily_returns = prices.pct_change(fill_method=None).iloc[1:]
    complete = (
        prices.notna().sum().eq(lookback + 1)
        & daily_returns.notna().sum().eq(lookback)
        & np.isfinite(daily_returns).all(axis=0)
    )
    if not complete.any():
        return pd.Series(dtype="float64")

    prices = prices.loc[:, complete]
    daily_returns = daily_returns.loc[:, complete]
    return_median = daily_returns.median(axis=0)
    mad = daily_returns.sub(return_median, axis="columns").abs().median(axis=0)
    trailing_return = prices.iloc[-1] / prices.iloc[0] - 1.0

    valid = (mad > 0.0) & np.isfinite(mad) & np.isfinite(trailing_return)
    return (trailing_return.loc[valid] / mad.loc[valid]).astype("float64")
