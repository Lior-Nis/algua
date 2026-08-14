"""Cadenced risk-adjusted relative strength with an incumbent rank buffer."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

GENERATED_BY = "agent"

CONFIG = StrategyConfig(
    name="cadenced_sharpe_rank_buffer",
    universe=["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "KO", "MSFT", "PG", "WMT", "XOM"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 126, "cadence": 5, "selection_k": 3, "buffer_rank": 5},
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=147,
)


def _risk_adjusted_scores(
    wide: pd.DataFrame, anchor_pos: int, lookback: int
) -> pd.Series:
    """Mean daily adjusted-price return divided by trailing sample volatility."""
    window = wide.iloc[anchor_pos - lookback : anchor_pos + 1]
    finite_prices = pd.Series(
        np.isfinite(window.to_numpy(dtype="float64")).sum(axis=0),
        index=window.columns,
    )
    complete = finite_prices == lookback + 1
    if not complete.any():
        return pd.Series(dtype="float64")

    daily = window.loc[:, complete].pct_change(fill_method=None).iloc[1:]
    finite_returns = pd.Series(
        np.isfinite(daily.to_numpy(dtype="float64")).sum(axis=0),
        index=daily.columns,
    )
    complete_returns = finite_returns == lookback
    daily = daily.loc[:, complete_returns]
    if daily.empty:
        return pd.Series(dtype="float64")

    mean_return = daily.mean()
    sample_vol = daily.std(ddof=1)
    valid = (
        np.isfinite(mean_return)
        & np.isfinite(sample_vol)
        & sample_vol.gt(0.0)
    )
    return (mean_return[valid] / sample_vol[valid]).sort_index()


def _ranked(scores: pd.Series) -> pd.Series:
    """Rank deterministically by score descending, then symbol ascending."""
    return scores.sort_index(kind="stable").sort_values(ascending=False, kind="stable")


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Return a buffered selection-priority score frozen between cadence anchors."""
    lookback = int(params["lookback"])
    cadence = int(params["cadence"])
    selection_k = int(params["selection_k"])
    buffer_rank = int(params["buffer_rank"])
    if lookback < 2 or cadence <= 0 or selection_k <= 0 or buffer_rank < selection_k:
        return pd.Series(dtype="float64")

    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    anchor_pos = ((len(wide) - 1) // cadence) * cadence
    prior_anchor_pos = anchor_pos - cadence
    if prior_anchor_pos < lookback:
        return pd.Series(dtype="float64")

    current = _ranked(_risk_adjusted_scores(wide, anchor_pos, lookback))
    prior = _ranked(_risk_adjusted_scores(wide, prior_anchor_pos, lookback))
    if current.empty or prior.empty:
        return pd.Series(dtype="float64")

    current_order = list(current.index)
    current_rank = {symbol: rank for rank, symbol in enumerate(current_order, start=1)}
    incumbents = {
        symbol
        for symbol in prior.index[:selection_k]
        if current_rank.get(symbol, buffer_rank + 1) <= buffer_rank
    }
    selected = [symbol for symbol in current_order if symbol in incumbents]
    selected.extend(
        symbol
        for symbol in current_order
        if symbol not in incumbents
    )
    selected = selected[:selection_k]

    # Encode the buffered winners ahead of the remaining raw rank while preserving deterministic
    # current-anchor order inside each group. The top-k policy therefore holds exactly this basket.
    selected_set = set(selected)
    buffered_order = selected + [symbol for symbol in current_order if symbol not in selected_set]
    return pd.Series(
        range(len(buffered_order), 0, -1),
        index=buffered_order,
        dtype="float64",
    )
