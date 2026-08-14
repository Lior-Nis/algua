"""Skip-month momentum tilted toward gains distributed broadly across the price path."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

GENERATED_BY = "agent"

CONFIG = StrategyConfig(
    name="distributed_gains_quality_momentum",
    universe=[
        "AAPL",
        "AMZN",
        "GOOGL",
        "JNJ",
        "JPM",
        "KO",
        "MSFT",
        "PG",
        "WMT",
        "XOM",
    ],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={
        "distribution_lookback": 252,
        "momentum_lookback": 126,
        "skip_bars": 21,
        "quality_weight": 1.0,
    },
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=273,  # distribution_lookback + skip_bars -> walk-forward embargo (#345)
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Rank momentum higher when positive log-return contributions span more pre-skip days."""
    distribution_lookback = int(params["distribution_lookback"])
    momentum_lookback = int(params["momentum_lookback"])
    skip_bars = int(params["skip_bars"])
    quality_weight = float(params["quality_weight"])
    if (
        distribution_lookback < 3
        or momentum_lookback <= 0
        or skip_bars < 0
        or not np.isfinite(quality_weight)
    ):
        return pd.Series(dtype="float64")

    required = max(distribution_lookback - 1, momentum_lookback) + skip_bars
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) <= required:
        return pd.Series(dtype="float64")

    anchor_position = len(wide) - 1 - skip_bars
    momentum_prices = wide.iloc[
        anchor_position - momentum_lookback : anchor_position + 1
    ]
    valid_momentum_window = (
        (momentum_prices.notna().sum() == momentum_lookback + 1)
        & np.isfinite(momentum_prices).all()
        & (momentum_prices > 0.0).all()
    )
    momentum = (
        momentum_prices.iloc[-1] / momentum_prices.iloc[0] - 1.0
    ).where(valid_momentum_window)

    distribution_prices = wide.iloc[
        anchor_position - distribution_lookback + 1 : anchor_position + 1
    ]
    valid_distribution_window = (
        (distribution_prices.notna().sum() == distribution_lookback)
        & np.isfinite(distribution_prices).all()
        & (distribution_prices > 0.0).all()
    )
    log_returns = np.log(distribution_prices.where(distribution_prices > 0.0)).diff().iloc[1:]
    return_count = distribution_lookback - 1
    valid_returns = (
        (log_returns.notna().sum() == return_count)
        & np.isfinite(log_returns).all()
        & valid_distribution_window
    )
    positive_contributions = log_returns.clip(lower=0.0)
    positive_total = positive_contributions.sum(axis=0)
    contribution_hhi = positive_contributions.pow(2).sum(axis=0) / positive_total.pow(2)
    effective_positive_days = 1.0 / contribution_hhi
    contribution_breadth = (effective_positive_days - 1.0) / (return_count - 1.0)
    contribution_breadth = contribution_breadth.where(
        valid_returns
        & np.isfinite(positive_total)
        & (positive_total > 0.0)
        & np.isfinite(contribution_hhi)
        & (contribution_hhi > 0.0)
    ).clip(lower=0.0, upper=1.0)

    components = pd.concat(
        {"momentum": momentum, "contribution_breadth": contribution_breadth}, axis=1
    ).replace([np.inf, -np.inf], np.nan)
    components = components.dropna()
    if components.empty:
        return pd.Series(dtype="float64")

    momentum_rank = components["momentum"].rank(method="average", pct=True)
    quality_rank = components["contribution_breadth"].rank(method="average", pct=True)
    scores = momentum_rank + quality_weight * quality_rank
    return scores.astype("float64")
