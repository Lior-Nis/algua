"""Skip-month momentum tilted toward smooth, well-fitted adjusted-price trends."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

GENERATED_BY = "agent"

CONFIG = StrategyConfig(
    name="trend_fit_quality_momentum",
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
        "trend_lookback": 252,
        "momentum_lookback": 126,
        "skip_bars": 21,
        "quality_weight": 1.0,
    },
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=273,  # trend_lookback + skip_bars -> walk-forward embargo (#345)
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Rank momentum higher when its pre-skip log-price trend has a strong time-fit R-squared."""
    trend_lookback = int(params["trend_lookback"])
    momentum_lookback = int(params["momentum_lookback"])
    skip_bars = int(params["skip_bars"])
    quality_weight = float(params["quality_weight"])
    if (
        trend_lookback < 2
        or momentum_lookback <= 0
        or skip_bars < 0
        or not np.isfinite(quality_weight)
    ):
        return pd.Series(dtype="float64")

    required = max(trend_lookback - 1, momentum_lookback) + skip_bars
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

    trend_prices = wide.iloc[anchor_position - trend_lookback + 1 : anchor_position + 1]
    valid_trend_window = (
        (trend_prices.notna().sum() == trend_lookback)
        & np.isfinite(trend_prices).all()
        & (trend_prices > 0.0).all()
    )
    log_prices = np.log(trend_prices.where(trend_prices > 0.0))
    centered_prices = log_prices - log_prices.mean(axis=0)
    centered_time = pd.Series(
        np.arange(trend_lookback, dtype="float64") - (trend_lookback - 1) / 2.0,
        index=trend_prices.index,
    )
    time_sum_squares = float(centered_time.pow(2).sum())
    price_sum_squares = centered_prices.pow(2).sum(axis=0)
    time_price_cross_product = centered_prices.mul(centered_time, axis=0).sum(axis=0)
    trend_r_squared = (
        time_price_cross_product.pow(2) / (time_sum_squares * price_sum_squares)
    ).where(
        valid_trend_window
        & np.isfinite(price_sum_squares)
        & (price_sum_squares > 0.0)
    )
    trend_r_squared = trend_r_squared.clip(lower=0.0, upper=1.0)

    components = pd.concat(
        {"momentum": momentum, "trend_r_squared": trend_r_squared}, axis=1
    ).replace([np.inf, -np.inf], np.nan)
    components = components.dropna()
    if components.empty:
        return pd.Series(dtype="float64")

    momentum_rank = components["momentum"].rank(method="average", pct=True)
    quality_rank = components["trend_r_squared"].rank(method="average", pct=True)
    scores = momentum_rank + quality_weight * quality_rank
    return scores.astype("float64")
