"""Liquidity-stable quality momentum with a penalty for jumpy flow."""
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
    name="liquidity_stable_quality_momentum",
    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={
        "price_lookback": 168,
        "momentum_lookback": 126,
        "volume_lookback": 63,
        "skip": 21,
        "volume_weight": 0.5,
        "quality_weight": 0.5,
    },
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=168,
)


def _rank_score(values: pd.Series) -> pd.Series:
    return values.dropna().rank(method="average", pct=True)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Return cross-sectional scores; construction maps the top names to equal weights."""
    price_lookback = int(params["price_lookback"])
    momentum_lookback = int(params["momentum_lookback"])
    volume_lookback = int(params["volume_lookback"])
    skip = int(params["skip"])
    volume_weight = float(params["volume_weight"])
    quality_weight = float(params["quality_weight"])
    required = max(price_lookback, momentum_lookback + skip, volume_lookback)

    bars = view.reset_index()[["timestamp", "symbol", "adj_close", "volume"]].copy()
    bars["dollar_volume"] = bars["adj_close"] * bars["volume"]
    prices = bars.pivot(index="timestamp", columns="symbol", values="adj_close").sort_index()
    dollar_volume = (
        bars.pivot(index="timestamp", columns="symbol", values="dollar_volume").sort_index()
    )
    if len(prices) <= required or len(dollar_volume) <= volume_lookback:
        return pd.Series(dtype="float64")

    momentum_now = prices.iloc[-1 - skip]
    momentum_then = prices.iloc[-1 - skip - momentum_lookback]
    momentum_window = prices.iloc[-1 - skip - momentum_lookback : -skip]
    valid_momentum = (
        (momentum_window.notna().sum() == momentum_lookback + 1)
        & np.isfinite(momentum_now)
        & np.isfinite(momentum_then)
        & (momentum_then > 0.0)
    )
    momentum = (momentum_now / momentum_then - 1.0).where(valid_momentum)

    price_window = prices.iloc[-1 - price_lookback :]
    log_prices = np.log(price_window.where(price_window > 0.0))
    valid_prices = log_prices.notna().sum() == price_lookback + 1
    path_distance = log_prices.diff().abs().sum()
    net_move = (log_prices.iloc[-1] - log_prices.iloc[0]).abs()
    efficiency = (net_move / path_distance).where(
        valid_prices & np.isfinite(path_distance) & (path_distance > 0.0)
    )

    volume_window = dollar_volume.iloc[-1 - volume_lookback :]
    log_dollar_volume = np.log(volume_window.where(volume_window > 0.0))
    valid_volume = log_dollar_volume.notna().sum() == volume_lookback + 1
    volume_jitter = log_dollar_volume.diff().std().where(valid_volume)

    components = pd.concat(
        {
            "momentum": _rank_score(momentum),
            "efficiency": _rank_score(efficiency),
            "volume_jitter": _rank_score(volume_jitter),
        },
        axis=1,
    ).dropna()
    if components.empty:
        return pd.Series(dtype="float64")

    scores = (
        components["momentum"]
        + quality_weight * components["efficiency"]
        - volume_weight * components["volume_jitter"]
    )
    return scores.replace([np.inf, -np.inf], np.nan).dropna()
