"""Path-efficiency quality momentum: smooth adjusted-close momentum beats choppy momentum."""
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
    name="path_efficiency_quality_momentum",
    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={
        "lookback": 252,
        "momentum_lookback": 126,
        "skip": 21,
        "quality_weight": 0.75,
    },
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=252,  # largest trailing window the signal reads -> walk-forward embargo (#345)
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Return cross-sectional scores; construction maps the top names to equal weights."""
    lookback = int(params["lookback"])
    momentum_lookback = int(params["momentum_lookback"])
    skip = int(params["skip"])
    quality_weight = float(params["quality_weight"])

    if lookback <= 0 or momentum_lookback <= 0 or skip < 0:
        return pd.Series(dtype="float64")

    required_prices = max(lookback, momentum_lookback + skip) + 1
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) < required_prices:
        return pd.Series(dtype="float64")

    momentum_end = wide.iloc[-1 - skip]
    momentum_start = wide.iloc[-1 - skip - momentum_lookback]
    valid_momentum = (
        momentum_end.notna()
        & momentum_start.notna()
        & np.isfinite(momentum_end)
        & np.isfinite(momentum_start)
        & (momentum_start != 0.0)
    )
    momentum = (momentum_end / momentum_start - 1.0).where(valid_momentum)

    prices = wide.iloc[-lookback - 1 :]
    full_price_window = prices.notna().sum(axis=0) == lookback + 1
    daily_returns = prices.pct_change(fill_method=None).iloc[1:]
    full_return_window = daily_returns.notna().sum(axis=0) == lookback

    start = prices.iloc[0]
    end = prices.iloc[-1]
    total_abs_return = (end / start - 1.0).abs()
    path_length = daily_returns.abs().sum(axis=0)
    valid_path = (
        full_price_window
        & full_return_window
        & start.notna()
        & end.notna()
        & np.isfinite(start)
        & np.isfinite(end)
        & np.isfinite(path_length)
        & (start != 0.0)
        & (path_length > 0.0)
    )
    path_efficiency = (total_abs_return / path_length).where(valid_path)

    running_peak = prices.cummax()
    drawdown = prices / running_peak - 1.0
    drawdown_depth = (-drawdown.min(axis=0)).where(full_price_window)
    underwater_share = (drawdown < 0.0).sum(axis=0) / lookback
    underwater_penalty = (0.5 * drawdown_depth + 0.5 * underwater_share).where(valid_path)
    quality = path_efficiency - underwater_penalty

    valid = momentum.notna() & quality.notna() & np.isfinite(momentum) & np.isfinite(quality)
    if not valid.any():
        return pd.Series(dtype="float64")

    momentum_rank = momentum[valid].rank(pct=True)
    quality_rank = quality[valid].rank(pct=True)
    score = (1.0 - quality_weight) * momentum_rank + quality_weight * quality_rank
    return score.dropna().astype("float64")
