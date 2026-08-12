"""Orderly six-day adjusted-price selloff rebound with volatility-spike penalty."""
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
    name="orderly_six_day_rebound",
    universe=["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "KO", "MSFT", "PG", "WMT", "XOM"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={
        "selloff_lookback": 6,
        "volatility_lookback": 20,
        "spike_window": 6,
        "min_down_days": 4,
    },
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=20,  # longest trailing return/volatility window -> walk-forward embargo (#345)
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Rank orderly short-horizon losers for rebound.

    Higher scores mean deeper, more persistent six-day selloffs, discounted when the most recent
    volatility is elevated versus the trailing baseline. Uses only point-in-time `adj_close` bars.
    """
    selloff_lookback = int(params["selloff_lookback"])
    volatility_lookback = int(params["volatility_lookback"])
    spike_window = int(params["spike_window"])
    min_down_days = int(params["min_down_days"])

    if selloff_lookback <= 0 or volatility_lookback <= 1 or spike_window <= 1:
        return pd.Series(dtype="float64")

    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    required_prices = max(selloff_lookback, volatility_lookback, spike_window) + 1
    if len(wide) < required_prices:
        return pd.Series(dtype="float64")

    returns = wide.pct_change()
    selloff_returns = returns.tail(selloff_lookback)
    baseline_returns = returns.tail(volatility_lookback)
    spike_returns = returns.tail(spike_window)

    enough_selloff = selloff_returns.notna().sum() == selloff_lookback
    enough_baseline = baseline_returns.notna().sum() == volatility_lookback
    enough_spike = spike_returns.notna().sum() == spike_window

    six_day_return = wide.iloc[-1] / wide.iloc[-1 - selloff_lookback] - 1.0
    down_days = (selloff_returns < 0.0).sum()
    down_consistency = down_days / float(selloff_lookback)

    baseline_vol = baseline_returns.std()
    recent_vol = spike_returns.std()
    valid_vol = (
        enough_baseline
        & enough_spike
        & baseline_vol.gt(0.0)
        & np.isfinite(baseline_vol)
        & np.isfinite(recent_vol)
    )
    spike_ratio = recent_vol / baseline_vol

    loss_depth = (-six_day_return).clip(lower=0.0)
    thesis_holds = (
        enough_selloff
        & six_day_return.lt(0.0)
        & down_days.ge(min_down_days)
        & valid_vol
        & spike_ratio.gt(0.0)
        & np.isfinite(spike_ratio)
    )
    scores = (loss_depth * down_consistency) / spike_ratio
    return scores.where(thesis_holds).dropna()
