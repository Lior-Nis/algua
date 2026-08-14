"""Low realized volatility with stale momentum as a cross-sectional tie-break."""
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
    name="low_vol_skip_momentum_top3",
    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "UNH", "XOM"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={
        "vol_lookback": 120,
        "momentum_lookback": 252,
        "skip_recent": 21,
        "momentum_rank_weight": 0.01,
    },
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=252,
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Score lower realized-volatility names higher with skip-month momentum as tie-break."""
    vol_lookback = int(params["vol_lookback"])
    momentum_lookback = int(params["momentum_lookback"])
    skip_recent = int(params["skip_recent"])
    momentum_rank_weight = float(params["momentum_rank_weight"])

    if vol_lookback <= 0 or momentum_lookback <= skip_recent or skip_recent < 0:
        return pd.Series(dtype="float64")

    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) <= momentum_lookback or len(wide) <= vol_lookback:
        return pd.Series(dtype="float64")

    prices = wide.where(np.isfinite(wide) & (wide > 0.0))
    returns = prices.pct_change(fill_method=None)
    vol_window = returns.tail(vol_lookback).replace([np.inf, -np.inf], np.nan)
    complete_vol = vol_window.count() == vol_lookback
    realized_vol = vol_window.std(ddof=0) * np.sqrt(252.0)

    recent = prices.iloc[-1 - skip_recent]
    stale = prices.iloc[-1 - momentum_lookback]
    momentum = recent / stale - 1.0

    candidates = pd.DataFrame({
        "realized_vol": realized_vol,
        "momentum": momentum,
    })
    candidates = candidates[complete_vol]
    candidates = candidates[np.isfinite(candidates["realized_vol"])]
    candidates = candidates[np.isfinite(candidates["momentum"])]
    candidates = candidates[candidates["realized_vol"] >= 0.0]
    if candidates.empty:
        return pd.Series(dtype="float64")

    low_vol_rank = candidates["realized_vol"].rank(method="dense", ascending=False)
    momentum_rank = candidates["momentum"].rank(method="first", ascending=True, pct=True)
    score = low_vol_rank + momentum_rank_weight * momentum_rank
    return score.dropna()
