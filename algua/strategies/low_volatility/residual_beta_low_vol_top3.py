"""Residual beta-adjusted defensive low-volatility scores."""
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
    name="residual_beta_low_vol_top3",
    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "UNH", "XOM"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 120, "beta_penalty": 0.01},
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=120,  # trailing returns read by the signal -> walk-forward embargo (#345)
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Score low idiosyncratic volatility after equal-weight market beta adjustment."""
    lookback = int(params["lookback"])
    beta_penalty = float(params["beta_penalty"])
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")

    returns = wide.pct_change(fill_method=None).iloc[-lookback:]
    market = returns.mean(axis=1, skipna=True)
    market_var = float(market.var(ddof=1))
    if not np.isfinite(market_var) or market_var <= 0.0:
        return pd.Series(dtype="float64")

    scores: dict[str, float] = {}
    for symbol in returns.columns:
        asset = returns[symbol]
        sample = pd.concat([asset, market], axis=1, keys=["asset", "market"]).dropna()
        if len(sample) != lookback:
            continue
        beta = float(sample["asset"].cov(sample["market"]) / market_var)
        if not np.isfinite(beta):
            continue
        intercept = float(sample["asset"].mean() - beta * sample["market"].mean())
        residual = sample["asset"] - intercept - beta * sample["market"]
        residual_vol = float(residual.std(ddof=1))
        if not np.isfinite(residual_vol):
            continue
        scores[str(symbol)] = -(residual_vol + beta_penalty * max(beta, 0.0))

    return pd.Series(scores, dtype="float64").dropna()
