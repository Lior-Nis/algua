"""Cross-horizon low-volatility consensus using point-in-time adjusted prices."""
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
    name="cross_horizon_low_vol_consensus",
    universe=["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "KO", "MSFT", "PG", "WMT", "XOM"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"short_lookback": 90, "long_lookback": 120, "short_weight": 1.0},
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=121,
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Rank names by consensus between short- and long-horizon realized volatility."""
    short_lookback = int(params["short_lookback"])
    long_lookback = int(params["long_lookback"])
    short_weight = float(params["short_weight"])

    if (
        short_lookback < 2
        or long_lookback < 2
        or short_lookback > long_lookback
        or not np.isfinite(short_weight)
        or short_weight < 0.0
    ):
        return pd.Series(dtype="float64")

    required_prices = long_lookback + 1
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) < required_prices:
        return pd.Series(dtype="float64")

    prices = wide.iloc[-required_prices:]
    valid_prices = prices.notna() & np.isfinite(prices) & prices.gt(0.0)
    returns = prices.pct_change(fill_method=None).iloc[1:]

    long_returns = returns.iloc[-long_lookback:]
    short_returns = returns.iloc[-short_lookback:]
    full_long_window = (
        valid_prices.sum(axis=0).eq(required_prices)
        & long_returns.notna().sum(axis=0).eq(long_lookback)
        & np.isfinite(long_returns).sum(axis=0).eq(long_lookback)
    )
    full_short_window = (
        short_returns.notna().sum(axis=0).eq(short_lookback)
        & np.isfinite(short_returns).sum(axis=0).eq(short_lookback)
    )

    long_vol = long_returns.std(axis=0, ddof=1)
    short_vol = short_returns.std(axis=0, ddof=1)
    valid = (
        full_long_window
        & full_short_window
        & np.isfinite(long_vol)
        & np.isfinite(short_vol)
        & long_vol.gt(0.0)
        & short_vol.gt(0.0)
    )
    if not valid.any():
        return pd.Series(dtype="float64")

    long_calm_rank = long_vol[valid].rank(method="average", ascending=False, pct=True)
    short_calm_rank = short_vol[valid].rank(method="average", ascending=False, pct=True)
    score = (long_calm_rank + short_weight * short_calm_rank) / (1.0 + short_weight)
    return score.dropna().astype("float64")
