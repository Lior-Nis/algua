"""Rank low-volatility names when standard deviation and robust dispersion agree."""
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
    name="robust_dispersion_low_volatility",
    universe=["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "KO", "MSFT", "PG", "WMT", "XOM"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 60, "mad_weight": 0.5},
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=121,
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Score names by low standard volatility and low median absolute deviation."""
    lookback = int(params["lookback"])
    mad_weight = float(params["mad_weight"])
    if lookback < 2 or not 0.0 <= mad_weight < float("inf"):
        return pd.Series(dtype="float64")

    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")

    prices = wide.iloc[-(lookback + 1) :]
    prices = prices.where(prices.gt(0.0) & np.isfinite(prices))
    returns = prices.pct_change(fill_method=None).iloc[1:]
    finite_returns = returns.where(
        returns.gt(-float("inf")) & returns.lt(float("inf"))
    )
    valid_symbols = finite_returns.count().eq(lookback)
    if not valid_symbols.any():
        return pd.Series(dtype="float64")

    window = finite_returns.loc[:, valid_symbols]
    standard_vol = window.std(ddof=1)
    median_return = window.median()
    median_absolute_deviation = window.sub(median_return).abs().median()
    usable = (
        standard_vol.gt(0.0)
        & standard_vol.lt(float("inf"))
        & median_absolute_deviation.gt(0.0)
        & median_absolute_deviation.lt(float("inf"))
    )
    standard_vol = standard_vol[usable]
    median_absolute_deviation = median_absolute_deviation[usable]
    if standard_vol.empty:
        return pd.Series(dtype="float64")

    standard_score = standard_vol.rank(method="average", ascending=False, pct=True)
    mad_score = median_absolute_deviation.rank(method="average", ascending=False, pct=True)
    return (standard_score + mad_weight * mad_score) / (1.0 + mad_weight)
