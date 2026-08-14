"""Cadenced relative strength scaled by trailing expected-shortfall magnitude."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

# Provenance marker (additions-only discipline): every scaffolded module is agent-authored.
# Informational only — read by `algua doctor`'s advisory generated_provenance probe, NOT a trust
# or authorization control.
GENERATED_BY = "agent"

CONFIG = StrategyConfig(
    name="cadenced_tail_risk_relative_strength",
    universe=["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "KO", "MSFT", "PG", "WMT", "XOM"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 126, "cadence": 10, "tail_frac": 0.2},
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=136,
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Rank trailing return per unit of expected-shortfall magnitude at cadence anchors."""
    lookback = int(params["lookback"])
    cadence = int(params["cadence"])
    tail_frac = float(params["tail_frac"])
    if lookback < 1 or cadence < 1 or not math.isfinite(tail_frac) or not 0.0 < tail_frac <= 1.0:
        return pd.Series(dtype="float64")

    tail_count = math.ceil(lookback * tail_frac)
    if tail_count < 1:
        return pd.Series(dtype="float64")

    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()

    current_pos = len(wide) - 1
    anchor_pos = current_pos - current_pos % cadence
    if anchor_pos < lookback:
        return pd.Series(dtype="float64")

    prices = wide.iloc[anchor_pos - lookback : anchor_pos + 1]
    trailing_returns = prices.pct_change(fill_method=None).iloc[1:]
    complete = trailing_returns.notna().sum().eq(lookback)
    finite = trailing_returns.apply(lambda column: column.map(math.isfinite)).all()
    trailing_returns = trailing_returns.loc[:, complete & finite]
    if trailing_returns.empty:
        return pd.Series(dtype="float64")

    expected_shortfall = trailing_returns.apply(
        lambda column: abs(column.nsmallest(tail_count).mean())
    )
    valid_risk = expected_shortfall.gt(0.0) & expected_shortfall.map(math.isfinite)
    expected_shortfall = expected_shortfall.loc[valid_risk]
    if expected_shortfall.empty:
        return pd.Series(dtype="float64")

    trailing_return = prices.iloc[-1] / prices.iloc[0] - 1.0
    scores = trailing_return.loc[expected_shortfall.index] / expected_shortfall
    return scores.loc[scores.map(math.isfinite)].dropna().astype("float64")
