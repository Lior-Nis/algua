"""Dual-horizon skip-month momentum that rewards persistent relative winners."""
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
    name="dual_horizon_skip_month_persistence",
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
        "short_lookback": 126,
        "long_lookback": 252,
        "skip_bars": 21,
        "disagreement_penalty": 0.5,
    },
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=273,  # long_lookback + skip_bars -> walk-forward embargo (#345)
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Score symbols by 6/12-month skip-month momentum, penalizing horizon disagreement."""
    short_lookback = int(params["short_lookback"])
    long_lookback = int(params["long_lookback"])
    skip_bars = int(params["skip_bars"])
    disagreement_penalty = float(params["disagreement_penalty"])
    if short_lookback <= 0 or long_lookback <= 0 or skip_bars < 0:
        return pd.Series(dtype="float64")

    required = max(short_lookback, long_lookback) + skip_bars
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) <= required:
        return pd.Series(dtype="float64")

    anchor = wide.iloc[-1 - skip_bars]
    short_base = wide.iloc[-1 - skip_bars - short_lookback]
    long_base = wide.iloc[-1 - skip_bars - long_lookback]

    valid = (
        anchor.notna()
        & short_base.notna()
        & long_base.notna()
        & np.isfinite(anchor)
        & np.isfinite(short_base)
        & np.isfinite(long_base)
        & (short_base > 0.0)
        & (long_base > 0.0)
    )
    if not valid.any():
        return pd.Series(dtype="float64")

    short_return = anchor[valid] / short_base[valid] - 1.0
    long_return = anchor[valid] / long_base[valid] - 1.0
    score = (short_return + long_return) / 2.0
    score -= disagreement_penalty * (short_return - long_return).abs()
    return score.replace([np.inf, -np.inf], np.nan).dropna()
