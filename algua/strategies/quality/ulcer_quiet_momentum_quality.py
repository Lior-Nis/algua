"""Quiet-underwater quality tilt among positive skip-month momentum names."""
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
    name="ulcer_quiet_momentum_quality",
    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={
        "lookback": 168,
        "momentum_lookback": 126,
        "skip": 21,
        "ulcer_weight": 1.0,
    },
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    # Bars of trailing history the signal reads -> walk-forward embargo (#345).
    feature_lookback=168,
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Return cross-sectional scores; higher favors momentum with shallower underwater paths."""
    lookback = int(params["lookback"])
    momentum_lookback = int(params["momentum_lookback"])
    skip = int(params["skip"])
    ulcer_weight = float(params["ulcer_weight"])

    min_history = max(lookback, momentum_lookback + skip + 1)
    if lookback <= 1 or momentum_lookback <= 0 or skip < 0 or not np.isfinite(ulcer_weight):
        return pd.Series(dtype="float64")

    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) < min_history:
        return pd.Series(dtype="float64")

    trailing = wide.iloc[-lookback:]
    complete = trailing.notna().sum() == lookback
    positive_prices = (trailing > 0.0).all()
    finite_prices = np.isfinite(trailing).all()
    valid_trailing = complete & positive_prices & finite_prices

    momentum_end = wide.iloc[-1 - skip]
    momentum_start = wide.iloc[-1 - skip - momentum_lookback]
    valid_momentum = (
        momentum_start.notna()
        & momentum_end.notna()
        & np.isfinite(momentum_start)
        & np.isfinite(momentum_end)
        & (momentum_start > 0.0)
    )
    momentum = (momentum_end / momentum_start) - 1.0

    running_peak = trailing.cummax()
    valid_peak = (running_peak > 0.0).all() & np.isfinite(running_peak).all()
    drawdown = (1.0 - trailing / running_peak).clip(lower=0.0)
    ulcer_index = np.sqrt((drawdown.pow(2)).mean())
    average_drawdown = drawdown.mean()

    valid = valid_trailing & valid_momentum & valid_peak & (momentum > 0.0)
    score = momentum - ulcer_weight * (ulcer_index + average_drawdown)
    score = score[valid]
    score = score[np.isfinite(score)]
    return score.astype("float64")
