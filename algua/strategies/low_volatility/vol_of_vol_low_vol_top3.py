"""Low realized volatility with stable short-window volatility."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

# Provenance marker (additions-only discipline): every scaffolded module is agent-authored.
# Informational only — read by `algua doctor`'s advisory generated_provenance probe, NOT a trust
# or authorization control.
GENERATED_BY = "agent"

CONFIG = StrategyConfig(
    name="vol_of_vol_low_vol_top3",
    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={
        "realized_vol_lookback": 60,
        "stability_lookback": 120,
        "short_vol_window": 20,
        "stability_penalty": 1.0,
    },
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    feature_lookback=140,  # 120 trailing 20-day realized-vol observations -> embargo (#345)
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Rank quiet names that have also stayed quiet; higher scores are more attractive."""
    realized_vol_lookback = int(params["realized_vol_lookback"])
    stability_lookback = int(params["stability_lookback"])
    short_vol_window = int(params["short_vol_window"])
    stability_penalty = float(params["stability_penalty"])

    required_bars = stability_lookback + short_vol_window
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) < required_bars:
        return pd.Series(dtype="float64")

    returns = wide.pct_change()

    realized_window = returns.tail(realized_vol_lookback)
    realized_counts = realized_window.count()
    realized_vol = realized_window.std()

    short_realized_vols = returns.rolling(
        window=short_vol_window,
        min_periods=short_vol_window,
    ).std()
    stability_window = short_realized_vols.tail(stability_lookback)
    stability_counts = stability_window.count()
    vol_of_vol = stability_window.std()

    has_full_history = (realized_counts == realized_vol_lookback) & (
        stability_counts == stability_lookback
    )
    score = -(realized_vol + stability_penalty * vol_of_vol)
    score = score.where(has_full_history)
    return score.replace([float("inf"), float("-inf")], pd.NA).dropna()
