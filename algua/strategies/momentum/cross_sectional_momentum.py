"""Cross-sectional momentum: hold the top-k trailing-return names, equal weight."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

CONFIG = StrategyConfig(
    name="cross_sectional_momentum",
    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 60, "top_k": 3},
)


def _winners_to_weights(scores: pd.Series, top_k: int) -> pd.Series:
    """Top-k by score, equal weight. Shared by the per-bar and panel paths so both rank and weight
    identically (the engine's parity guard enforces this agreement)."""
    winners = scores.sort_values(ascending=False).head(top_k).index
    if len(winners) == 0:
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / len(winners), index=winners)


def compute_weights(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    lookback = int(params["lookback"])
    top_k = int(params["top_k"])
    # Wide adj_close (index=timestamp, columns=symbol) from the point-in-time view.
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")
    # Last-row momentum only — avoid materializing the full momentum matrix per bar.
    scores = (wide.iloc[-1] / wide.iloc[-1 - lookback] - 1.0).dropna()
    return _winners_to_weights(scores, top_k)


def compute_weights_panel(bars: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """OPTIONAL vectorized fast-path twin of `compute_weights` (the canonical signal stays above).

    Decision-time (PRE-lag) weights for the WHOLE period in one shot: pivot once, compute the full
    trailing-return matrix once (a single vectorized shift), then per row hold the top-k equal
    weight — IDENTICAL to running `compute_weights` on the expanding view at each bar. The engine's
    fail-closed parity guard verifies that identity on every run; this is purely an acceleration.

    Rows where there is not yet `lookback`-bars of history are flat (matching `compute_weights`,
    which returns an empty Series until `len(wide) > lookback`).
    """
    lookback = int(params["lookback"])
    top_k = int(params["top_k"])
    wide = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    # Trailing return over `lookback` bars for every (timestamp, symbol) at once.
    momentum = wide / wide.shift(lookback) - 1.0
    weights = pd.DataFrame(0.0, index=wide.index, columns=wide.columns)
    # The per-bar path forms a view only once len(wide) > lookback, i.e. from positional row
    # `lookback+1` onward (0-based index `lookback`). Earlier rows stay flat.
    for pos in range(lookback, len(wide.index)):
        scores = momentum.iloc[pos].dropna()
        w = _winners_to_weights(scores, top_k)
        if len(w):
            weights.iloc[pos, weights.columns.get_indexer(w.index)] = w.to_numpy()
    return weights
