"""Skip-month momentum that rewards persistent cross-sectional leadership."""
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
    name="lagged_rank_persistence_momentum",
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
        "formation_lookback": 252,
        "skip_bars": 21,
        "anchor_spacing": 21,
        "anchor_count": 4,
        "persistence_weight": 0.75,
    },
    construction="top_k_equal_weight",
    construction_params={"top_k": 3},
    # formation_lookback + skip_bars + (anchor_count - 1) * anchor_spacing
    feature_lookback=336,
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Rank skip-month returns and reward stable leadership across monthly anchors."""
    formation_lookback = int(params["formation_lookback"])
    skip_bars = int(params["skip_bars"])
    anchor_spacing = int(params["anchor_spacing"])
    anchor_count = int(params["anchor_count"])
    persistence_weight = float(params["persistence_weight"])
    if (
        formation_lookback <= 0
        or skip_bars < 0
        or anchor_spacing <= 0
        or anchor_count <= 0
        or not np.isfinite(persistence_weight)
        or persistence_weight < 0.0
    ):
        return pd.Series(dtype="float64")

    required = formation_lookback + skip_bars + (anchor_count - 1) * anchor_spacing
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) <= required:
        return pd.Series(dtype="float64")

    endpoints: list[tuple[pd.Series, pd.Series]] = []
    valid = pd.Series(True, index=wide.columns, dtype="bool")
    for anchor_number in range(anchor_count):
        end_offset = skip_bars + anchor_number * anchor_spacing
        end = wide.iloc[-1 - end_offset]
        start = wide.iloc[-1 - end_offset - formation_lookback]
        endpoints.append((start, end))
        valid &= (
            start.notna()
            & end.notna()
            & np.isfinite(start)
            & np.isfinite(end)
            & (start > 0.0)
            & (end > 0.0)
        )

    if not valid.any():
        return pd.Series(dtype="float64")

    anchor_ranks: list[pd.Series] = []
    for start, end in endpoints:
        returns = end[valid] / start[valid] - 1.0
        anchor_ranks.append(returns.rank(pct=True))

    ranks = pd.concat(anchor_ranks, axis="columns")
    current_rank = ranks.iloc[:, 0]
    persistence = ranks.mean(axis="columns") - ranks.std(axis="columns", ddof=0)
    score = current_rank + persistence_weight * persistence
    return score.replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
