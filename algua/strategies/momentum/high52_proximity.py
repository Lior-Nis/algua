"""52-week-high proximity: hold the names trading closest to their trailing 252-bar high.

Anchoring effect (George & Hwang 2004): investors under-react to good news for names near their
52-week high, so proximity to the high predicts continuation. The score is the ratio of the latest
adjusted close to its trailing `lookback`-bar maximum (1.0 = at the high); a ranking policy keeps
the top names.
"""
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
    name="high52_proximity",
    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"lookback": 252},
    construction="top_k_equal_weight",
    construction_params={"top_k": 5},
    feature_lookback=252,  # the trailing high spans `lookback` bars -> walk-forward embargo (#345)
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Latest adj_close / trailing `lookback`-bar max adj_close per symbol (higher = nearer the
    high). Scores are in (0, 1]; a symbol without a complete window is not scored."""
    lookback = int(params["lookback"])
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) < lookback:
        return pd.Series(dtype="float64")
    window = wide.iloc[-lookback:]
    complete = window.notna().all(axis=0)  # a sparse window would shrink the max silently
    high = window.max(axis=0)
    score = window.iloc[-1] / high.where(high > 0)
    return score[complete].dropna()
