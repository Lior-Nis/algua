"""Earnings-yield tilt: SIGNAL = latest KNOWN diluted EPS per symbol; CONSTRUCTION = equal-weight
the positive-score names. A minimal demonstration of the as-of fundamentals lane (issue #132)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

CONFIG = StrategyConfig(
    name="fundamentals_earnings_tilt",
    universe=["AAPL", "MSFT", "NVDA"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"metric": "eps_diluted"},
    construction="equal_weight_positive",
    needs_fundamentals=True,
)


def signal(view: pd.DataFrame, params: dict[str, Any], fundamentals: pd.DataFrame) -> pd.Series:
    """Most-recent-fiscal-period value of `metric` per symbol (the score). equal_weight_positive
    then holds the names with a positive score, equal weight.

    "Latest" means the latest fiscal period: the as-of engine already keeps each
    (symbol, fiscal_period_end) at its latest revision, so we sort EXPLICITLY by
    fiscal_period_end and take the last per symbol. Relying on `.last()` over the incoming frame
    order would silently depend on the engine's internal knowable_at ordering — an implementation
    detail, not the strategy-input contract — and a restatement of an old period landing with a
    later knowable_at could flip the selection (#274)."""
    metric = str(params["metric"])
    rows = fundamentals[fundamentals["metric"] == metric]
    if rows.empty:
        return pd.Series(dtype="float64")
    # Take the POSITIONAL last row per symbol after sorting — not groupby(...).last(), which
    # SKIPS nulls and would fall back to an older period when the newest period's value is NaN
    # ('reported-but-unavailable' is a valid contract value). The newest period's actual value
    # (incl. NaN -> a non-positive score, so the name isn't held) must win.
    ordered = rows.sort_values("fiscal_period_end", kind="stable")
    return ordered.drop_duplicates("symbol", keep="last").set_index("symbol")["value"]
