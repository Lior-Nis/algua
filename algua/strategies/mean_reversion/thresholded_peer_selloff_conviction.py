"""Buy meaningful multi-session adjusted-price selloffs relative to liquid peers."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

GENERATED_BY = "agent"

CONFIG = StrategyConfig(
    name="thresholded_peer_selloff_conviction",
    universe=["AAPL", "AMZN", "GOOGL", "JNJ", "JPM", "KO", "MSFT", "PG", "WMT", "XOM"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"selloff_lookback": 5, "min_relative_selloff": 0.03},
    construction="score_proportional_long",
    feature_lookback=6,
)


def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Score complete-window selloffs by their excess magnitude versus other valid peers."""
    lookback = int(params["selloff_lookback"])
    threshold = float(params["min_relative_selloff"])
    if lookback < 1 or not float("-inf") < threshold < float("inf") or threshold < 0.0:
        return pd.Series(dtype="float64")

    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    wide = wide.sort_index()
    if len(wide) <= lookback:
        return pd.Series(dtype="float64")

    window = wide.iloc[-lookback - 1 :]
    complete = window.notna().sum().eq(lookback + 1)
    finite = window.gt(float("-inf")).all() & window.lt(float("inf")).all()
    positive = window.gt(0.0).all()
    valid = complete & finite & positive
    if valid.sum() < 2:
        return pd.Series(dtype="float64")

    valid_window = window.loc[:, valid]
    trailing_returns = valid_window.iloc[-1] / valid_window.iloc[0] - 1.0
    peer_returns = pd.Series(
        {
            symbol: trailing_returns.drop(index=symbol).median()
            for symbol in trailing_returns.index
        },
        dtype="float64",
    )
    excess_selloff = peer_returns - trailing_returns - threshold
    return excess_selloff[excess_selloff > 0.0].dropna().astype("float64")
