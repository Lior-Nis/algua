"""Headline-coverage tilt: SIGNAL = count of an article-mention per symbol within a trailing
window of the as-of news; CONSTRUCTION = equal-weight the positively-covered names. A minimal
demonstration of the as-of news lane (issue #132) — no sentiment/NLP, just coverage counts."""
from __future__ import annotations

from typing import Any

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.strategies.base import StrategyConfig

# Provenance marker (additions-only discipline): these bundled seed examples are hand-authored.
# Informational only — read by `algua doctor`'s advisory generated_provenance probe, NOT a trust
# or authorization control.
GENERATED_BY = "human"

CONFIG = StrategyConfig(
    name="news_coverage_tilt",
    universe=["AAPL", "MSFT", "NVDA"],
    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
    params={"window_days": 5},
    construction="equal_weight_positive",
    needs_news=True,
    feature_lookback=5,  # 5-day news coverage window -> walk-forward embargo (#345)
)


def signal(view: pd.DataFrame, params: dict[str, Any], news: pd.DataFrame) -> pd.Series:
    """Score = number of distinct articles mentioning each symbol whose published_at is within the
    last `window_days` of the latest bar. The engine's as-of mask already removed retracted
    mentions and any news not knowable by the decision bar; we window on published_at here."""
    if news.empty or view.empty:
        return pd.Series(dtype="float64")
    window = int(params["window_days"])
    cutoff = view.index.max() - pd.Timedelta(days=window)
    recent = news[news["published_at"] >= cutoff]
    if recent.empty:
        return pd.Series(dtype="float64")
    return recent.groupby("symbol")["article_id"].nunique().astype("float64")
