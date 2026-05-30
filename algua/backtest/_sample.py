"""Dev-only synthetic DataProvider producing bar-schema-conformant data.

NOT for production use. Exists so the research lane can build and test the backtest
engine end-to-end without the real data layer. Geometric-brownian-ish price paths.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from algua.calendar.market_calendar import MarketCalendar

_BAR_COLUMNS = ["symbol", "open", "high", "low", "close", "adj_close", "volume"]
_SUPPORTED = {"1d"}


class SyntheticProvider:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def get_bars(
        self, symbols: list[str], start: datetime, end: datetime, timeframe: str
    ) -> pd.DataFrame:
        if timeframe not in _SUPPORTED:
            raise ValueError(f"unsupported timeframe: {timeframe!r}")
        if not symbols:
            empty = pd.DataFrame(columns=_BAR_COLUMNS)
            empty.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
            return empty
        # Timestamp daily bars at the XNYS session close (tz-aware UTC), matching the frozen
        # bar schema and what the real provider emits — not naive midnight business days.
        sessions = pd.DatetimeIndex(
            MarketCalendar("XNYS").session_closes(start.date(), end.date()), name="timestamp"
        )
        frames = []
        for i, sym in enumerate(sorted(symbols)):
            # Deterministic per-symbol drift/vol from a child RNG.
            sub = np.random.default_rng(self.seed + i + 1)
            rets = sub.normal(loc=0.0005, scale=0.02, size=len(sessions))
            price = 100.0 * np.exp(np.cumsum(rets))
            frames.append(pd.DataFrame({
                "timestamp": sessions, "symbol": sym,
                "open": price, "high": price * 1.01, "low": price * 0.99,
                "close": price, "adj_close": price, "volume": 1_000_000.0,
            }))
        out = pd.concat(frames).set_index("timestamp").sort_values(["timestamp", "symbol"])
        return out[_BAR_COLUMNS]
