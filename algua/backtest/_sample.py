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
    # Deterministic (fixed seed) -> a reproducible data source for the agent promote guard (#205):
    # the OOS bars are identical on a re-run, so a burned holdout is reproducible w/o a snapshot.
    reproducible = True

    def __init__(self, seed: int = 0, *, drift: float = 0.0005, vol: float = 0.02) -> None:
        # drift/vol are the per-bar GBM mean/stddev of log-returns. They default to the original
        # constants so SyntheticProvider(seed=k) is byte-identical to before this param was added
        # (a regression test pins this). They exist so the gate-eval harness (#347) can realize
        # scenarios of known true edge (strong / zero / negative drift) from one generator.
        self.seed = seed
        self.drift = drift
        self.vol = vol

    def get_bars(
        self, symbols: list[str], start: datetime, end: datetime, timeframe: str
    ) -> pd.DataFrame:
        if timeframe not in _SUPPORTED:
            raise ValueError(f"unsupported timeframe: {timeframe!r}")
        if not symbols:
            empty = pd.DataFrame(columns=_BAR_COLUMNS)
            empty.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
            return empty
        # Daily bars are timestamped at the session date (tz-aware UTC midnight), per the bar
        # schema and what real daily sources provide. Calendar-based so holidays are skipped.
        session_dates = MarketCalendar("XNYS").sessions_in_range(start.date(), end.date())
        sessions = pd.DatetimeIndex(
            [pd.Timestamp(d, tz="UTC") for d in session_dates], name="timestamp"
        )
        frames = []
        for i, sym in enumerate(sorted(symbols)):
            # Deterministic per-symbol drift/vol from a child RNG.
            sub = np.random.default_rng(self.seed + i + 1)
            rets = sub.normal(loc=self.drift, scale=self.vol, size=len(sessions))
            close = 100.0 * np.exp(np.cumsum(rets))
            # Open differs from close so high/low-dependent logic can be exercised.
            open_ = close * np.exp(sub.normal(loc=0.0, scale=0.005, size=len(sessions)))
            # Jitter widens the bar so high > max(open,close) and low < min(open,close);
            # the OHLC invariant low <= open,close <= high always holds by construction.
            jitter = sub.uniform(0.001, 0.008, size=len(sessions))
            high = np.maximum(open_, close) * (1.0 + jitter)
            low = np.minimum(open_, close) * (1.0 - jitter)
            frames.append(pd.DataFrame({
                "timestamp": sessions, "symbol": sym,
                "open": open_, "high": high, "low": low,
                "close": close, "adj_close": close, "volume": 1_000_000.0,
            }))
        out = pd.concat(frames).set_index("timestamp").sort_values(["timestamp", "symbol"])
        return out[_BAR_COLUMNS]
