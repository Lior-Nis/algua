"""Deterministic, human-readable price paths shared by acceptance scenarios."""
from __future__ import annotations

import pandas as pd

from algua.data.schema import BAR_COLUMNS, validate_bars


def decision_bars(*, periods: int = 61) -> pd.DataFrame:
    """Return obvious +60%, +30%, flat, -30%, and short-history paths.

    ``periods=61`` ends on the first 60-bar decision. ``periods=62`` adds the next bar where that
    decision may execute under the mandatory one-bar lag.
    """
    if periods not in (61, 62):
        raise ValueError("the approved reference market supports 61 or 62 periods")
    sessions = pd.bdate_range("2026-01-02", periods=periods, tz="UTC", name="timestamp")
    paths = {
        "AAPL": [100.0 + i for i in range(periods)],
        "NVDA": [100.0 + 0.5 * i for i in range(periods)],
        "MSFT": [100.0] * periods,
        "AMZN": [100.0 - 0.5 * i for i in range(periods)],
        "GOOGL": [None] * 30 + [80.0 + i for i in range(periods - 30)],
    }
    rows: list[dict[str, object]] = []
    for symbol, closes in paths.items():
        for timestamp, close in zip(sessions, closes, strict=True):
            if close is None:
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000.0,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["timestamp", "symbol"]).set_index("timestamp")
    return validate_bars(frame[BAR_COLUMNS], timeframe="1d")
