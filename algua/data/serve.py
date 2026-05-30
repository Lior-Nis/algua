from __future__ import annotations

from datetime import datetime

import pandas as pd

from algua.data.store import DataStore


class StoreBackedProvider:
    """Serves a single ingested bars snapshot through the DataProvider protocol.

    Point-in-time and reproducible: a backtest against this provider is pinned to exactly one
    snapshot, whose id is exposed for stamping into the result.
    """

    def __init__(self, store: DataStore, snapshot_id: str) -> None:
        self.store = store
        self.snapshot_id = snapshot_id

    def get_bars(
        self, symbols: list[str], start: datetime, end: datetime, timeframe: str
    ) -> pd.DataFrame:
        bars = self.store.read_bars(self.snapshot_id)  # bar-schema, validated
        bars = bars[bars["symbol"].isin(set(symbols))]
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        return bars[(bars.index >= start_ts) & (bars.index <= end_ts)]
