from __future__ import annotations

from datetime import datetime

import pandas as pd

from algua.data.store import DataStore


class StoreBackedProvider:
    """Serves a single ingested bars snapshot through the serving `DataProvider` protocol
    (`algua.contracts.types`) — distinct from the ingestion `BarProvider` seam.

    Point-in-time and reproducible: a backtest against this provider is pinned to exactly one
    snapshot, whose id is exposed for stamping into the result.

    The `[start, end)` window is half-open — bars timestamped exactly at `end` are excluded. This
    is the look-ahead-safe reading of "bars up to time T": asking for data as of T must not hand
    back the bar that prints at T. This is the canonical serving-seam convention (see
    `docs/contracts/bar-schema.md`).
    """

    def __init__(self, store: DataStore, snapshot_id: str) -> None:
        self.store = store
        self.snapshot_id = snapshot_id

    def get_bars(
        self, symbols: list[str], start: datetime, end: datetime, timeframe: str
    ) -> pd.DataFrame:
        rec = self.store.get_snapshot(self.snapshot_id)
        if timeframe != rec.metadata.timeframe:
            raise ValueError(
                f"snapshot {self.snapshot_id} is timeframe {rec.metadata.timeframe!r}, "
                f"not {timeframe!r}"
            )
        bars = self.store.read_bars(self.snapshot_id)
        bars = bars[bars["symbol"].isin(set(symbols))]
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        return bars[(bars.index >= start_ts) & (bars.index < end_ts)]
