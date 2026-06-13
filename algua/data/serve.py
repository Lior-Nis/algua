from __future__ import annotations

from datetime import datetime

import pandas as pd

from algua.data.store import DataStore


class StoreBackedFundamentalsProvider:
    """Serves one fundamentals snapshot through the as-of `FundamentalsProvider` seam. Returns the
    FULL bitemporal history with knowable_at < end (no lower bound — the first decision bar needs
    the latest prior report). The engine applies the per-bar knowable_at <= t mask; this provider
    never sees `t`."""

    def __init__(self, store: DataStore, snapshot_id: str) -> None:
        self.store = store
        self.snapshot_id = snapshot_id

    def get_fundamentals(self, symbols: list[str], end: datetime) -> pd.DataFrame:
        frame = self.store.read_fundamentals(self.snapshot_id, symbols=symbols)
        end_ts = pd.Timestamp(end)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
        return frame[frame["knowable_at"] < end_ts].reset_index(drop=True)


class StoreBackedNewsProvider:
    """Serves one news snapshot through the as-of `NewsProvider` seam. Returns the FULL bitemporal
    history (including tombstones) with knowable_at < end (no lower bound — the first decision bar
    needs prior news). The engine applies the per-bar knowable_at <= t mask; this provider never
    sees `t`."""

    def __init__(self, store: DataStore, snapshot_id: str) -> None:
        self.store = store
        self.snapshot_id = snapshot_id

    def get_news(self, symbols: list[str], end: datetime) -> pd.DataFrame:
        frame = self.store.read_news(self.snapshot_id, symbols=symbols)
        end_ts = pd.Timestamp(end)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
        return frame[frame["knowable_at"] < end_ts].reset_index(drop=True)


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
        # Filtering (symbol pruning + half-open [start, end) on ts, with naive->UTC normalization)
        # is pushed down to the partitioned dataset in read_bars — no full-snapshot materialization.
        return self.store.read_bars(self.snapshot_id, symbols=symbols, start=start, end=end)
