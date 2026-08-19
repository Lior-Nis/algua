from __future__ import annotations

import math
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from algua.data.models import Dataset, Kind, SnapshotRecord
from algua.data.store.identity import IdentityMixin, build_metadata, normalize_symbols

if TYPE_CHECKING:
    from algua.backtest.delisting import DelistingRecord


class DelistingsStoreMixin(IdentityMixin):
    def ingest_delistings(
        self,
        *,
        frame: pd.DataFrame,
        as_of: str,
        source: str,
        provider: str = "local",
    ) -> SnapshotRecord:
        """Persist a point-in-time delistings snapshot: columns symbol, delisting_date,
        delisting_value (per-share terminal price in adj_close units, strictly > 0).

        Fails closed on value <= 0 / non-finite (zero-proceeds write-off deferred) and on a
        duplicate (symbol, delisting_date) event."""
        required = {"symbol", "delisting_date", "delisting_value"}
        if not required.issubset(frame.columns):
            raise ValueError(f"delistings frame must have columns {sorted(required)}")
        clean = frame.copy()
        clean["symbol"] = [s.strip().upper() for s in clean["symbol"].astype(str)]
        clean["delisting_date"] = [
            date.fromisoformat(str(d).strip()).isoformat() for d in clean["delisting_date"]
        ]
        clean["delisting_value"] = clean["delisting_value"].astype(float)
        for v in clean["delisting_value"]:
            if not (v > 0) or not math.isfinite(v):
                raise ValueError(
                    "delisting_value must be finite and > 0 (zero-proceeds write-off deferred)"
                )
        if bool(clean.duplicated(subset=["symbol", "delisting_date"]).any()):
            raise ValueError("duplicate (symbol, delisting_date) delisting event")
        symbols = normalize_symbols(list(clean["symbol"]))
        metadata = build_metadata(
            dataset=Dataset.DELISTINGS,
            provider=provider,
            symbols=symbols,
            start=min(clean["delisting_date"]),
            end=max(clean["delisting_date"]),
            as_of=as_of,
            source=source,
            kind=Kind.DELISTING,
        )
        return self._ingest_parquet(
            metadata=metadata, frame=clean.reset_index(drop=True), filename="delistings.parquet"
        )

    def _latest_delistings_record(self, as_of: str | None) -> SnapshotRecord | None:
        """Return the newest DELISTINGS snapshot record as-of `as_of` (or overall if None)."""
        records = self.manifest.list_records(Dataset.DELISTINGS)
        if as_of is not None:
            records = [r for r in records if r.metadata.as_of <= as_of]
        return max(records, key=lambda r: r.metadata.as_of) if records else None

    def latest_delistings_snapshot_id(self, as_of: str | None = None) -> str | None:
        """Return the snapshot_id of the newest DELISTINGS snapshot as-of `as_of`, or None."""
        rec = self._latest_delistings_record(as_of)
        return rec.snapshot_id if rec is not None else None

    def _parse_delistings(self, rec: SnapshotRecord) -> dict[str, list[DelistingRecord]]:
        from algua.backtest.delisting import (  # lazy: keep algua.data off algua.backtest
            DelistingRecord,
        )

        frame = pd.read_parquet(self.data_dir / rec.data_path)
        out: dict[str, list[DelistingRecord]] = {}
        for row in frame.itertuples(index=False):
            out.setdefault(str(row.symbol), []).append(
                DelistingRecord(
                    delisting_date=date.fromisoformat(str(row.delisting_date)),
                    terminal_price=float(row.delisting_value),
                    source=str(rec.metadata.source),
                )
            )
        return out

    def read_delistings(self, as_of: str | None = None) -> dict[str, list[DelistingRecord]]:
        """Point-in-time delistings read: the latest DELISTINGS snapshot with metadata.as_of <=
        `as_of` (or the latest overall when `as_of is None`). Returns
        {symbol: list[DelistingRecord]} (multiple events per symbol allowed). Empty dict if none."""
        latest = self._latest_delistings_record(as_of)
        return self._parse_delistings(latest) if latest is not None else {}

    def read_delistings_with_snapshot(
        self, as_of: str | None = None
    ) -> tuple[dict[str, list[DelistingRecord]], str | None]:
        """Like `read_delistings` but returns the records AND the snapshot_id they came from,
        selected from a SINGLE manifest read so the two can never disagree under a concurrent
        ingest (the records and the stamped provenance id are guaranteed consistent)."""
        latest = self._latest_delistings_record(as_of)
        if latest is None:
            return {}, None
        return self._parse_delistings(latest), latest.snapshot_id
