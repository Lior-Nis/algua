from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from algua.data.files import frame_to_parquet_bytes
from algua.data.fundamentals_schema import (
    empty_fundamentals,
    logical_fundamentals_hash,
    to_fundamentals_schema,
)
from algua.data.manifest import SnapshotManifest
from algua.data.models import Dataset, Kind, SnapshotRecord
from algua.data.store.identity import build_metadata, compute_snapshot_id, normalize_symbols
from algua.primitives.atomic_io import write_bytes_durable


class FundamentalsStoreMixin:
    data_dir: Path
    manifest: SnapshotManifest

    if TYPE_CHECKING:  # provided by the DataStore facade; mypy-only declaration
        def get_snapshot(self, snapshot_id: str) -> SnapshotRecord: ...

    def ingest_fundamentals(
        self,
        *,
        provider: str,
        symbols: list[str],
        as_of: str,
        source: str,
        frame: pd.DataFrame,
        source_metadata: dict[str, str] | None = None,
    ) -> SnapshotRecord:
        """Validate + normalize a tidy fundamentals frame and persist one immutable snapshot.
        `start`/`end` are DERIVED from the data (knowable_at range); every knowable_at must be
        <= `as_of` (you cannot have fetched a record that becomes knowable after you fetched it)."""
        canon = to_fundamentals_schema(frame)
        if canon.empty:
            raise ValueError("cannot ingest an empty fundamentals frame")
        as_of_ts = pd.Timestamp(as_of)
        as_of_ts = (
            as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
        )
        if (canon["knowable_at"] > as_of_ts).any():
            raise ValueError(
                "fundamentals knowable_at must be <= as_of "
                "(cannot ingest a record knowable after the fetch time)"
            )
        start = canon["knowable_at"].min().date().isoformat()
        end = canon["knowable_at"].max().date().isoformat()
        metadata = build_metadata(
            dataset=Dataset.FUNDAMENTALS,
            provider=provider,
            symbols=symbols,
            start=start,
            end=end,
            as_of=as_of,
            source=source,
            kind=Kind.FUNDAMENTALS,
            source_metadata=source_metadata,
        )
        content_hash = logical_fundamentals_hash(canon)
        snapshot_id = compute_snapshot_id(metadata, content_hash)
        existing = self.manifest.find(snapshot_id)
        if existing is not None:
            return existing
        relative_path = (
            Path("snapshots") / metadata.dataset / snapshot_id / "fundamentals.parquet"
        )
        write_bytes_durable(
            frame_to_parquet_bytes(canon),
            self.data_dir / relative_path,
            durable_root=self.data_dir,
        )
        rec = SnapshotRecord(
            snapshot_id=snapshot_id,
            metadata=metadata,
            row_count=len(canon),
            content_hash=content_hash,
            data_path=relative_path,
            created_at=datetime.now(UTC).isoformat(),
            storage_format="parquet",
        )
        return self.manifest.append_if_absent(rec)

    def read_fundamentals(
        self, snapshot_id: str, *, symbols: list[str] | None = None
    ) -> pd.DataFrame:
        """Read a fundamentals snapshot as a validated tidy frame. `symbols` filters in-memory
        (fundamentals are far smaller than bars; partitioned pushdown is deferred). Re-normalizes
        on read so parquet dtype drift cannot escape the schema. Empty => empty_fundamentals()."""
        rec = self.get_snapshot(snapshot_id)
        if rec.dataset != Dataset.FUNDAMENTALS:
            raise ValueError(
                f"snapshot {snapshot_id} is dataset {rec.dataset.value!r}, "
                f"not {Dataset.FUNDAMENTALS.value!r}"
            )
        raw = pd.read_parquet(self.data_dir / rec.data_path)
        if symbols is not None:
            wanted = set(normalize_symbols(symbols))
            raw = raw[raw["symbol"].astype(str).str.upper().isin(wanted)]
        if raw.empty:
            return empty_fundamentals()
        return to_fundamentals_schema(raw)
