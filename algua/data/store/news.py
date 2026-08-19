from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from algua.data.files import frame_to_parquet_bytes
from algua.data.manifest import SnapshotManifest
from algua.data.models import Dataset, Kind, SnapshotRecord
from algua.data.news_schema import (
    empty_news,
    explode_news_symbols,
    logical_news_hash,
    to_news_schema,
)
from algua.data.store.identity import build_metadata, compute_snapshot_id, normalize_symbols
from algua.primitives.atomic_io import write_bytes_durable


class NewsStoreMixin:
    data_dir: Path
    manifest: SnapshotManifest

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord:
        # provided by the DataStore facade; stub for mypy only
        raise NotImplementedError

    def ingest_news(
        self,
        *,
        provider: str,
        as_of: str,
        frame: pd.DataFrame,
        source_metadata: dict[str, str] | None = None,
    ) -> SnapshotRecord:
        """Explode + normalize a per-article news frame and persist one immutable snapshot.
        `start`/`end` and the covered symbol/source sets are DERIVED from the data; every
        knowable_at must be <= `as_of`. `metadata.source` is the ingest `provider` label; the
        derived row-source/symbol sets live in `source_metadata` (multi-source dataset)."""
        canon = to_news_schema(explode_news_symbols(frame))
        if canon.empty:
            raise ValueError("cannot ingest an empty news frame")
        as_of_ts = pd.Timestamp(as_of)
        as_of_ts = (
            as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
        )
        if (canon["knowable_at"] > as_of_ts).any():
            raise ValueError(
                "news knowable_at must be <= as_of "
                "(cannot ingest a record knowable after the fetch time)"
            )
        start = canon["knowable_at"].min().date().isoformat()
        end = canon["knowable_at"].max().date().isoformat()
        symbols = sorted(canon["symbol"].unique())
        sources = sorted(canon["source"].unique())
        # Derived coverage is authoritative — a caller cannot overwrite row_sources/row_symbols
        # with values that lie about the data (GATE-2); their other keys are preserved.
        derived = {
            **(source_metadata or {}),
            "row_sources": ",".join(sources),
            "row_symbols": ",".join(symbols),
        }
        metadata = build_metadata(
            dataset=Dataset.NEWS,
            provider=provider,
            symbols=symbols,
            start=start,
            end=end,
            as_of=as_of,
            source=provider,
            kind=Kind.NEWS,
            source_metadata=derived,
        )
        content_hash = logical_news_hash(canon)
        snapshot_id = compute_snapshot_id(metadata, content_hash)
        existing = self.manifest.find(snapshot_id)
        if existing is not None:
            return existing
        relative_path = Path("snapshots") / metadata.dataset / snapshot_id / "news.parquet"
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

    def read_news(self, snapshot_id: str, *, symbols: list[str] | None = None) -> pd.DataFrame:
        """Read a news snapshot as a validated tidy frame. `symbols` filters in-memory.
        Re-normalizes on read (idempotent) so parquet dtype drift cannot escape the schema.
        Empty => empty_news()."""
        rec = self.get_snapshot(snapshot_id)
        if rec.dataset != Dataset.NEWS:
            raise ValueError(
                f"snapshot {snapshot_id} is dataset {rec.dataset.value!r}, "
                f"not {Dataset.NEWS.value!r}"
            )
        raw = pd.read_parquet(self.data_dir / rec.data_path)
        if symbols is not None:
            wanted = set(normalize_symbols(symbols))
            raw = raw[raw["symbol"].astype(str).str.upper().isin(wanted)]
        if raw.empty:
            return empty_news()
        return to_news_schema(raw)
