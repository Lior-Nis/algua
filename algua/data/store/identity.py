from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from algua.data.files import frame_to_parquet_bytes, sha256_bytes
from algua.data.manifest import SnapshotManifest
from algua.data.models import Dataset, Kind, SnapshotMetadata, SnapshotRecord
from algua.primitives.atomic_io import write_bytes_durable


class SnapshotNotFound(LookupError):
    pass


class IdentityMixin:
    """Shared identity/parquet-publish plumbing for the dataset mixins that need it
    (universe, delistings). Declares the collaborator attributes it assumes the
    composing DataStore provides — never assigned here, only annotated."""

    data_dir: Path
    manifest: SnapshotManifest

    def _ingest_parquet(
        self,
        *,
        metadata: SnapshotMetadata,
        frame: pd.DataFrame,
        filename: str,
        conflict_check: Callable[[list[SnapshotRecord], SnapshotRecord], None] | None = None,
    ) -> SnapshotRecord:
        """Hash a frame to parquet, dedup on snapshot id, write it, and append the manifest record.

        The shared tail of ``ingest_bars`` and ``ingest_universe``: both differ only in how they
        build ``metadata``/``frame`` and the on-disk ``filename``. The dataset path component is
        ``metadata.dataset`` (already a clean enum value for both parquet datasets).
        """
        payload = frame_to_parquet_bytes(frame)
        content_hash = sha256_bytes(payload)
        snapshot_id = compute_snapshot_id(metadata, content_hash)

        existing = self.manifest.find(snapshot_id)
        if existing is not None:
            return existing

        relative_path = Path("snapshots") / metadata.dataset / snapshot_id / filename
        write_bytes_durable(
            payload, self.data_dir / relative_path, durable_root=self.data_dir
        )
        rec = SnapshotRecord(
            snapshot_id=snapshot_id,
            metadata=metadata,
            row_count=len(frame),
            content_hash=content_hash,
            data_path=relative_path,
            created_at=datetime.now(UTC).isoformat(),
            storage_format="parquet",
        )
        return self.manifest.append_if_absent(rec, conflict_check=conflict_check)


def normalize_symbols(symbols: list[str]) -> list[str]:
    """Canonicalize a symbol list: strip, upper-case, de-duplicate, sort.

    The single source of truth for symbol normalization across the data layer and CLI.
    """
    clean = sorted({s.strip().upper() for s in symbols if s.strip()})
    if not clean:
        raise ValueError("symbols must not be empty")
    return clean


def build_metadata(
    *,
    dataset: Dataset,
    provider: str,
    symbols: list[str],
    start: str,
    end: str,
    as_of: str,
    source: str,
    kind: Kind = Kind.FILE,
    timeframe: str | None = None,
    adjustment: str | None = None,
    universe: str | None = None,
    source_metadata: dict[str, str] | None = None,
) -> SnapshotMetadata:
    validate_non_empty("provider", provider)
    validate_non_empty("source", source)
    validate_date_bounds(start, end)
    validate_datetime("as_of", as_of)
    return SnapshotMetadata(
        dataset=dataset,
        provider=provider,
        symbols=tuple(normalize_symbols(symbols)),
        start=start,
        end=end,
        as_of=as_of,
        source=source,
        kind=kind,
        timeframe=timeframe,
        adjustment=adjustment,
        universe=universe,
        source_metadata=source_metadata or {},
    )


def validate_non_empty(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def validate_date_bounds(start: str, end: str) -> None:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    if start_date > end_date:
        raise ValueError("start must be <= end")


def validate_datetime(name: str, value: str) -> None:
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO datetime") from exc


def compute_snapshot_id(metadata: SnapshotMetadata, content_hash: str) -> str:
    payload: dict[str, Any] = {
        "dataset": metadata.dataset,
        "provider": metadata.provider,
        "symbols": list(metadata.symbols),
        "start": metadata.start,
        "end": metadata.end,
        "as_of": metadata.as_of,
        "source": metadata.source,
        "kind": metadata.kind,
        "timeframe": metadata.timeframe,
        "adjustment": metadata.adjustment,
        "universe": metadata.universe,
        "source_metadata": metadata.source_metadata or {},
        "content_hash": content_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def path_part(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.lower())
    return clean.strip("-") or "dataset"
