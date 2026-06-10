from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


class Dataset(StrEnum):
    """Dataset routing key — the manifest `dataset` field and snapshot path component."""

    BARS = "bars"
    UNIVERSES = "universes"
    FUNDAMENTALS = "fundamentals"
    NEWS = "news"


class Kind(StrEnum):
    """Snapshot `kind` — the provenance of a snapshot's payload."""

    BARS = "bars"
    UNIVERSE = "universe"
    FILE = "file"
    FUNDAMENTALS = "fundamentals"
    NEWS = "news"


@dataclass(frozen=True)
class SnapshotMetadata:
    dataset: str
    provider: str
    symbols: tuple[str, ...]
    start: str
    end: str
    as_of: str
    source: str
    kind: str = "file"
    timeframe: str | None = None
    adjustment: str | None = None
    universe: str | None = None
    source_metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    metadata: SnapshotMetadata
    row_count: int | None
    content_hash: str
    data_path: Path
    created_at: str
    storage_format: str = "file"
    schema_version: int = SCHEMA_VERSION

    @property
    def dataset(self) -> str:
        return self.metadata.dataset

    @property
    def provider(self) -> str:
        return self.metadata.provider

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.metadata.symbols

    @property
    def start(self) -> str:
        return self.metadata.start

    @property
    def end(self) -> str:
        return self.metadata.end

    @property
    def as_of(self) -> str:
        return self.metadata.as_of

    @property
    def source(self) -> str:
        return self.metadata.source

    @property
    def kind(self) -> str:
        return self.metadata.kind

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "dataset": self.dataset,
            "provider": self.provider,
            "symbols": list(self.symbols),
            "start": self.start,
            "end": self.end,
            "as_of": self.as_of,
            "source": self.source,
            "kind": self.kind,
            "timeframe": self.metadata.timeframe,
            "adjustment": self.metadata.adjustment,
            "universe": self.metadata.universe,
            "source_metadata": self.metadata.source_metadata or {},
            "row_count": self.row_count,
            "content_hash": self.content_hash,
            "data_path": self.data_path.as_posix(),
            "storage_format": self.storage_format,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SnapshotRecord:
        row_count = payload["row_count"]
        metadata = SnapshotMetadata(
            dataset=str(payload["dataset"]),
            provider=str(payload["provider"]),
            symbols=tuple(str(s) for s in payload["symbols"]),
            start=str(payload["start"]),
            end=str(payload["end"]),
            as_of=str(payload["as_of"]),
            source=str(payload["source"]),
            kind=str(payload.get("kind", "file")),
            timeframe=_optional_str(payload.get("timeframe")),
            adjustment=_optional_str(payload.get("adjustment")),
            universe=_optional_str(payload.get("universe")),
            source_metadata={str(k): str(v) for k, v in payload.get("source_metadata", {}).items()},
        )
        return cls(
            snapshot_id=str(payload["snapshot_id"]),
            metadata=metadata,
            row_count=None if row_count is None else int(row_count),
            content_hash=str(payload["content_hash"]),
            data_path=Path(str(payload["data_path"])),
            created_at=str(payload["created_at"]),
            storage_format=str(payload.get("storage_format", "file")),
            schema_version=int(payload["schema_version"]),
        )


@dataclass(frozen=True)
class UniverseSnapshot:
    """One point-in-time universe-membership snapshot: the set of `symbols` that constituted a
    named universe as of `effective_date`. A time-varying universe is a timeline of these (one
    per effective date) sharing the same universe NAME. Data-layer-only value object — it never
    crosses into the backtest engine (the engine receives a plain date->symbols mapping)."""

    snapshot_id: str
    effective_date: date
    symbols: frozenset[str]


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
