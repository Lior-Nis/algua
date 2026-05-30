from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SnapshotMetadata:
    dataset: str
    provider: str
    symbols: tuple[str, ...]
    start: str
    end: str
    as_of: str
    source: str


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    metadata: SnapshotMetadata
    row_count: int | None
    content_hash: str
    data_path: Path
    created_at: str

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "dataset": self.dataset,
            "provider": self.provider,
            "symbols": list(self.symbols),
            "start": self.start,
            "end": self.end,
            "as_of": self.as_of,
            "source": self.source,
            "row_count": self.row_count,
            "content_hash": self.content_hash,
            "data_path": self.data_path.as_posix(),
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
        )
        return cls(
            snapshot_id=str(payload["snapshot_id"]),
            metadata=metadata,
            row_count=None if row_count is None else int(row_count),
            content_hash=str(payload["content_hash"]),
            data_path=Path(str(payload["data_path"])),
            created_at=str(payload["created_at"]),
        )
