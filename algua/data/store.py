from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


class SnapshotNotFound(LookupError):
    pass


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    dataset: str
    provider: str
    symbols: tuple[str, ...]
    start: str
    end: str
    as_of: str
    source: str
    row_count: int
    content_hash: str
    data_path: Path
    created_at: str

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
        return cls(
            snapshot_id=str(payload["snapshot_id"]),
            dataset=str(payload["dataset"]),
            provider=str(payload["provider"]),
            symbols=tuple(str(s) for s in payload["symbols"]),
            start=str(payload["start"]),
            end=str(payload["end"]),
            as_of=str(payload["as_of"]),
            source=str(payload["source"]),
            row_count=int(payload["row_count"]),
            content_hash=str(payload["content_hash"]),
            data_path=Path(str(payload["data_path"])),
            created_at=str(payload["created_at"]),
        )


class DataStore:
    """Filesystem-backed point-in-time data manifest.

    This first phase-2 slice records immutable local data snapshots. Provider-backed
    ingestion can build on the same manifest contract later.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.manifest_path = data_dir / "manifest.jsonl"

    def ingest_file(
        self,
        *,
        dataset: str,
        provider: str,
        symbols: list[str],
        start: str,
        end: str,
        as_of: str,
        source: str,
        file_path: Path,
    ) -> SnapshotRecord:
        source_path = file_path.expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(str(file_path))

        clean_symbols = _normalize_symbols(symbols)
        _validate_non_empty("dataset", dataset)
        _validate_non_empty("provider", provider)
        _validate_non_empty("source", source)
        _validate_date_bounds(start, end)
        _validate_datetime("as_of", as_of)

        content_hash = _sha256_file(source_path)
        row_count = _count_rows(source_path)
        snapshot_id = _snapshot_id(
            dataset=dataset,
            provider=provider,
            symbols=clean_symbols,
            start=start,
            end=end,
            as_of=as_of,
            source=source,
            content_hash=content_hash,
        )

        existing = self._find_snapshot(snapshot_id)
        if existing is not None:
            return existing

        relative_path = Path("snapshots") / _path_part(dataset) / snapshot_id / source_path.name
        target_path = self.data_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

        rec = SnapshotRecord(
            snapshot_id=snapshot_id,
            dataset=dataset,
            provider=provider,
            symbols=tuple(clean_symbols),
            start=start,
            end=end,
            as_of=as_of,
            source=source,
            row_count=row_count,
            content_hash=content_hash,
            data_path=relative_path,
            created_at=datetime.now(UTC).isoformat(),
        )
        self._append(rec)
        return rec

    def list_snapshots(self, dataset: str | None = None) -> list[SnapshotRecord]:
        records = self._read_manifest()
        if dataset is not None:
            records = [r for r in records if r.dataset == dataset]
        return records

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord:
        rec = self._find_snapshot(snapshot_id)
        if rec is None:
            raise SnapshotNotFound(snapshot_id)
        return rec

    def _find_snapshot(self, snapshot_id: str) -> SnapshotRecord | None:
        for rec in self._read_manifest():
            if rec.snapshot_id == snapshot_id:
                return rec
        return None

    def _read_manifest(self) -> list[SnapshotRecord]:
        if not self.manifest_path.exists():
            return []
        records = []
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(SnapshotRecord.from_dict(json.loads(line)))
        return records

    def _append(self, rec: SnapshotRecord) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")


def _normalize_symbols(symbols: list[str]) -> list[str]:
    clean = sorted({s.strip().upper() for s in symbols if s.strip()})
    if not clean:
        raise ValueError("symbols must not be empty")
    return clean


def _validate_non_empty(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _validate_date_bounds(start: str, end: str) -> None:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    if start_date > end_date:
        raise ValueError("start must be <= end")


def _validate_datetime(name: str, value: str) -> None:
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO datetime") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_rows(path: Path) -> int:
    if path.suffix.lower() != ".csv":
        return 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    return max(len(rows) - 1, 0)


def _snapshot_id(**payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _path_part(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.lower())
    return clean.strip("-") or "dataset"
