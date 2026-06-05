from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from algua.data.files import (
    copy_snapshot,
    count_tabular_rows,
    frame_to_parquet_bytes,
    sha256_bytes,
    sha256_file,
    write_bytes_snapshot,
)
from algua.data.manifest import SnapshotManifest
from algua.data.models import Dataset, Kind, SnapshotMetadata, SnapshotRecord
from algua.data.schema import to_bar_schema


class SnapshotNotFound(LookupError):
    pass


class DataStore:
    """Filesystem-backed point-in-time data manifest.

    This first phase-2 slice records immutable local data snapshots. Provider-backed
    ingestion can build on the same manifest contract later.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.manifest = SnapshotManifest(data_dir / "manifest.jsonl")

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
        kind: str = Kind.FILE.value,
        timeframe: str | None = None,
        adjustment: str | None = None,
        universe: str | None = None,
        source_metadata: dict[str, str] | None = None,
    ) -> SnapshotRecord:
        source_path = file_path.expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(str(file_path))

        metadata = _metadata(
            dataset=dataset,
            provider=provider,
            symbols=symbols,
            start=start,
            end=end,
            as_of=as_of,
            source=source,
            kind=kind,
            timeframe=timeframe,
            adjustment=adjustment,
            universe=universe,
            source_metadata=source_metadata,
        )
        content_hash = sha256_file(source_path)
        row_count = count_tabular_rows(source_path)
        snapshot_id = _snapshot_id(metadata, content_hash)

        existing = self.manifest.find(snapshot_id)
        if existing is not None:
            return existing

        relative_path = (
            Path("snapshots") / _path_part(metadata.dataset) / snapshot_id / source_path.name
        )
        copy_snapshot(source_path, self.data_dir, relative_path)

        rec = SnapshotRecord(
            snapshot_id=snapshot_id,
            metadata=metadata,
            row_count=row_count,
            content_hash=content_hash,
            data_path=relative_path,
            created_at=datetime.now(UTC).isoformat(),
            storage_format=source_path.suffix.lower().lstrip(".") or "file",
        )
        self.manifest.append(rec)
        return rec

    def ingest_bars(
        self,
        *,
        provider: str,
        symbols: list[str],
        start: str,
        end: str,
        as_of: str,
        source: str,
        frame: pd.DataFrame,
        timeframe: str = "1d",
        adjustment: str = "none",
        source_metadata: dict[str, str] | None = None,
    ) -> SnapshotRecord:
        metadata = _metadata(
            dataset=Dataset.BARS.value,
            provider=provider,
            symbols=symbols,
            start=start,
            end=end,
            as_of=as_of,
            source=source,
            kind=Kind.BARS.value,
            timeframe=timeframe,
            adjustment=adjustment,
            source_metadata=source_metadata,
        )
        return self._ingest_parquet(
            metadata=metadata, frame=_normalize_bar_frame(frame), filename="bars.parquet"
        )

    def ingest_universe(
        self,
        *,
        universe: str,
        symbols: list[str],
        effective_date: str,
        as_of: str,
        source: str,
        provider: str = "local",
        source_metadata: dict[str, str] | None = None,
    ) -> SnapshotRecord:
        clean_symbols = normalize_symbols(symbols)
        frame = pd.DataFrame(
            {"effective_date": effective_date, "universe": universe, "symbol": clean_symbols}
        )
        metadata = _metadata(
            dataset=Dataset.UNIVERSES.value,
            provider=provider,
            symbols=clean_symbols,
            start=effective_date,
            end=effective_date,
            as_of=as_of,
            source=source,
            kind=Kind.UNIVERSE.value,
            universe=universe,
            source_metadata=source_metadata,
        )
        return self._ingest_parquet(
            metadata=metadata, frame=frame, filename="universe.parquet"
        )

    def _ingest_parquet(
        self, *, metadata: SnapshotMetadata, frame: pd.DataFrame, filename: str
    ) -> SnapshotRecord:
        """Hash a frame to parquet, dedup on snapshot id, write it, and append the manifest record.

        The shared tail of ``ingest_bars`` and ``ingest_universe``: both differ only in how they
        build ``metadata``/``frame`` and the on-disk ``filename``. The dataset path component is
        ``metadata.dataset`` (already a clean enum value for both parquet datasets).
        """
        payload = frame_to_parquet_bytes(frame)
        content_hash = sha256_bytes(payload)
        snapshot_id = _snapshot_id(metadata, content_hash)

        existing = self.manifest.find(snapshot_id)
        if existing is not None:
            return existing

        relative_path = Path("snapshots") / metadata.dataset / snapshot_id / filename
        write_bytes_snapshot(payload, self.data_dir, relative_path)
        rec = SnapshotRecord(
            snapshot_id=snapshot_id,
            metadata=metadata,
            row_count=len(frame),
            content_hash=content_hash,
            data_path=relative_path,
            created_at=datetime.now(UTC).isoformat(),
            storage_format="parquet",
        )
        self.manifest.append(rec)
        return rec

    def list_snapshots(self, dataset: str | None = None) -> list[SnapshotRecord]:
        return self.manifest.list_records(dataset)

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord:
        rec = self.manifest.find(snapshot_id)
        if rec is None:
            raise SnapshotNotFound(snapshot_id)
        return rec

    def read_bars(self, snapshot_id: str) -> pd.DataFrame:
        """Read a bars snapshot back as a bar-schema DataFrame (tz-aware UTC timestamp index)."""
        rec = self.get_snapshot(snapshot_id)  # raises SnapshotNotFound
        if rec.dataset != Dataset.BARS.value:
            raise ValueError(
                f"snapshot {snapshot_id} is dataset {rec.dataset!r}, not {Dataset.BARS.value!r}"
            )
        frame = pd.read_parquet(self.data_dir / rec.data_path)
        return to_bar_schema(frame)

    def summary(self) -> dict[str, Any]:
        records = self.list_snapshots()
        datasets: dict[str, dict[str, Any]] = {}
        for rec in records:
            item = datasets.setdefault(
                rec.dataset,
                {
                    "dataset": rec.dataset,
                    "snapshots": 0,
                    "symbols": set(),
                    "start": rec.start,
                    "end": rec.end,
                    "providers": set(),
                    "storage_formats": set(),
                },
            )
            item["snapshots"] += 1
            item["symbols"].update(rec.symbols)
            item["start"] = min(item["start"], rec.start)
            item["end"] = max(item["end"], rec.end)
            item["providers"].add(rec.provider)
            item["storage_formats"].add(rec.storage_format)
        return {
            "snapshots": len(records),
            "datasets": [
                {
                    **item,
                    "symbols": sorted(item["symbols"]),
                    "providers": sorted(item["providers"]),
                    "storage_formats": sorted(item["storage_formats"]),
                }
                for item in sorted(datasets.values(), key=lambda d: d["dataset"])
            ],
        }


def normalize_symbols(symbols: list[str]) -> list[str]:
    """Canonicalize a symbol list: strip, upper-case, de-duplicate, sort.

    The single source of truth for symbol normalization across the data layer and CLI.
    """
    clean = sorted({s.strip().upper() for s in symbols if s.strip()})
    if not clean:
        raise ValueError("symbols must not be empty")
    return clean


def _metadata(
    *,
    dataset: str,
    provider: str,
    symbols: list[str],
    start: str,
    end: str,
    as_of: str,
    source: str,
    kind: str = Kind.FILE.value,
    timeframe: str | None = None,
    adjustment: str | None = None,
    universe: str | None = None,
    source_metadata: dict[str, str] | None = None,
) -> SnapshotMetadata:
    _validate_non_empty("dataset", dataset)
    _validate_non_empty("provider", provider)
    _validate_non_empty("source", source)
    _validate_date_bounds(start, end)
    _validate_datetime("as_of", as_of)
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


def _snapshot_id(metadata: SnapshotMetadata, content_hash: str) -> str:
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


def _path_part(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.lower())
    return clean.strip("-") or "dataset"


def _normalize_bar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("bars frame must not be empty")
    return to_bar_schema(frame).reset_index()
