from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from algua.data.files import (
    copy_snapshot,
    count_tabular_rows,
    frame_to_parquet_bytes,
    sha256_bytes,
    sha256_file,
    write_bytes_snapshot,
)
from algua.data.manifest import SnapshotManifest
from algua.data.models import (
    Dataset,
    Kind,
    SnapshotMetadata,
    SnapshotRecord,
    UniverseSnapshot,
)
from algua.data.schema import to_bar_schema

# Above this row count, a streamed import warns and self-marks "not servable until #130", because
# the read path still fully materializes a snapshot. Not a hard cap — deep history is the point.
IMPORT_WARN_ROWS = 5_000_000


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

    def clear_staging(self, *, max_age_seconds: float = 3600.0) -> None:
        """Remove stale streamed-import staging dirs (crash residue) older than `max_age_seconds`.

        Age-based so a concurrent in-progress import (its own fresh UUID staging dir) is never
        deleted out from under its writer. Each `ingest_bars_streamed` run already cleans its own
        staging dir in a `finally`; this only sweeps residue left by a hard kill.
        """
        staging = self.data_dir / "snapshots" / "_staging"
        if not staging.exists():
            return
        cutoff = time.time() - max_age_seconds
        for child in staging.iterdir():
            try:
                if child.stat().st_mtime < cutoff:
                    shutil.rmtree(child, ignore_errors=True)
            except OSError:
                continue

    def ingest_bars_streamed(
        self,
        *,
        provider: str,
        symbols: list[str],
        as_of: str,
        source: str,
        chunks: Iterable[pd.DataFrame],
        timeframe: str = "1d",
        adjustment: str = "split_div",
        start: str | None = None,
        end: str | None = None,
        source_metadata: dict[str, str] | None = None,
    ) -> SnapshotRecord:
        """Stream per-symbol bar chunks into one consolidated bars snapshot.

        Crash-safe: stream -> staging file, hash, dedup on snapshot_id (idempotent re-ingest),
        atomic rename into the immutable snapshot path, append the manifest last. Each chunk is
        normalized via `to_bar_schema` (so output is schema-valid) and written as one row group,
        in the order received (the importer yields canonical sorted-symbol order — required for a
        stable `snapshot_id`).

        Cross-chunk integrity: each symbol must appear in exactly one chunk — the method rejects a
        symbol that recurs in a later chunk (so the consolidated snapshot is globally unique on
        (timestamp, symbol) given each chunk is internally schema-valid). The FirstRate importer
        satisfies this by yielding one chunk per symbol.

        Note: when `start`/`end` are given, the coverage check is span-only (observed range covers
        the requested endpoints); it does not detect interior gaps.
        """
        staging_dir = self.data_dir / "snapshots" / "_staging" / uuid.uuid4().hex
        staging_file = staging_dir / "bars.parquet"
        staging_dir.mkdir(parents=True, exist_ok=True)
        writer: pq.ParquetWriter | None = None
        row_count = 0
        observed_min: pd.Timestamp | None = None
        observed_max: pd.Timestamp | None = None
        seen_symbols_set: set[str] = set()
        try:
            for chunk in chunks:
                normalized = to_bar_schema(chunk).reset_index()  # columns: timestamp, *BAR_COLUMNS
                chunk_symbols = set(normalized["symbol"].unique())
                clash = chunk_symbols & seen_symbols_set
                if clash:
                    raise ValueError(
                        f"symbol(s) {sorted(clash)} appear in more than one chunk; streamed "
                        "ingest requires each symbol's bars in a single contiguous chunk"
                    )
                seen_symbols_set |= chunk_symbols
                table = pa.Table.from_pandas(
                    normalized, preserve_index=False
                ).replace_schema_metadata(None)
                if writer is None:
                    writer = pq.ParquetWriter(
                        staging_file, table.schema, compression="snappy", version="2.6"
                    )
                writer.write_table(table)
                row_count += len(normalized)
                cmin = normalized["timestamp"].min()
                cmax = normalized["timestamp"].max()
                observed_min = cmin if observed_min is None else min(observed_min, cmin)
                observed_max = cmax if observed_max is None else max(observed_max, cmax)
            if writer is None:
                raise ValueError("no bars to ingest (empty chunk stream)")
            writer.close()
            writer = None

            if observed_min is None or observed_max is None:  # unreachable: writer set => loop ran
                raise ValueError("no bars to ingest (empty chunk stream)")
            observed_start = observed_min.date().isoformat()
            observed_end = observed_max.date().isoformat()
            if start is not None or end is not None:
                if (start is not None and observed_start > start) or (
                    end is not None and observed_end < end
                ):
                    raise ValueError(
                        f"observed coverage [{observed_start}, {observed_end}] does not cover "
                        f"requested [{start}, {end}]"
                    )

            meta_extra = dict(source_metadata or {})
            if start is not None:
                meta_extra["requested_start"] = start
            if end is not None:
                meta_extra["requested_end"] = end
            meta_extra["observed_start"] = observed_start
            meta_extra["observed_end"] = observed_end
            if row_count >= IMPORT_WARN_ROWS:
                meta_extra["servable"] = "deferred-130"

            metadata = _metadata(
                dataset=Dataset.BARS.value,
                provider=provider,
                symbols=symbols,
                start=observed_start,
                end=observed_end,
                as_of=as_of,
                source=source,
                kind=Kind.BARS.value,
                timeframe=timeframe,
                adjustment=adjustment,
                source_metadata=meta_extra,
            )
            content_hash = sha256_file(staging_file)
            snapshot_id = _snapshot_id(metadata, content_hash)

            existing = self.manifest.find(snapshot_id)
            if existing is not None:
                return existing

            relative_path = Path("snapshots") / metadata.dataset / snapshot_id / "bars.parquet"
            target = self.data_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_file, target)
            rec = SnapshotRecord(
                snapshot_id=snapshot_id,
                metadata=metadata,
                row_count=row_count,
                content_hash=content_hash,
                data_path=relative_path,
                created_at=datetime.now(UTC).isoformat(),
                storage_format="parquet",
            )
            self.manifest.append(rec)
            return rec
        finally:
            if writer is not None:
                writer.close()
            shutil.rmtree(staging_dir, ignore_errors=True)

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

    def read_universe(self, universe: str) -> list[UniverseSnapshot]:
        """Read a named universe's point-in-time membership timeline.

        A time-varying universe is recorded as one membership snapshot per `effective_date`,
        all sharing the universe NAME (see `ingest_universe`). This reads every snapshot tagged
        with `universe`, normalizes its symbols, and returns the timeline sorted ascending by
        `effective_date`. The as-of-date-t membership is the snapshot with the greatest
        `effective_date <= t` (empty before the earliest effective date) — that resolution is the
        consumer's, but the timeline this returns is what makes it leak-free.

        Raises ``ValueError`` if two snapshots share an `effective_date` but disagree on
        membership: the as-of answer for that date would be ambiguous, so we refuse rather than
        silently pick one.
        """
        records = [
            rec
            for rec in self.manifest.list_records(Dataset.UNIVERSES.value)
            if rec.metadata.universe == universe
        ]
        by_date: dict[date, UniverseSnapshot] = {}
        for rec in records:
            frame = pd.read_parquet(self.data_dir / rec.data_path)
            eff = date.fromisoformat(str(frame["effective_date"].iloc[0]))
            symbols = frozenset(normalize_symbols([str(s) for s in frame["symbol"]]))
            existing = by_date.get(eff)
            if existing is not None and existing.symbols != symbols:
                raise ValueError(
                    f"ambiguous as-of membership for universe {universe!r} on {eff.isoformat()}: "
                    f"two snapshots disagree ({sorted(existing.symbols)} vs {sorted(symbols)})"
                )
            by_date[eff] = UniverseSnapshot(
                snapshot_id=rec.snapshot_id, effective_date=eff, symbols=symbols
            )
        return [by_date[eff] for eff in sorted(by_date)]

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
