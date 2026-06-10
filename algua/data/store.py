from __future__ import annotations

import errno
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

from algua.data.files import (
    BARS_STREAMED_HASH_ALGO,
    compose_bars_symbol_hash,
    copy_snapshot,
    count_tabular_rows,
    frame_to_parquet_bytes,
    logical_bars_hash,
    read_partitioned_bars,
    sha256_bytes,
    sha256_file,
    write_bytes_snapshot,
    write_partitioned_bars,
)
from algua.data.fundamentals_schema import (
    empty_fundamentals,
    logical_fundamentals_hash,
    to_fundamentals_schema,
)
from algua.data.manifest import SnapshotManifest
from algua.data.models import (
    Dataset,
    Kind,
    SnapshotMetadata,
    SnapshotRecord,
    UniverseSnapshot,
)
from algua.data.news_schema import (
    empty_news,
    explode_news_symbols,
    logical_news_hash,
    to_news_schema,
)
from algua.data.schema import empty_bars, to_bar_schema


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
        canon = to_bar_schema(frame).reset_index().rename(columns={"timestamp": "ts"})
        content_hash = logical_bars_hash(canon)
        snapshot_id = _snapshot_id(metadata, content_hash)

        existing = self.manifest.find(snapshot_id)
        if existing is not None:
            return existing

        relative_path = Path("snapshots") / metadata.dataset / snapshot_id
        write_partitioned_bars(
            canon.sort_values(["symbol", "ts"]), self.data_dir / relative_path
        )
        rec = SnapshotRecord(
            snapshot_id=snapshot_id,
            metadata=metadata,
            row_count=len(canon),
            content_hash=content_hash,
            data_path=relative_path,
            created_at=datetime.now(UTC).isoformat(),
            storage_format="parquet_dataset",
        )
        self.manifest.append(rec)
        return rec

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
        """Stream per-symbol bar chunks into one hive-partitioned-by-symbol bars snapshot.

        Crash-safe: each chunk is normalized via `to_bar_schema` (so output is schema-valid) and
        written as its own `symbol=<SYM>/` partition under a UUID staging dir (one chunk in memory
        at a time -> bounded RAM). The per-symbol logical leaf hashes are composed (sorted by
        symbol, so order-independent) into the snapshot `content_hash`; the snapshot_id is content-
        addressed. Commit is adopt-on-target-exists: dedup on snapshot_id (idempotent re-ingest),
        then `os.replace` the staging dir onto the immutable snapshot dir — if that target dir
        already exists (an orphan from a crash between rename and manifest-append, or a concurrent
        winner) re-check the manifest and otherwise adopt it. The manifest record is appended last,
        with `storage_format="parquet_dataset"` so `read_bars` serves it with pushdown.

        Cross-chunk integrity: each symbol must appear in exactly one chunk — the method rejects a
        symbol that recurs in a later chunk (so each `symbol=<SYM>/` partition is written once and
        the snapshot is globally unique on (timestamp, symbol) given each chunk is internally
        schema-valid). The FirstRate importer satisfies this by yielding one chunk per symbol.

        Note: when `start`/`end` are given, the coverage check is span-only (observed range covers
        the requested endpoints); it does not detect interior gaps.
        """
        staging_dir = self.data_dir / "snapshots" / "_staging" / uuid.uuid4().hex
        staging_dir.mkdir(parents=True, exist_ok=True)
        row_count = 0
        observed_min: pd.Timestamp | None = None
        observed_max: pd.Timestamp | None = None
        seen_symbols_set: set[str] = set()
        leaves: list[tuple[str, int, str]] = []
        try:
            for chunk in chunks:
                chunk_canon = (
                    to_bar_schema(chunk).reset_index().rename(columns={"timestamp": "ts"})
                )
                chunk_symbols = set(chunk_canon["symbol"].unique())
                clash = chunk_symbols & seen_symbols_set
                if clash:
                    raise ValueError(
                        f"symbol(s) {sorted(clash)} appear in more than one chunk; streamed "
                        "ingest requires each symbol's bars in a single contiguous chunk"
                    )
                seen_symbols_set |= chunk_symbols
                write_partitioned_bars(chunk_canon, staging_dir)
                row_count += len(chunk_canon)
                cmin = chunk_canon["ts"].min()
                cmax = chunk_canon["ts"].max()
                observed_min = cmin if observed_min is None else min(observed_min, cmin)
                observed_max = cmax if observed_max is None else max(observed_max, cmax)
                for sym, group in chunk_canon.groupby("symbol"):
                    leaves.append((str(sym), len(group), logical_bars_hash(group)))
            if not leaves or row_count == 0:
                raise ValueError("no bars to ingest (empty chunk stream)")

            if observed_min is None or observed_max is None:  # unreachable: leaves => loop ran
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
            meta_extra["content_hash_algorithm"] = BARS_STREAMED_HASH_ALGO

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
            content_hash = compose_bars_symbol_hash(leaves)
            snapshot_id = _snapshot_id(metadata, content_hash)

            relative_path = Path("snapshots") / metadata.dataset / snapshot_id  # a DIR now
            target = self.data_dir / relative_path
            existing = self.manifest.find(snapshot_id)
            if existing is not None:
                return existing
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(staging_dir, target)
            except OSError as exc:
                # Adopt ONLY the expected "target dir already exists and is non-empty" failure (an
                # orphan from a crash between rename and manifest-append, or a concurrent winner).
                # Re-raise anything else (permission, I/O, cross-device) — those are not adoptable.
                if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST) or not target.is_dir():
                    raise
                found = self.manifest.find(snapshot_id)
                if found is not None:
                    return found
                # else: adopt the orphan target dir (content-addressed → identical content) by
                # falling through to append the manifest record
            rec = SnapshotRecord(
                snapshot_id=snapshot_id,
                metadata=metadata,
                row_count=row_count,
                content_hash=content_hash,
                data_path=relative_path,
                created_at=datetime.now(UTC).isoformat(),
                storage_format="parquet_dataset",
            )
            self.manifest.append(rec)
            return rec
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def list_snapshots(self, dataset: str | None = None) -> list[SnapshotRecord]:
        return self.manifest.list_records(dataset)

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord:
        rec = self.manifest.find(snapshot_id)
        if rec is None:
            raise SnapshotNotFound(snapshot_id)
        return rec

    def read_bars(
        self,
        snapshot_id: str,
        *,
        symbols: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Read a bars snapshot as a bar-schema DataFrame, pushing `symbols` + half-open
        `[start, end)` filters down to the partitioned parquet dataset (issue #130). Any filter left
        as None is unbounded. Empty result => the contract's empty-but-typed frame."""
        rec = self.get_snapshot(snapshot_id)  # raises SnapshotNotFound
        if rec.dataset != Dataset.BARS.value:
            raise ValueError(
                f"snapshot {snapshot_id} is dataset {rec.dataset!r}, not {Dataset.BARS.value!r}"
            )
        if rec.storage_format != "parquet_dataset":
            raise ValueError(
                f"snapshot {snapshot_id} is a legacy single-file bars snapshot "
                f"({rec.storage_format!r}); re-ingest under the partitioned layout"
            )
        raw = read_partitioned_bars(
            self.data_dir / rec.data_path, symbols=symbols, start=start, end=end
        )
        if raw.empty:
            return empty_bars()
        return to_bar_schema(raw)

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
        metadata = _metadata(
            dataset=Dataset.FUNDAMENTALS.value,
            provider=provider,
            symbols=symbols,
            start=start,
            end=end,
            as_of=as_of,
            source=source,
            kind=Kind.FUNDAMENTALS.value,
            source_metadata=source_metadata,
        )
        content_hash = logical_fundamentals_hash(canon)
        snapshot_id = _snapshot_id(metadata, content_hash)
        existing = self.manifest.find(snapshot_id)
        if existing is not None:
            return existing
        relative_path = (
            Path("snapshots") / metadata.dataset / snapshot_id / "fundamentals.parquet"
        )
        write_bytes_snapshot(frame_to_parquet_bytes(canon), self.data_dir, relative_path)
        rec = SnapshotRecord(
            snapshot_id=snapshot_id,
            metadata=metadata,
            row_count=len(canon),
            content_hash=content_hash,
            data_path=relative_path,
            created_at=datetime.now(UTC).isoformat(),
            storage_format="parquet",
        )
        self.manifest.append(rec)
        return rec

    def read_fundamentals(
        self, snapshot_id: str, *, symbols: list[str] | None = None
    ) -> pd.DataFrame:
        """Read a fundamentals snapshot as a validated tidy frame. `symbols` filters in-memory
        (fundamentals are far smaller than bars; partitioned pushdown is deferred). Re-normalizes
        on read so parquet dtype drift cannot escape the schema. Empty => empty_fundamentals()."""
        rec = self.get_snapshot(snapshot_id)
        if rec.dataset != Dataset.FUNDAMENTALS.value:
            raise ValueError(
                f"snapshot {snapshot_id} is dataset {rec.dataset!r}, "
                f"not {Dataset.FUNDAMENTALS.value!r}"
            )
        raw = pd.read_parquet(self.data_dir / rec.data_path)
        if symbols is not None:
            wanted = set(normalize_symbols(symbols))
            raw = raw[raw["symbol"].astype(str).str.upper().isin(wanted)]
        if raw.empty:
            return empty_fundamentals()
        return to_fundamentals_schema(raw)

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
        metadata = _metadata(
            dataset=Dataset.NEWS.value,
            provider=provider,
            symbols=symbols,
            start=start,
            end=end,
            as_of=as_of,
            source=provider,
            kind=Kind.NEWS.value,
            source_metadata=derived,
        )
        content_hash = logical_news_hash(canon)
        snapshot_id = _snapshot_id(metadata, content_hash)
        existing = self.manifest.find(snapshot_id)
        if existing is not None:
            return existing
        relative_path = Path("snapshots") / metadata.dataset / snapshot_id / "news.parquet"
        write_bytes_snapshot(frame_to_parquet_bytes(canon), self.data_dir, relative_path)
        rec = SnapshotRecord(
            snapshot_id=snapshot_id,
            metadata=metadata,
            row_count=len(canon),
            content_hash=content_hash,
            data_path=relative_path,
            created_at=datetime.now(UTC).isoformat(),
            storage_format="parquet",
        )
        self.manifest.append(rec)
        return rec

    def read_news(self, snapshot_id: str, *, symbols: list[str] | None = None) -> pd.DataFrame:
        """Read a news snapshot as a validated tidy frame. `symbols` filters in-memory.
        Re-normalizes on read (idempotent) so parquet dtype drift cannot escape the schema.
        Empty => empty_news()."""
        rec = self.get_snapshot(snapshot_id)
        if rec.dataset != Dataset.NEWS.value:
            raise ValueError(
                f"snapshot {snapshot_id} is dataset {rec.dataset!r}, not {Dataset.NEWS.value!r}"
            )
        raw = pd.read_parquet(self.data_dir / rec.data_path)
        if symbols is not None:
            wanted = set(normalize_symbols(symbols))
            raw = raw[raw["symbol"].astype(str).str.upper().isin(wanted)]
        if raw.empty:
            return empty_news()
        return to_news_schema(raw)

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


