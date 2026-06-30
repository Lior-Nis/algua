from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from algua.data.files import (
    BARS_STREAMED_HASH_ALGO,
    compose_bars_symbol_hash,
    count_tabular_rows,
    frame_to_parquet_bytes,
    fsync_file,
    fsync_parents,
    fsync_tree,
    logical_bars_hash,
    parquet_dataset_row_count,
    parquet_file_row_count,
    read_partitioned_bars,
    sha256_bytes,
    sha256_file,
    validate_partitioned_bars_dir,
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
from algua.data.timeframes import validate_timeframe

if TYPE_CHECKING:
    from algua.backtest.delisting import DelistingRecord


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
        dataset: Dataset,
        provider: str,
        symbols: list[str],
        start: str,
        end: str,
        as_of: str,
        source: str,
        file_path: Path,
        kind: Kind = Kind.FILE,
        timeframe: str | None = None,
        adjustment: str | None = None,
        universe: str | None = None,
        source_metadata: dict[str, str] | None = None,
    ) -> SnapshotRecord:
        """Register a local file as an immutable snapshot via staged copy + atomic publish.

        Note: `snapshot_id` excludes the source filename but `data_path` includes it — a
        same-content-different-filename race resolves to the winner's canonical record; the
        loser's published file may remain as a benign orphan in the same content-addressed
        snapshot dir; reads always resolve via `record.data_path`."""
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
        staging_dir, lock_fd, lock_path = self._new_leased_staging()
        try:
            # Copy the external source ONCE, then hash/count THE STAGING COPY and publish that
            # exact artifact (#158): a source mutating mid-ingest can no longer commit bytes
            # that don't match content_hash.
            staged = staging_dir / source_path.name
            shutil.copy2(source_path, staged)
            content_hash = sha256_file(staged)
            row_count = count_tabular_rows(staged)
            snapshot_id = _snapshot_id(metadata, content_hash)

            existing = self.manifest.find(snapshot_id)
            if existing is not None:
                return existing

            relative_path = (
                Path("snapshots") / _path_part(metadata.dataset) / snapshot_id / source_path.name
            )
            target = self.data_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            fsync_file(staged)  # copy2 does not fsync; make the bytes durable before publish
            os.replace(staged, target)
            fsync_parents(target, stop_at=self.data_dir)  # rename entry + new ancestors durable

            rec = SnapshotRecord(
                snapshot_id=snapshot_id,
                metadata=metadata,
                row_count=row_count,
                content_hash=content_hash,
                data_path=relative_path,
                created_at=datetime.now(UTC).isoformat(),
                storage_format=source_path.suffix.lower().lstrip(".") or "file",
            )
            return self.manifest.append_if_absent(rec)
        finally:
            self._release_leased_staging(staging_dir, lock_fd, lock_path)

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
        validate_timeframe(timeframe)
        metadata = _metadata(
            dataset=Dataset.BARS,
            provider=provider,
            symbols=symbols,
            start=start,
            end=end,
            as_of=as_of,
            source=source,
            kind=Kind.BARS,
            timeframe=timeframe,
            adjustment=adjustment,
            source_metadata=source_metadata,
        )
        canon = (
            to_bar_schema(frame, timeframe=timeframe)
            .reset_index()
            .rename(columns={"timestamp": "ts"})
        )
        content_hash = logical_bars_hash(canon)
        snapshot_id = _snapshot_id(metadata, content_hash)

        existing = self.manifest.find(snapshot_id)
        if existing is not None:
            return existing

        relative_path = Path("snapshots") / metadata.dataset / snapshot_id
        rec = SnapshotRecord(
            snapshot_id=snapshot_id,
            metadata=metadata,
            row_count=len(canon),
            content_hash=content_hash,
            data_path=relative_path,
            created_at=datetime.now(UTC).isoformat(),
            storage_format="parquet_dataset",
        )
        staging_dir, lock_fd, lock_path = self._new_leased_staging()
        try:
            write_partitioned_bars(canon.sort_values(["symbol", "ts"]), staging_dir)
            return self._commit_bars_dir(
                rec, staging_dir, expected_symbols={str(s) for s in canon["symbol"].unique()}
            )
        finally:
            self._release_leased_staging(staging_dir, lock_fd, lock_path)

    def _commit_bars_dir(
        self, rec: SnapshotRecord, staging_dir: Path, *, expected_symbols: set[str]
    ) -> SnapshotRecord:
        """Atomically publish a fully-written staging dir at `rec.data_path` and commit the
        manifest record (#158). On rename collision (target dir already exists): if the id is
        already committed, return that record; otherwise VALIDATE the existing dir (legacy
        direct-write ingest could have left a partial dir) and adopt it. Fails closed on
        validation mismatch — never deletes the suspect dir. The caller owns `staging_dir`
        creation and `finally`-cleanup.

        Power-loss durable (#184): on the publish branch the staging tree is fsynced before
        the rename and the target's parent chain after; on the adoption branch the same
        barrier (tree + parent chain) runs before the manifest append, since a concurrent or
        prior writer may have renamed the dir into place without fsyncing it and we are about
        to commit it."""
        target = self.data_dir / rec.data_path
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            fsync_tree(staging_dir)  # all part-files + dir entries durable before publish
            os.replace(staging_dir, target)
            fsync_parents(target, stop_at=self.data_dir)
        except OSError as exc:
            # Adopt ONLY the expected "target dir already exists and is non-empty" failure.
            # Re-raise anything else (permission, I/O, cross-device).
            if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST) or not target.is_dir():
                raise
            found = self.manifest.find(rec.snapshot_id)
            if found is not None:
                return found
            validate_partitioned_bars_dir(
                target,
                expected_row_count=rec.row_count or 0,
                expected_symbols=expected_symbols,
            )
            # Independent durability barrier: the adopter is about to commit the manifest.
            fsync_tree(target)
            fsync_parents(target, stop_at=self.data_dir)
        return self.manifest.append_if_absent(rec)

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
            dataset=Dataset.UNIVERSES,
            provider=provider,
            symbols=clean_symbols,
            start=effective_date,
            end=effective_date,
            as_of=as_of,
            source=source,
            kind=Kind.UNIVERSE,
            universe=universe,
            source_metadata=source_metadata,
        )

        # Universes are immutable on EVERY ingest path (#263): a same-(universe, effective_date)
        # change with different membership aborts at the manifest commit (append_if_absent, under
        # the manifest lock so it is race-safe), so no conflicting record is ever committed — not
        # just caught later at read time. (A rejected ingest may leave an inert orphan parquet that
        # the manifest never references — the shared _ingest_parquet publish-then-commit behavior;
        # it feeds no read.) Corrections require a new universe name.
        def conflict_check(committed, rec):
            for other in committed:
                if (
                    other.dataset == Dataset.UNIVERSES
                    and other.metadata.universe == universe
                    and other.metadata.start == effective_date
                    and other.content_hash != rec.content_hash
                ):
                    raise ValueError(
                        f"universe {universe!r} already has a DIFFERENT membership on "
                        f"{effective_date} (immutable; corrections require a new name)"
                    )

        return self._ingest_parquet(
            metadata=metadata, frame=frame, filename="universe.parquet",
            conflict_check=conflict_check,
        )

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
        metadata = _metadata(
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
        return self.manifest.append_if_absent(rec, conflict_check=conflict_check)

    def clear_staging(self, *, max_age_seconds: float = 3600.0) -> None:
        """Remove stale staging dirs (crash residue) older than `max_age_seconds`.

        Age alone is unsafe: a staging dir's root mtime is set once at `mkdir` and does NOT refresh
        as writes land in `symbol=<SYM>/` subdirs (or a long file copy), so a >1h in-flight import
        looks "stale" and would be rmtree'd mid-write (#255). So an old dir is swept only when its
        staging LEASE — an exclusive `flock` on the sibling `<uuid>.lock` marker, held for the
        writer's lifetime by `_new_leased_staging` (used by EVERY staging writer) — is NOT held. The
        lease auto-releases on the writer's death (even a hard kill), so true crash residue reads as
        unheld and is swept; a live writer's dir reads as held and is spared. Each run also cleans
        its own dir in a `finally`; this only sweeps what a hard kill left behind.
        """
        staging = self.data_dir / "snapshots" / "_staging"
        if not staging.exists():
            return
        cutoff = time.time() - max_age_seconds
        for child in staging.iterdir():
            try:
                if child.stat().st_mtime >= cutoff:
                    continue  # fresh — a just-started import may own it
                if child.is_dir():
                    if self._lock_held(staging / f"{child.name}.lock"):
                        continue  # in-progress import holds the lease (#255)
                    shutil.rmtree(child, ignore_errors=True)
                    (staging / f"{child.name}.lock").unlink(missing_ok=True)
                elif child.suffix == ".lock":
                    # An orphan lease marker (its staging dir already gone): clean it unless a dir
                    # still pairs with it (handled above) or a writer still holds it.
                    if (staging / child.stem).is_dir() or self._lock_held(child):
                        continue
                    child.unlink(missing_ok=True)
            except OSError:
                continue

    @staticmethod
    def _lock_held(lock_path: Path) -> bool:
        """True iff a live writer currently holds the exclusive `flock` on `lock_path` (an
        in-progress staging writer). A non-blocking probe. FAIL CLOSED: only a genuinely absent
        marker (`FileNotFoundError`) counts as not-held (sweepable); any other open/lock error
        (ENOLCK, permission, unsupported flock, transient I/O) is treated as held, so cleanup never
        deletes a dir it cannot prove is abandoned — leftover residue is recoverable, a deleted live
        write is not. flock is freed by the kernel on the holder's death, so a crash is unheld."""
        try:
            fd = os.open(lock_path, os.O_RDWR)
        except FileNotFoundError:
            return False  # no lease marker — true crash residue or a pre-lease dir
        except OSError:
            return True  # can't even open it — refuse to sweep (fail closed)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True  # a writer holds it
        except OSError:
            return True  # lock probe failed — refuse to sweep (fail closed)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False  # acquired freely → not held
        finally:
            os.close(fd)

    def _new_leased_staging(self) -> tuple[Path, int, Path]:
        """Take an exclusive `flock` lease on a unique SIBLING `<uuid>.lock` marker, THEN create the
        `_staging/<uuid>` dir under it — so there is never an unleased-dir window (#255). The marker
        is a sibling (not inside the dir) so `_commit_bars_dir`/`os.replace` move a clean snapshot
        dir. Used by EVERY staging writer so `clear_staging` can never rmtree any of them mid-write;
        the lease is released by `_release_leased_staging` (caller's finally). The unique path means
        LOCK_EX never contends; the kernel frees the lease on writer death. Self-cleaning: a failure
        before the caller takes over closes the fd and removes the marker/dir, leaking nothing."""
        staging_root = self.data_dir / "snapshots" / "_staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        name = uuid.uuid4().hex
        lock_path = staging_root / f"{name}.lock"
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            staging_dir = staging_root / name
            staging_dir.mkdir()
        except BaseException:
            os.close(lock_fd)
            lock_path.unlink(missing_ok=True)
            shutil.rmtree(staging_root / name, ignore_errors=True)
            raise
        return staging_dir, lock_fd, lock_path

    @staticmethod
    def _release_leased_staging(staging_dir: Path, lock_fd: int, lock_path: Path) -> None:
        """Release the lease and remove the staging dir + its sibling marker (idempotent — safe
        after a successful commit moved the dir away). Pair with `_new_leased_staging` in a try."""
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
        shutil.rmtree(staging_dir, ignore_errors=True)
        lock_path.unlink(missing_ok=True)

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
        addressed. Commit goes through the shared `_commit_bars_dir` protocol (#158): dedup on
        snapshot_id (idempotent re-ingest), then `os.replace` the staging dir onto the immutable
        snapshot dir — if that target dir already exists (an orphan from a crash between rename and
        manifest-append, or a concurrent winner) re-check the manifest, otherwise VALIDATE the
        existing dir (fail closed on a partial/foreign dir) and adopt it. The manifest record is
        committed last via `append_if_absent`, with `storage_format="parquet_dataset"` so
        `read_bars` serves it with pushdown.

        Cross-chunk integrity: each symbol must appear in exactly one chunk — the method rejects a
        symbol that recurs in a later chunk (so each `symbol=<SYM>/` partition is written once and
        the snapshot is globally unique on (timestamp, symbol) given each chunk is internally
        schema-valid). The FirstRate importer satisfies this by yielding one chunk per symbol.

        Note: when `start`/`end` are given, the coverage check is span-only (observed range covers
        the requested endpoints); it does not detect interior gaps.
        """
        validate_timeframe(timeframe)
        # Lease the staging dir for the whole import so a concurrent clear_staging can't rmtree it
        # mid-write — the staging-root mtime is set once at mkdir and never refreshes (#255).
        staging_dir, lock_fd, lock_path = self._new_leased_staging()
        row_count = 0
        observed_min: pd.Timestamp | None = None
        observed_max: pd.Timestamp | None = None
        seen_symbols_set: set[str] = set()
        leaves: list[tuple[str, int, str]] = []
        try:
            for chunk in chunks:
                chunk_canon = (
                    to_bar_schema(chunk, timeframe=timeframe)
                    .reset_index()
                    .rename(columns={"timestamp": "ts"})
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
                dataset=Dataset.BARS,
                provider=provider,
                symbols=symbols,
                start=observed_start,
                end=observed_end,
                as_of=as_of,
                source=source,
                kind=Kind.BARS,
                timeframe=timeframe,
                adjustment=adjustment,
                source_metadata=meta_extra,
            )
            content_hash = compose_bars_symbol_hash(leaves)
            snapshot_id = _snapshot_id(metadata, content_hash)

            relative_path = Path("snapshots") / metadata.dataset / snapshot_id  # a DIR
            existing = self.manifest.find(snapshot_id)
            if existing is not None:
                return existing
            rec = SnapshotRecord(
                snapshot_id=snapshot_id,
                metadata=metadata,
                row_count=row_count,
                content_hash=content_hash,
                data_path=relative_path,
                created_at=datetime.now(UTC).isoformat(),
                storage_format="parquet_dataset",
            )
            return self._commit_bars_dir(rec, staging_dir, expected_symbols=seen_symbols_set)
        finally:
            self._release_leased_staging(staging_dir, lock_fd, lock_path)

    def list_snapshots(self, dataset: Dataset | None = None) -> list[SnapshotRecord]:
        return self.manifest.list_records(dataset)

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord:
        rec = self.manifest.find(snapshot_id)
        if rec is None:
            raise SnapshotNotFound(snapshot_id)
        return rec

    def verify_snapshot(self, rec: SnapshotRecord) -> None:
        """Power-loss read-back of one snapshot's payload (#184). Reads the bytes back to prove
        they are durable and decompressible, and checks the row count against the record. Raises
        on any damage (the caller decides how to surface it). Dispatch by `storage_format`:

        - ``parquet_dataset`` (bars): full read of every partition; summed rows == ``row_count``.
        - ``parquet`` (universe/fundamentals/news, or a ``.parquet`` via ``ingest_file``): full
          read of the single file; ``num_rows == row_count``. Readability check, NOT a
          content-hash recompute. For a ``.parquet`` ingested via ``ingest_file`` (byte-hash
          ``content_hash``) this is a strictly weaker check than the ``else`` branch's
          ``sha256_file`` comparison — by design, since verify targets power-loss readability,
          not tampering.
        - anything else (``ingest_file`` csv/generic): ``sha256_file == content_hash`` (a full
          read). Fails closed: a record whose ``content_hash`` is not a byte hash would report a
          (false) failure rather than a false pass — that signals the dispatch needs extending.
        """
        target = self.data_dir / rec.data_path
        fmt = rec.storage_format
        if fmt == "parquet_dataset":
            if not target.is_dir():
                raise ValueError(f"snapshot {rec.snapshot_id}: payload dir missing at {target}")
            rows = parquet_dataset_row_count(target)
            if rec.row_count is not None and rows != rec.row_count:
                raise ValueError(
                    f"snapshot {rec.snapshot_id}: read {rows} rows, expected {rec.row_count}"
                )
        elif fmt == "parquet":
            if not target.is_file():
                raise ValueError(f"snapshot {rec.snapshot_id}: payload file missing at {target}")
            rows = parquet_file_row_count(target)
            if rec.row_count is not None and rows != rec.row_count:
                raise ValueError(
                    f"snapshot {rec.snapshot_id}: read {rows} rows, expected {rec.row_count}"
                )
        else:
            if not target.is_file():
                raise ValueError(f"snapshot {rec.snapshot_id}: payload file missing at {target}")
            actual = sha256_file(target)
            if actual != rec.content_hash:
                raise ValueError(
                    f"snapshot {rec.snapshot_id}: content hash {actual} != {rec.content_hash}"
                )

    def verify_snapshots(self, snapshot_id: str | None = None) -> list[dict[str, Any]]:
        """Verify one snapshot (`snapshot_id`) or all committed snapshots. Returns one result
        row per snapshot: ``{snapshot_id, dataset, storage_format, ok, error}``. Never raises for
        a damaged payload — the damage is captured in the row (`ok=False`); the caller decides
        the exit code. A missing `snapshot_id` itself raises `SnapshotNotFound`."""
        records = (
            [self.get_snapshot(snapshot_id)] if snapshot_id is not None else self.list_snapshots()
        )
        results: list[dict[str, Any]] = []
        for rec in records:
            row: dict[str, Any] = {
                "snapshot_id": rec.snapshot_id,
                "dataset": rec.dataset.value,
                "storage_format": rec.storage_format,
                "ok": True,
                "error": None,
            }
            try:
                self.verify_snapshot(rec)
            except (OSError, ValueError) as exc:  # read-back/integrity failures; bugs propagate
                row["ok"] = False
                row["error"] = str(exc)
            results.append(row)
        return results

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
        if rec.dataset != Dataset.BARS:
            raise ValueError(
                f"snapshot {snapshot_id} is dataset {rec.dataset.value!r}, "
                f"not {Dataset.BARS.value!r}"
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
        # rec.metadata.timeframe may be None for a legacy snapshot that recorded no timeframe; that
        # skips only the daily UTC-midnight check (ingest is the gate), not the rest of the schema.
        return to_bar_schema(raw, timeframe=rec.metadata.timeframe)

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
            for rec in self.manifest.list_records(Dataset.UNIVERSES)
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
                rec.dataset.value,
                {
                    "dataset": rec.dataset.value,
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


