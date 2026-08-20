from __future__ import annotations

import errno
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from algua.data.files import (
    logical_bars_hash,
    read_partitioned_bars,
    validate_partitioned_bars_dir,
    write_partitioned_bars,
)
from algua.data.manifest import SnapshotManifest
from algua.data.models import Dataset, Kind, SnapshotRecord
from algua.data.schema import empty_bars, to_bar_schema
from algua.data.staging import SnapshotStagingLease
from algua.data.store.identity import build_metadata, compute_snapshot_id
from algua.data.timeframes import validate_timeframe
from algua.primitives.atomic_io import fsync_parents, fsync_tree


class BarsStoreMixin:
    data_dir: Path
    manifest: SnapshotManifest
    _staging: SnapshotStagingLease

    if TYPE_CHECKING:  # provided by the DataStore facade (store/__init__.py); mypy-only declaration
        def get_snapshot(self, snapshot_id: str) -> SnapshotRecord: ...

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
        metadata = build_metadata(
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
        snapshot_id = compute_snapshot_id(metadata, content_hash)

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
        staging_dir, lock_fd, lock_path = self._staging.new_leased_staging()
        try:
            write_partitioned_bars(canon.sort_values(["symbol", "ts"]), staging_dir)
            return self._commit_bars_dir(
                rec, staging_dir, expected_symbols={str(s) for s in canon["symbol"].unique()}
            )
        finally:
            self._staging.release_leased_staging(staging_dir, lock_fd, lock_path)

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
