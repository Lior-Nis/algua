"""Filesystem-backed point-in-time data manifest, carved by dataset (spec §6 item 1).

Each dataset's ingest/read logic lives in its own module (bars.py, bars_streamed.py,
universe.py, delistings.py, fundamentals.py, news.py); DataStore composes them via
mixins and keeps only the truly dataset-agnostic methods (ingest_file, staging/
manifest/verifier delegation, summary) directly on itself.
"""
from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from algua.data.files import count_tabular_rows, sha256_file
from algua.data.manifest import SnapshotManifest
from algua.data.models import Dataset, Kind, SnapshotRecord
from algua.data.staging import SnapshotStagingLease
from algua.data.store.bars_streamed import BarsStreamedStoreMixin
from algua.data.store.delistings import DelistingsStoreMixin
from algua.data.store.fundamentals import FundamentalsStoreMixin
from algua.data.store.identity import (
    SnapshotNotFound,
    build_metadata,
    compute_snapshot_id,
    path_part,
)
from algua.data.store.identity import (
    normalize_symbols as normalize_symbols,
)
from algua.data.store.news import NewsStoreMixin
from algua.data.store.universe import UniverseStoreMixin
from algua.data.verify import SnapshotVerifier
from algua.primitives.atomic_io import fsync_file, fsync_parents


class DataStore(
    BarsStreamedStoreMixin,
    UniverseStoreMixin,
    DelistingsStoreMixin,
    FundamentalsStoreMixin,
    NewsStoreMixin,
):
    """Filesystem-backed point-in-time data manifest.

    This first phase-2 slice records immutable local data snapshots. Provider-backed
    ingestion can build on the same manifest contract later.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.manifest = SnapshotManifest(data_dir / "manifest.jsonl")
        # Cohesive collaborators (#384): staging-lease concurrency plumbing and the power-loss
        # verifier live in their own modules; DataStore composes them and delegates.
        self._staging = SnapshotStagingLease(data_dir)
        self._verifier = SnapshotVerifier(data_dir)

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

        metadata = build_metadata(
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
        staging_dir, lock_fd, lock_path = self._staging.new_leased_staging()
        try:
            # Copy the external source ONCE, then hash/count THE STAGING COPY and publish that
            # exact artifact (#158): a source mutating mid-ingest can no longer commit bytes
            # that don't match content_hash.
            staged = staging_dir / source_path.name
            shutil.copy2(source_path, staged)
            content_hash = sha256_file(staged)
            row_count = count_tabular_rows(staged)
            snapshot_id = compute_snapshot_id(metadata, content_hash)

            existing = self.manifest.find(snapshot_id)
            if existing is not None:
                return existing

            relative_path = (
                Path("snapshots") / path_part(metadata.dataset) / snapshot_id / source_path.name
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
            self._staging.release_leased_staging(staging_dir, lock_fd, lock_path)

    def clear_staging(self, *, max_age_seconds: float = 3600.0) -> None:
        """Sweep stale staging residue (#255). Delegates to the staging-lease collaborator (#384),
        which owns the ``_staging`` dir lifecycle; kept on the facade for the CLI call site."""
        self._staging.clear_staging(max_age_seconds=max_age_seconds)

    def list_snapshots(self, dataset: Dataset | None = None) -> list[SnapshotRecord]:
        return self.manifest.list_records(dataset)

    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord:
        rec = self.manifest.find(snapshot_id)
        if rec is None:
            raise SnapshotNotFound(snapshot_id)
        return rec

    def verify_snapshot(self, rec: SnapshotRecord) -> None:
        """Power-loss read-back of one snapshot's payload (#184). Delegates the per-format dispatch
        to the verifier collaborator (#384); kept on the facade for existing callers/tests."""
        self._verifier.verify_snapshot(rec)

    def verify_snapshots(self, snapshot_id: str | None = None) -> list[dict[str, Any]]:
        """Verify one snapshot (`snapshot_id`) or all committed snapshots. Returns one result
        row per snapshot: ``{snapshot_id, dataset, storage_format, ok, error}``. Never raises for
        a damaged payload — the damage is captured in the row (`ok=False`); the caller decides
        the exit code. A missing `snapshot_id` itself raises `SnapshotNotFound`.

        The record-resolution + aggregation loop stay here (calling ``self.verify_snapshot`` /
        ``self.get_snapshot`` / ``self.list_snapshots``) so the dynamic self-dispatch an override or
        monkeypatch relies on is preserved; only the per-format read-back moved to the verifier."""
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
