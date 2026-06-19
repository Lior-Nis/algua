"""Multi-process tests for SnapshotManifest.append_if_absent (#158).

Real OS processes: flock serialization is cross-process; in-process probes can't prove it."""
from __future__ import annotations

import json
import multiprocessing
import queue
from pathlib import Path

from algua.data.manifest import SnapshotManifest
from algua.data.models import Dataset, Kind, SnapshotMetadata, SnapshotRecord

_CTX = multiprocessing.get_context("fork")


def _record(snapshot_id: str, worker: int) -> SnapshotRecord:
    return SnapshotRecord(
        snapshot_id=snapshot_id,
        metadata=SnapshotMetadata(
            dataset=Dataset.BARS, provider="p", symbols=("AAA",), start="2026-01-01",
            end="2026-01-01", as_of="2026-01-02T00:00:00+00:00", source="s", kind=Kind.BARS,
            timeframe="1d", adjustment="none",
        ),
        row_count=1, content_hash="h",
        data_path=Path(f"snapshots/bars/{snapshot_id}"),
        created_at=f"2026-01-01T00:00:0{worker}+00:00", storage_format="parquet_dataset",
    )


def _appender(manifest_path: str, worker: int, ids: list[str], barrier, errors) -> None:
    try:
        barrier.wait(timeout=30)
        manifest = SnapshotManifest(Path(manifest_path))
        for snapshot_id in ids:
            manifest.append_if_absent(_record(snapshot_id, worker))
    except Exception as exc:  # propagate to the parent — a silent worker is a vacuous pass
        errors.put(f"worker {worker}: {exc!r}")
        raise  # exit non-zero so the parent's assert p.exitcode == 0 catches it deterministically


def test_concurrent_appenders_one_record_per_id(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    n_workers = 4
    shared_ids = [f"id{i:04d}" for i in range(25)]  # every worker appends EVERY id
    barrier = _CTX.Barrier(n_workers)
    errors = _CTX.Queue()
    workers = [
        _CTX.Process(target=_appender, args=(str(manifest_path), w, shared_ids, barrier, errors))
        for w in range(n_workers)
    ]
    for p in workers:
        p.start()
    for p in workers:
        p.join(timeout=60)
        assert p.exitcode == 0
    # Drain the errors queue deterministically — mp.Queue.empty() is documented as unreliable
    collected: list[str] = []
    while True:
        try:
            collected.append(errors.get_nowait())
        except queue.Empty:
            break
    assert collected == [], "\n".join(collected)
    # exactly one committed record per id, file parses cleanly, every line newline-terminated
    raw = manifest_path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    ids = [json.loads(line)["snapshot_id"] for line in raw.splitlines() if line.strip()]
    assert sorted(ids) == sorted(shared_ids)
    recs = SnapshotManifest(manifest_path).list_records()
    assert len(recs) == len(shared_ids)


def _holder(lock_path: str, held, release) -> None:
    import fcntl
    import os

    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    held.set()
    release.wait(timeout=30)
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _blocked_appender(manifest_path: str, attempting, results) -> None:
    manifest = SnapshotManifest(Path(manifest_path))
    attempting.set()
    rec = manifest.append_if_absent(_record("contended", 0))
    results.put(rec.snapshot_id)


def test_appender_blocks_until_lock_holder_releases(tmp_path):
    # Deterministic contention: a holder process owns the flock; the appender must produce
    # NO result while it is held, and complete promptly once released.
    manifest_path = tmp_path / "manifest.jsonl"
    lock_path = str(manifest_path) + ".lock"
    held, release, attempting = _CTX.Event(), _CTX.Event(), _CTX.Event()
    results = _CTX.Queue()
    holder = _CTX.Process(target=_holder, args=(lock_path, held, release))
    holder.start()
    assert held.wait(timeout=10)
    appender = _CTX.Process(
        target=_blocked_appender, args=(str(manifest_path), attempting, results)
    )
    appender.start()
    assert attempting.wait(timeout=10)
    appender.join(timeout=0.5)  # generous beat: appender must still be blocked on the flock
    assert appender.is_alive(), "appender completed while the lock was held — no serialization"
    assert results.empty()
    release.set()
    appender.join(timeout=10)
    assert appender.exitcode == 0
    holder.join(timeout=10)
    assert results.get(timeout=5) == "contended"
    assert [r.snapshot_id for r in SnapshotManifest(manifest_path).list_records()] == ["contended"]
