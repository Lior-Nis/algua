from __future__ import annotations

import json
from pathlib import Path

from algua.data.models import SnapshotRecord


class SnapshotManifest:
    """Append-only jsonl manifest of snapshot records.

    Concurrency contract (#158): all writes go through `append_if_absent`, serialized by a
    blocking `fcntl.flock` on the sidecar `<manifest>.lock` file. flock semantics make this a
    LOCAL-LINUX-FILESYSTEM-ONLY contract (no NFS/remote mounts). The lock file is created on
    first use and must NEVER be deleted — unlinking it while a writer holds it silently breaks
    mutual exclusion (cleanup tooling must skip it). Readers are lock-free: a newline is the
    commit marker, so a racing reader sees at worst an uncommitted final tail, which it drops.
    `append_if_absent` may return the concurrent winner's record rather than the caller's —
    the returned record is canonical (its `created_at`/`data_path` stand)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def list_records(self, dataset: str | None = None) -> list[SnapshotRecord]:
        records = self._read_all()
        if dataset is not None:
            records = [r for r in records if r.dataset == dataset]
        return records

    def find(self, snapshot_id: str) -> SnapshotRecord | None:
        for rec in self._read_all():
            if rec.snapshot_id == snapshot_id:
                return rec
        return None

    def append(self, rec: SnapshotRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")

    def _read_all(self) -> list[SnapshotRecord]:
        if not self.path.exists():
            return []
        raw = self.path.read_text(encoding="utf-8")
        return self._parse_committed(self._committed_prefix(raw))

    @staticmethod
    def _committed_prefix(raw: str) -> str:
        """Newline is the commit marker: everything after the last "\\n" (a crash-torn or
        in-flight append) is uncommitted and dropped, EVEN IF it parses as JSON."""
        cut = raw.rfind("\n")
        return raw[: cut + 1] if cut >= 0 else ""

    @staticmethod
    def _parse_committed(committed: str) -> list[SnapshotRecord]:
        records: list[SnapshotRecord] = []
        for line in committed.splitlines():
            if not line.strip():
                continue
            # A committed (newline-terminated) line that fails to parse is real corruption.
            records.append(SnapshotRecord.from_dict(json.loads(line)))
        return records
