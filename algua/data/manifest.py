from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path

from algua.data.models import SnapshotRecord


class ManifestLockReplacedError(RuntimeError):
    """The sidecar manifest lock file was replaced/unlinked externally.

    This is environmental corruption (something deleted `manifest.jsonl.lock` out from under
    live writers — the lock file must NEVER be removed), not an ingest failure."""


_LOCK_ACQUIRE_RETRIES = 5
_REPAIR_TEMP_SUFFIX = ".repair-"


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

    def append_if_absent(self, rec: SnapshotRecord) -> SnapshotRecord:
        """Append `rec` unless a record with its snapshot_id is already committed; return the
        committed record (the caller's `rec`, or the concurrent winner's). Repairs any
        uncommitted tail (crash residue) before appending. The ONLY manifest write path."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = self._acquire_lock()
        try:
            raw = self.path.read_bytes() if self.path.exists() else b""
            committed = self._committed_prefix(raw)
            for existing in self._parse_committed(committed.decode("utf-8")):
                if existing.snapshot_id == rec.snapshot_id:
                    return existing
            self._clean_stale_repair_temps()
            if committed != raw:
                self._repair(committed)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            return rec
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _acquire_lock(self) -> int:
        """Blocking LOCK_EX on the sidecar lock file via a FRESH fd per call (flock is
        per-open-file-description: a cached/shared fd would silently self-grant). After
        acquiring, verify the path still names the locked inode — a mismatch means something
        replaced the lock file externally; retry bounded, then fail distinctly."""
        lock_path = self.path.with_name(self.path.name + ".lock")
        for _ in range(_LOCK_ACQUIRE_RETRIES):
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                fd_stat = os.fstat(fd)
                try:
                    path_stat = os.stat(lock_path)
                except FileNotFoundError:
                    path_stat = None
                if path_stat is not None and (
                    (path_stat.st_dev, path_stat.st_ino) == (fd_stat.st_dev, fd_stat.st_ino)
                ):
                    return fd
            except BaseException:
                os.close(fd)
                raise
            os.close(fd)
        raise ManifestLockReplacedError(
            f"lock file {lock_path} was replaced externally while acquiring; it must never "
            "be deleted (see SnapshotManifest contract)"
        )

    def _repair(self, committed: bytes) -> None:
        """Replace the manifest with its committed prefix via temp + atomic rename. Never
        truncate in place: a lock-free reader mid-read on a shrinking inode could splice
        old+new bytes into a malformed non-final line; the rename keeps the old inode
        complete, so a reader sees the whole old or whole new file."""
        temp_fd, temp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=f"{self.path.name}{_REPAIR_TEMP_SUFFIX}", suffix=".tmp"
        )
        try:
            with os.fdopen(temp_fd, "wb") as fh:
                fh.write(committed)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temp_name, self.path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def _clean_stale_repair_temps(self) -> None:
        """Best-effort sweep of repair temps left by a crashed writer (we hold the lock, so
        no live writer owns one)."""
        for stale in self.path.parent.glob(f"{self.path.name}{_REPAIR_TEMP_SUFFIX}*"):
            try:
                stale.unlink()
            except OSError:
                continue

    def _read_all(self) -> list[SnapshotRecord]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        return self._parse_committed(self._committed_prefix(raw).decode("utf-8"))

    @staticmethod
    def _committed_prefix(raw: bytes) -> bytes:
        """Newline is the commit marker: everything after the last b"\\n" (a crash-torn or
        in-flight append) is uncommitted and dropped, EVEN IF it parses as JSON.

        The cut must happen on BYTES, not decoded text: if the file has a torn tail that
        splits a multi-byte UTF-8 sequence, decoding first raises UnicodeDecodeError before
        the tail can be dropped, defeating self-healing. We cut at the last b"\\n", then
        decode only the committed prefix (which is guaranteed to be complete UTF-8 lines)."""
        cut = raw.rfind(b"\n")
        return raw[: cut + 1] if cut >= 0 else b""

    @staticmethod
    def _parse_committed(committed: str) -> list[SnapshotRecord]:
        records: list[SnapshotRecord] = []
        for line in committed.split("\n"):
            if not line.strip():
                continue
            # A committed (newline-terminated) line that fails to parse is real corruption.
            records.append(SnapshotRecord.from_dict(json.loads(line)))
        return records
