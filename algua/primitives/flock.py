"""THE cross-process flock primitive (spec §4.1) — replaces 8 hand-rolled implementations.

Parameterized on exactly the axes those sites differed on: blocking vs LOCK_NB, OSError
policy (fail-closed default; explicit degrade-to-unlocked opt-in for best-effort curation),
optional JSON holder metadata in the lock body, and inode re-verification for lock files
that must never be replaced externally. flock is advisory and per-open-file-description:
`acquire` always opens a FRESH fd — a cached/shared fd would silently self-grant. The kernel
releases a flock on holder death (even a hard kill), so a crashed holder never wedges the
next acquire.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal


class LockHeld(Exception):
    """A non-blocking acquire found a live holder. `holder` carries the parsed lock-body
    metadata (None when the body is absent/garbled) so the caller can report who holds it."""

    def __init__(self, holder: dict | None = None) -> None:
        super().__init__("lock is held by another process")
        self.holder = holder


class LockReplaced(Exception):
    """The lock file was replaced externally while acquiring (verify_inode=True): the locked
    fd's inode no longer matches the path. The lock file contract says it must never be
    deleted; callers fail distinctly rather than proceed on a phantom lock."""


def acquire(
    path: Path, *, blocking: bool = True, verify_inode: bool = False, retries: int = 5
) -> int:
    """Open a FRESH fd on `path` (creating it 0o644 if absent) and take LOCK_EX. Returns the
    fd; the caller MUST `release(fd)` in a finally. `blocking=False` raises LockHeld on
    contention. `verify_inode=True` re-checks that the path still names the locked inode —
    a mismatch means something replaced the lock file externally; retry bounded, then raise
    LockReplaced."""
    attempts = retries if verify_inode else 1
    for _ in range(attempts):
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            os.close(fd)
            raise LockHeld(read_holder(path)) from exc
        except BaseException:
            os.close(fd)
            raise
        if not verify_inode:
            return fd
        fd_stat = os.fstat(fd)
        try:
            path_stat = os.stat(path)
        except FileNotFoundError:
            path_stat = None
        if path_stat is not None and (
            (path_stat.st_dev, path_stat.st_ino) == (fd_stat.st_dev, fd_stat.st_ino)
        ):
            return fd
        os.close(fd)
    raise LockReplaced(f"lock file {path} was replaced externally while acquiring")


def release(fd: int) -> None:
    """Unlock and close. Closing releases the flock even if LOCK_UN raises."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def read_holder(path: Path) -> dict | None:
    """Recover holder metadata from the lock-file body without taking the lock (flock is
    advisory, so a read needs no lock). None on a missing/empty/garbled body."""
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@contextmanager
def file_lock(
    path: Path,
    *,
    blocking: bool = True,
    metadata: dict | None = None,
    on_oserror: Literal["raise", "proceed"] = "raise",
) -> Iterator[None]:
    """Scoped LOCK_EX on `path`. With `metadata`, the holder identity is written into the
    lock body (fsync'd) on entry and truncated on exit, so a wedged holder is recoverable on
    contention via LockHeld.holder. `on_oserror="proceed"` degrades to UNLOCKED when the
    acquire itself fails with a non-contention OSError (exotic FS / ENOLCK) — ONLY for
    best-effort paths whose writes are individually atomic (the kb-sync curation case);
    everything else keeps the fail-closed default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = acquire(path, blocking=blocking)
    except LockHeld:
        raise
    except OSError:
        if on_oserror == "proceed":
            yield
            return
        raise
    try:
        if metadata is not None:
            body = json.dumps(metadata).encode()
            os.ftruncate(fd, 0)
            os.pwrite(fd, body, 0)
            os.fsync(fd)
        yield
    finally:
        if metadata is not None:
            with contextlib.suppress(OSError):
                os.ftruncate(fd, 0)
        release(fd)


def probe_held(path: Path) -> bool:
    """True iff a live process holds the exclusive flock on `path`. Non-blocking. FAIL
    CLOSED: only a genuinely absent marker (FileNotFoundError) counts as not-held; any other
    open/lock error (ENOLCK, permission, unsupported flock, transient I/O) is treated as
    held, so a cleanup caller never deletes what it cannot prove is abandoned — leftover
    residue is recoverable, a deleted live write is not."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)
