from __future__ import annotations

import fcntl
import os
import shutil
import time
import uuid
from pathlib import Path


class SnapshotStagingLease:
    """Owns the ``snapshots/_staging`` dir lifecycle: leased staging dirs + stale-residue sweep.

    Extracted from ``DataStore`` (#384) as the cohesive concurrency-plumbing collaborator. It knows
    ONLY about ``data_dir/snapshots/_staging`` — never the manifest, snapshot ids, or metadata.
    Every staging writer takes a lease here; ``clear_staging`` sweeps only residue no writer holds.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def clear_staging(self, *, max_age_seconds: float = 3600.0) -> None:
        """Remove stale staging dirs (crash residue) older than `max_age_seconds`.

        Age alone is unsafe: a staging dir's root mtime is set once at `mkdir` and does NOT refresh
        as writes land in `symbol=<SYM>/` subdirs (or a long file copy), so a >1h in-flight import
        looks "stale" and would be rmtree'd mid-write (#255). So an old dir is swept only when its
        staging LEASE — an exclusive `flock` on the sibling `<uuid>.lock` marker, held for the
        writer's lifetime by `new_leased_staging` (used by EVERY staging writer) — is NOT held. The
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

    def new_leased_staging(self) -> tuple[Path, int, Path]:
        """Take an exclusive `flock` lease on a unique SIBLING `<uuid>.lock` marker, THEN create the
        `_staging/<uuid>` dir under it — so there is never an unleased-dir window (#255). The marker
        is a sibling (not inside the dir) so `_commit_bars_dir`/`os.replace` move a clean snapshot
        dir. Used by EVERY staging writer so `clear_staging` can never rmtree any of them mid-write;
        the lease is released by `release_leased_staging` (caller's finally). The unique path means
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
    def release_leased_staging(staging_dir: Path, lock_fd: int, lock_path: Path) -> None:
        """Release the lease and remove the staging dir + its sibling marker (idempotent — safe
        after a successful commit moved the dir away). Pair with `new_leased_staging` in a try."""
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
        shutil.rmtree(staging_dir, ignore_errors=True)
        lock_path.unlink(missing_ok=True)
