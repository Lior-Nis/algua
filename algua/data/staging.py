from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

from algua.primitives import flock


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
                    # FAIL CLOSED (#255): flock.probe_held treats anything but a genuinely
                    # absent marker as held, so cleanup never deletes a dir it cannot prove is
                    # abandoned — leftover residue is recoverable, a deleted live write is not.
                    if flock.probe_held(staging / f"{child.name}.lock"):
                        continue  # in-progress import holds the lease (#255)
                    shutil.rmtree(child, ignore_errors=True)
                    (staging / f"{child.name}.lock").unlink(missing_ok=True)
                elif child.suffix == ".lock":
                    # An orphan lease marker (its staging dir already gone): clean it unless a dir
                    # still pairs with it (handled above) or a writer still holds it (#255,
                    # fail-closed via flock.probe_held).
                    if (staging / child.stem).is_dir() or flock.probe_held(child):
                        continue
                    child.unlink(missing_ok=True)
            except OSError:
                continue

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
        lock_fd: int | None = None
        try:
            lock_fd = flock.acquire(lock_path)
            staging_dir = staging_root / name
            staging_dir.mkdir()
        except BaseException:
            # flock.acquire already closes its fd internally when IT is what raised, so
            # lock_fd is unbound in that case — only release here if acquire succeeded but a
            # later step (mkdir) failed, to avoid a double-close/unbound-variable reference.
            if lock_fd is not None:
                flock.release(lock_fd)
            lock_path.unlink(missing_ok=True)
            shutil.rmtree(staging_root / name, ignore_errors=True)
            raise
        return staging_dir, lock_fd, lock_path

    @staticmethod
    def release_leased_staging(staging_dir: Path, lock_fd: int, lock_path: Path) -> None:
        """Release the lease and remove the staging dir + its sibling marker (idempotent — safe
        after a successful commit moved the dir away). Pair with `new_leased_staging` in a try."""
        flock.release(lock_fd)
        shutil.rmtree(staging_dir, ignore_errors=True)
        lock_path.unlink(missing_ok=True)
