"""Global CPU-core budget shared across concurrent `backtest sweep` processes (#327).

Each `backtest sweep` is a separate OS process (a CLI invocation) that fans its grid combos
out over a `ProcessPoolExecutor`. Historically each sized its pool to the WHOLE box
(`min(os.cpu_count(), n_combos)`), so K concurrent sweeps spawned K×cpu_count workers on
cpu_count cores — quadratic oversubscription that thrashes the machine exactly when you scale
out. BLAS pinning (`threadpool_limits(1)`) only stops oversubscription WITHIN one sweep.

This module makes "how many workers may I use" a GRANT admitted from a shared, injected core
budget rather than "all of them":

* ``cpu_budget()`` — the injected total (env ``ALGUA_SWEEP_CPU_BUDGET``, else ``os.cpu_count``).
* ``admit(overridden_count)`` — a context manager. On entry it takes a unique per-sweep
  ``flock`` LEASE, then, under a short-lived global admission lock, SUBTRACTS the worker grants
  already recorded by live sibling sweeps from the budget and records its own grant into its
  lease. It yields the granted worker count. The lease is held for the whole pool lifetime and
  is released (and the marker unlinked) on exit.

Robustness (the properties the design review pinned down):

* Crash-safe: the lease is a kernel ``flock``, freed automatically when the holder dies, so a
  crashed sweep leaves NO stale grant — the next scanner reclaims (unlinks) the orphan.
* Deadlock-free: only ONE lock is ever held while waiting — the global admission lock — and it
  is acquired and released within a bounded, allocation-free critical section; the per-sweep
  lease is non-blocking. No sweep waits for a second resource while holding the first.
* Bounded oversubscription: with the floor of 1 worker per sweep (progress beats a strict cap),
  total granted workers across K live sweeps is at most ``budget + (K - 1)`` — never
  K×cpu_count, never the harmonic ``budget·H_K``.
* fd inheritance: worker processes fork WITHOUT exec, so the lease fd is inherited; a worker
  holding it would keep the ``flock`` alive after the parent dies. Callers pass ``lease_fd``
  to ``close_lease_fd_in_worker`` as the pool ``initializer`` so every worker drops it.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from algua.config.settings import get_settings

_LEASE_DIRNAME = "sweep_leases"
_LEASE_SUFFIX = ".lease"
_ADMISSION_LOCK_NAME = ".admission.lock"


def cpu_budget() -> int:
    """Injected total core budget for ALL concurrent sweeps.

    ``ALGUA_SWEEP_CPU_BUDGET`` (a positive int) if set and valid, else ``os.cpu_count() or 1``.
    A malformed or non-positive override is ignored (falls back to cpu_count) rather than
    crashing a sweep on a typo. Always ``>= 1``.
    """
    raw = os.environ.get("ALGUA_SWEEP_CPU_BUDGET")
    if raw is not None:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value >= 1:
            return value
    return os.cpu_count() or 1


def _lease_dir() -> Path:
    d = get_settings().data_dir / _LEASE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_grant(path: Path) -> int:
    """Read the worker grant recorded in a live sibling lease.

    FAIL CLOSED: a missing/empty/garbled/non-positive grant is treated as the FULL budget (the
    most cores a sweep could be using), so an unreadable sibling can only REDUCE the parallelism
    we admit ourselves — never inflate it into oversubscription.
    """
    try:
        text = path.read_text().strip()
    except OSError:
        return cpu_budget()
    try:
        grant = int(text)
    except ValueError:
        return cpu_budget()
    return grant if grant >= 1 else cpu_budget()


def _sum_live_grants(lease_dir: Path, own_path: Path) -> int:
    """Sum the recorded grants of every LIVE sibling lease (our own path excluded).

    For each ``*.lease`` other than our own, probe its ``flock`` non-blocking on a fresh fd:
      * probe FAILS (held) → a live sweep → add its recorded grant.
      * probe SUCCEEDS (free) → an orphaned marker from a crashed holder → best-effort unlink,
        do not count.
      * any other OSError (ENOLCK / unsupported flock) → fail closed: treat as live at FULL
        budget so we never oversubscribe on an unprobeable sibling.
    Our own lease is NEVER probed or unlinked (self is implicit — we are about to reserve).
    Every probe fd is closed. Returns the summed grant of live siblings.
    """
    own = own_path.resolve()
    used = 0
    for marker in lease_dir.glob(f"*{_LEASE_SUFFIX}"):
        if marker.resolve() == own:
            continue
        try:
            fd = os.open(marker, os.O_RDONLY | os.O_CLOEXEC)
        except FileNotFoundError:
            continue  # vanished between glob and open — gone, ignore
        except OSError:
            used += cpu_budget()  # cannot even open — fail closed
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                used += _read_grant(marker)  # held by a live sweep
                continue
            except OSError:
                used += cpu_budget()  # unprobeable — fail closed
                continue
            # Acquired freely → orphaned marker from a dead holder. Reclaim it.
            fcntl.flock(fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                marker.unlink()
        finally:
            os.close(fd)
    return used


@dataclass
class _Lease:
    grant: int
    lease_fd: int


@contextlib.contextmanager
def admit(overridden_count: int) -> Iterator[_Lease]:
    """Admit this sweep into the global core budget; yield its ``_Lease`` (grant + lease fd).

    On entry: create a unique per-sweep lease marker and hold an exclusive non-blocking
    ``flock`` on it for the WHOLE ``with`` body (the pool's lifetime). Then, under a short-lived
    GLOBAL admission ``flock`` (serialises concurrent admissions so the grant accounting is a
    consistent read-then-reserve), compute::

        grant = clamp(cpu_budget() - sum(live sibling grants), low=1, high=cpu_budget())

    capped at ``overridden_count``, and record it into the lease file. Release the admission
    lock immediately (it must NOT serialise sweep execution). The lease flock is held until
    exit, when it is released and the marker unlinked (best-effort; correctness rests on the
    kernel-freed flock, not the unlink).
    """
    lease_dir = _lease_dir()
    budget = cpu_budget()
    lease_path = lease_dir / f"{os.getpid()}-{uuid.uuid4().hex}{_LEASE_SUFFIX}"
    # O_CLOEXEC so any exec'd grandchild drops it; forked workers still inherit it (no exec),
    # which is why the caller closes it via the pool initializer.
    lease_fd = os.open(lease_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # --- short-lived admission critical section ---
        admission_fd = os.open(
            lease_dir / _ADMISSION_LOCK_NAME, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o644
        )
        try:
            fcntl.flock(admission_fd, fcntl.LOCK_EX)  # blocking; freed on holder death
            used = _sum_live_grants(lease_dir, lease_path)
            grant = max(1, min(budget, budget - used))
            grant = min(grant, max(1, overridden_count))
            os.ftruncate(lease_fd, 0)
            os.pwrite(lease_fd, str(grant).encode("ascii"), 0)
            os.fsync(lease_fd)
        finally:
            os.close(admission_fd)  # releases the admission flock
        # --- admission lock released; sweep now runs under only its own lease ---
        yield _Lease(grant=grant, lease_fd=lease_fd)
    finally:
        with contextlib.suppress(OSError):
            os.close(lease_fd)  # releases the lease flock
        with contextlib.suppress(OSError):
            lease_path.unlink()


def close_lease_fd_in_worker(lease_fd: int) -> None:
    """``ProcessPoolExecutor`` ``initializer``: drop the inherited lease fd in every worker.

    Under the forced ``fork`` start method the worker inherits the parent's lease fd, and while
    that fd stays open the ``flock`` survives even after the parent dies — defeating crash
    reclaim. Closing it here severs that. Tolerant of EBADF/OSError (fd already gone) so it can
    never corrupt worker stdio/pipes.
    """
    with contextlib.suppress(OSError):
        os.close(lease_fd)
