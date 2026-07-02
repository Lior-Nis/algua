"""Tests for the global sweep CPU-core budget (#327).

The budget stops K concurrent `backtest sweep` processes each grabbing cpu_count workers
(K×cpu_count oversubscription). These tests exercise the grant-based admission, the flock-lease
liveness/crash-reclaim, self-exclusion, and the worst-case bound.
"""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from algua.backtest import core_budget as cb


@pytest.fixture(autouse=True)
def _isolate_data_dir(monkeypatch, tmp_path):
    # Point data_dir (and thus the lease dir) at a per-test tmp tree.
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ALGUA_SWEEP_CPU_BUDGET", raising=False)
    yield


def test_cpu_budget_env_override(monkeypatch):
    monkeypatch.setenv("ALGUA_SWEEP_CPU_BUDGET", "5")
    assert cb.cpu_budget() == 5


def test_cpu_budget_ignores_invalid_override(monkeypatch):
    for bad in ("0", "-3", "abc", "2.5"):
        monkeypatch.setenv("ALGUA_SWEEP_CPU_BUDGET", bad)
        assert cb.cpu_budget() == (os.cpu_count() or 1)


def test_cpu_budget_default_is_cpu_count(monkeypatch):
    monkeypatch.delenv("ALGUA_SWEEP_CPU_BUDGET", raising=False)
    assert cb.cpu_budget() == (os.cpu_count() or 1)


def test_solo_sweep_gets_full_budget(monkeypatch):
    monkeypatch.setenv("ALGUA_SWEEP_CPU_BUDGET", "8")
    with cb.admit(overridden_count=100) as lease:
        # No live siblings -> grant is the full budget.
        assert lease.grant == 8


def test_grant_capped_at_combo_count(monkeypatch):
    monkeypatch.setenv("ALGUA_SWEEP_CPU_BUDGET", "8")
    with cb.admit(overridden_count=3) as lease:
        assert lease.grant == 3


def test_lease_marker_created_and_removed_on_exit(monkeypatch):
    monkeypatch.setenv("ALGUA_SWEEP_CPU_BUDGET", "4")
    lease_dir = Path(os.environ["ALGUA_DATA_DIR"]) / "sweep_leases"
    with cb.admit(overridden_count=4):
        markers = list(lease_dir.glob("*.lease"))
        assert len(markers) == 1
        assert markers[0].read_text().strip() == "4"
    # Cleaned up on exit.
    assert list(lease_dir.glob("*.lease")) == []


def _hold_lease_script(lease_dir: str, grant_text: str) -> str:
    # A standalone process that creates a lease marker, records `grant_text` (the raw grant string,
    # possibly garbled to exercise fail-closed reads), holds an exclusive flock on it, signals
    # readiness by printing the path, then sleeps until killed.
    return f"""
import fcntl, os, sys, time, uuid
d = {lease_dir!r}
os.makedirs(d, exist_ok=True)
p = os.path.join(d, f"{{os.getpid()}}-{{uuid.uuid4().hex}}.lease")
fd = os.open(p, os.O_CREAT | os.O_RDWR, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
os.pwrite(fd, {grant_text!r}.encode(), 0)
print(p, flush=True)
time.sleep(300)
"""


def _spawn_holder(lease_dir: Path, grant: int) -> tuple[subprocess.Popen, str]:
    proc = subprocess.Popen(
        [sys.executable, "-c", _hold_lease_script(str(lease_dir), str(grant))],
        stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None
    path = proc.stdout.readline().strip()  # blocks until the holder has locked its lease
    assert path
    return proc, path


def test_grant_subtracts_live_sibling_grants(monkeypatch):
    monkeypatch.setenv("ALGUA_SWEEP_CPU_BUDGET", "8")
    lease_dir = Path(os.environ["ALGUA_DATA_DIR"]) / "sweep_leases"
    lease_dir.mkdir(parents=True, exist_ok=True)
    # Two live sibling sweeps holding grants of 5 and 2 => used=7 => we get 8-7=1.
    h1, _ = _spawn_holder(lease_dir, 5)
    h2, _ = _spawn_holder(lease_dir, 2)
    try:
        with cb.admit(overridden_count=100) as lease:
            assert lease.grant == 1  # max(1, 8-7)
    finally:
        for h in (h1, h2):
            h.send_signal(signal.SIGKILL)
            h.wait()


def test_worst_case_bound_budget_plus_k_minus_1(monkeypatch):
    # budget=8, a live sibling already granted the full 8 => the next admit floors at 1, so the
    # live total is 8+1=9 = budget + (K-1) with K=2. This is the documented worst case: NEVER
    # K×cpu_count, NEVER harmonic.
    monkeypatch.setenv("ALGUA_SWEEP_CPU_BUDGET", "8")
    lease_dir = Path(os.environ["ALGUA_DATA_DIR"]) / "sweep_leases"
    lease_dir.mkdir(parents=True, exist_ok=True)
    h1, _ = _spawn_holder(lease_dir, 8)
    try:
        with cb.admit(overridden_count=100) as lease:
            assert lease.grant == 1
    finally:
        h1.send_signal(signal.SIGKILL)
        h1.wait()


def test_orphaned_marker_is_reclaimed_not_counted(monkeypatch):
    # A holder that is SIGKILLed leaves its marker behind but the kernel frees the flock. The next
    # admit must reclaim (unlink) the orphan and NOT count its grant.
    monkeypatch.setenv("ALGUA_SWEEP_CPU_BUDGET", "8")
    lease_dir = Path(os.environ["ALGUA_DATA_DIR"]) / "sweep_leases"
    lease_dir.mkdir(parents=True, exist_ok=True)
    h1, path = _spawn_holder(lease_dir, 6)
    h1.send_signal(signal.SIGKILL)
    h1.wait()
    # Give the kernel a beat to release the flock of the dead process.
    for _ in range(50):
        if _flock_free(path):
            break
        time.sleep(0.05)
    with cb.admit(overridden_count=100) as lease:
        assert lease.grant == 8  # orphan's grant of 6 NOT subtracted
    assert not Path(path).exists()  # orphan reclaimed (unlinked)


def _flock_free(path: str) -> bool:
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def test_self_lease_never_probed_or_unlinked(monkeypatch):
    # Our own lease must survive the admission scan (self is counted implicitly, never probed).
    monkeypatch.setenv("ALGUA_SWEEP_CPU_BUDGET", "8")
    lease_dir = Path(os.environ["ALGUA_DATA_DIR"]) / "sweep_leases"
    with cb.admit(overridden_count=100):
        markers = list(lease_dir.glob("*.lease"))
        assert len(markers) == 1  # our own marker is still present mid-body


def test_garbled_sibling_grant_fails_closed(monkeypatch):
    # A live sibling whose lease has a garbled grant must be treated as the FULL budget (fail
    # closed toward LESS parallelism), so we never oversubscribe on an unreadable sibling.
    monkeypatch.setenv("ALGUA_SWEEP_CPU_BUDGET", "8")
    lease_dir = Path(os.environ["ALGUA_DATA_DIR"]) / "sweep_leases"
    lease_dir.mkdir(parents=True, exist_ok=True)
    # Holder writes "garbage" as its grant; _read_grant -> full budget (8) => we get max(1,8-8)=1.
    proc = subprocess.Popen(
        [sys.executable, "-c", _hold_lease_script(str(lease_dir), "garbage")],
        stdout=subprocess.PIPE, text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip()
    try:
        with cb.admit(overridden_count=100) as lease:
            assert lease.grant == 1
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait()


def test_non_lease_files_ignored(monkeypatch):
    monkeypatch.setenv("ALGUA_SWEEP_CPU_BUDGET", "8")
    lease_dir = Path(os.environ["ALGUA_DATA_DIR"]) / "sweep_leases"
    lease_dir.mkdir(parents=True, exist_ok=True)
    (lease_dir / "not-a-lease.txt").write_text("999")
    (lease_dir / ".admission.lock").write_text("")
    with cb.admit(overridden_count=100) as lease:
        assert lease.grant == 8  # foreign files ignored -> full budget


def test_close_lease_fd_in_worker_is_tolerant():
    # Closing a bogus/already-closed fd must never raise (would corrupt worker startup).
    cb.close_lease_fd_in_worker(999_999)  # EBADF swallowed


def _parent_sweep_script(data_dir: str) -> str:
    # A parent that admits a lease (grant>1) and builds a fork pool whose workers inherit the lease
    # fd, then blocks forever. When THIS parent is SIGKILLed the lease must be reclaimable — proving
    # the worker initializer closed the inherited fd so no child keeps the flock alive.
    return f"""
import os, time, multiprocessing
os.environ["ALGUA_DATA_DIR"] = {data_dir!r}
os.environ["ALGUA_SWEEP_CPU_BUDGET"] = "4"
from concurrent.futures import ProcessPoolExecutor
from algua.backtest import core_budget as cb

def _idle(_):
    time.sleep(300)
    return 0

ctx = multiprocessing.get_context("fork")
with cb.admit(overridden_count=4) as lease:
    with ProcessPoolExecutor(
        max_workers=lease.grant, mp_context=ctx,
        initializer=cb.close_lease_fd_in_worker, initargs=(lease.lease_fd,),
    ) as ex:
        fut = [ex.submit(_idle, i) for i in range(lease.grant)]
        print("READY", flush=True)
        time.sleep(300)
"""


def test_lease_reclaimed_after_parent_sigkill(monkeypatch, tmp_path):
    # The crash-liveness guarantee: a SIGKILLed parent whose fork-pool workers inherited the lease
    # fd must still leave the lease reclaimable (the initializer closed the fd in every worker).
    data_dir = tmp_path / "d"
    lease_dir = data_dir / "sweep_leases"
    env = {**os.environ, "ALGUA_DATA_DIR": str(data_dir), "ALGUA_SWEEP_CPU_BUDGET": "4"}
    proc = subprocess.Popen(
        [sys.executable, "-c", _parent_sweep_script(str(data_dir))],
        stdout=subprocess.PIPE, text=True, env=env,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "READY"  # pool up, workers spawned
    markers = list(lease_dir.glob("*.lease"))
    assert len(markers) == 1
    lease_path = markers[0]
    # Kill ONLY the parent (workers may briefly outlive it before the pool tears down).
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    # Within a bounded wait, the lease flock is free (no surviving worker holds the inherited fd).
    freed = False
    for _ in range(200):
        if _flock_free(str(lease_path)):
            freed = True
            break
        time.sleep(0.05)
    assert freed, "lease flock still held after parent SIGKILL -> a worker kept the inherited fd"


def test_published_lease_is_always_flock_held(monkeypatch):
    # Fail-open-race guard: any marker visible under the *.lease glob must ALREADY be flock-held
    # (published via lock-then-rename). A concurrent scanner must never see an unlocked lease and
    # mistake it for an orphan. Also: no stray .tmp remains mid-body.
    monkeypatch.setenv("ALGUA_SWEEP_CPU_BUDGET", "8")
    lease_dir = Path(os.environ["ALGUA_DATA_DIR"]) / "sweep_leases"
    with cb.admit(overridden_count=100):
        published = list(lease_dir.glob("*.lease"))
        assert len(published) == 1
        assert not _flock_free(str(published[0]))  # held by us
        assert list(lease_dir.glob("*.tmp")) == []  # rename completed, no residue
