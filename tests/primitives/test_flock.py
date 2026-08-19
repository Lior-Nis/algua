"""algua.primitives.flock — the ONE cross-process flock primitive (spec §4.1)."""
from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest

from algua.primitives.flock import (
    LockHeld,
    LockReplaced,
    acquire,
    file_lock,
    probe_held,
    read_holder,
    release,
)


def _hold_lock(path: str, acquired_evt, release_evt) -> None:
    fd = acquire(Path(path))
    acquired_evt.set()
    release_evt.wait(timeout=30)
    release(fd)


@pytest.fixture()
def holder(tmp_path):
    """A live child process holding LOCK_EX on tmp_path/'x.lock'."""
    lock = tmp_path / "x.lock"
    acquired = multiprocessing.Event()
    released = multiprocessing.Event()
    proc = multiprocessing.Process(target=_hold_lock, args=(str(lock), acquired, released))
    proc.start()
    assert acquired.wait(timeout=10)
    yield lock
    released.set()
    proc.join(timeout=10)


def test_acquire_release_roundtrip(tmp_path):
    lock = tmp_path / "a.lock"
    fd = acquire(lock)
    release(fd)
    fd2 = acquire(lock, blocking=False)  # re-acquirable after release
    release(fd2)


def test_nonblocking_contention_raises_lockheld(holder):
    with pytest.raises(LockHeld):
        acquire(holder, blocking=False)


def test_lockheld_carries_holder_metadata(tmp_path):
    lock = tmp_path / "m.lock"
    with file_lock(lock, metadata={"pid": 123, "job": "sweep"}):
        assert read_holder(lock) == {"pid": 123, "job": "sweep"}
        with pytest.raises(LockHeld) as exc_info:
            with file_lock(lock, blocking=False):
                pass
        assert exc_info.value.holder == {"pid": 123, "job": "sweep"}
    # body truncated on release
    assert read_holder(lock) is None


def test_probe_held_fail_closed(tmp_path, holder):
    assert probe_held(holder) is True          # live holder -> held
    assert probe_held(tmp_path / "no.lock") is False  # absent marker -> not held
    free = tmp_path / "free.lock"
    free.touch()
    assert probe_held(free) is False           # present but unlocked -> not held


def test_verify_inode_detects_replacement(tmp_path, monkeypatch):
    lock = tmp_path / "v.lock"

    real_stat = os.stat

    def replaced_stat(p, *a, **kw):
        if Path(p) == lock:
            lock.unlink(missing_ok=True)
            lock.write_text("")  # new inode every check
        return real_stat(p, *a, **kw)

    monkeypatch.setattr(os, "stat", replaced_stat)
    with pytest.raises(LockReplaced):
        acquire(lock, verify_inode=True, retries=3)


def test_file_lock_on_oserror_proceed(tmp_path, monkeypatch):
    lock = tmp_path / "e.lock"

    def boom(*a, **kw):
        raise OSError("ENOLCK")

    monkeypatch.setattr("algua.primitives.flock.acquire", boom)
    entered = False
    with file_lock(lock, on_oserror="proceed"):
        entered = True
    assert entered  # degraded to unlocked, did not raise


def test_read_holder_garbled_body_is_none(tmp_path):
    lock = tmp_path / "g.lock"
    lock.write_text("not json")
    assert read_holder(lock) is None
