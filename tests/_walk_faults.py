"""Deterministic directory-traversal fault injection for fail-closed walk tests.

Permission bits are not a reliable way to make a directory unreadable (a privileged CI runner
ignores them), so the fault is injected at `os.scandir`, the call `os.walk` makes to list each
directory. File-descriptor calls (such as `shutil.rmtree`'s) pass through untouched.
"""
from __future__ import annotations

import errno
import os
from collections.abc import Callable
from pathlib import Path

import pytest


def fail_scandir_once(
    monkeypatch: pytest.MonkeyPatch, matches: Callable[[Path], bool],
) -> list[Path]:
    """Make the first listing of a matching directory fail; return the directories failed."""
    original = os.scandir
    failed: list[Path] = []

    def scandir(path=".", *args, **kwargs):
        if not isinstance(path, int) and not failed:
            candidate = Path(os.fsdecode(path))
            if matches(candidate):
                failed.append(candidate)
                raise PermissionError(errno.EACCES, "injected traversal fault", str(candidate))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", scandir)
    return failed


class _CountingIterator:
    """A scandir iterator proxy that records how many entries were pulled from one directory."""

    def __init__(self, inner, pulls: dict[Path, int], path: Path) -> None:
        self._inner = inner
        self._pulls = pulls
        self._path = path

    def __iter__(self):
        return self

    def __next__(self):
        entry = next(self._inner)
        self._pulls[self._path] = self._pulls.get(self._path, 0) + 1
        return entry

    def close(self) -> None:
        self._inner.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def count_scandir_pulls(monkeypatch: pytest.MonkeyPatch) -> dict[Path, int]:
    """Record, per listed directory, how many entries a traversal pulled from `os.scandir`."""
    original = os.scandir
    pulls: dict[Path, int] = {}

    def scandir(path=".", *args, **kwargs):
        inner = original(path, *args, **kwargs)
        if isinstance(path, int):
            return inner
        return _CountingIterator(inner, pulls, Path(os.fsdecode(path)))

    monkeypatch.setattr(os, "scandir", scandir)
    return pulls


class TrackedListing:
    """A scandir proxy that records whether its real listing was released.

    A faulty proxy raises its fault from `close()`; by default it releases the listing first, and
    with `release=False` it raises before releasing. `times` bounds how many closes fail.
    """

    def __init__(self, inner, path: Path, fault, closed: dict[Path, bool], *,
                 release: bool = True, times: int | None = None) -> None:
        self._inner = inner
        self._path = path
        self._fault = fault
        self._release = release
        self._times = times
        self._closed = closed
        closed[path] = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._inner)

    def close(self) -> None:
        failing = self._fault is not None and (self._times is None or self._times > 0)
        if failing and self._times is not None:
            self._times -= 1
        if self._release or not failing:
            self._inner.close()
            self._closed[self._path] = True
        if failing:
            if isinstance(self._fault, int):
                raise OSError(self._fault, "injected close fault")
            raise self._fault("injected close fault")


def track_closes(
    monkeypatch: pytest.MonkeyPatch, faulty: Path, *, also: dict[Path, object] | None = None,
    fault: object = errno.EIO, release: bool = True, times: int | None = None,
) -> dict[Path, bool]:
    original = os.scandir
    closed: dict[Path, bool] = {}
    faults = {faulty: fault, **(also or {})}

    def scandir(path):
        target = Path(os.fsdecode(path))
        return TrackedListing(original(path), target, faults.get(target), closed,
                        release=release, times=times)

    monkeypatch.setattr(os, "scandir", scandir)
    return closed
