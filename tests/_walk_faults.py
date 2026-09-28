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
