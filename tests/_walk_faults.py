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
