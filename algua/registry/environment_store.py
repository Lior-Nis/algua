"""Atomic publication of verified relocatable planner environments."""
from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path

from algua.primitives.atomic_io import fsync_dir, fsync_file, fsync_parents
from algua.primitives.flock import file_lock
from algua.registry.artifact_store import resolve_locator
from algua.registry.environment_contract import EnvironmentDescriptor
from algua.registry.planner_environment import (
    EnvironmentIncompatible,
    inventory_environment,
    verify_environment,
)


class EnvironmentStoreError(ValueError):
    """A published environment is missing, corrupt, or unsafe."""


def _cleanup(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    for dirpath, dirnames, _files in os.walk(path, topdown=False, followlinks=False):
        for name in dirnames:
            child = Path(dirpath) / name
            if not child.is_symlink():
                child.chmod(0o700)
        Path(dirpath).chmod(0o700)
    shutil.rmtree(path)


def _seal_and_sync(root: Path) -> None:
    directories: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        directory = Path(dirpath)
        directories.append(directory)
        for name in dirnames:
            child = directory / name
            if child.is_symlink():
                raise EnvironmentStoreError("environment contains a directory symlink")
        for name in filenames:
            path = directory / name
            if path.is_symlink():
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise EnvironmentStoreError("environment contains unsafe file content")
            path.chmod(0o555 if info.st_mode & stat.S_IXUSR else 0o444)
            fsync_file(path)
        directory.chmod(0o555)
        fsync_dir(directory)


def _assert_sealed(root: Path) -> None:
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        directory = Path(dirpath)
        if directory.is_symlink() or stat.S_IMODE(directory.stat().st_mode) != 0o555:
            raise EnvironmentStoreError("environment directory permissions drifted")
        for name in filenames:
            path = directory / name
            if path.is_symlink():
                continue
            if stat.S_IMODE(path.stat().st_mode) not in {0o444, 0o555}:
                raise EnvironmentStoreError("environment file permissions drifted")


def verify_published_environment(
    store_root: Path, descriptor: EnvironmentDescriptor,
) -> Path:
    target = resolve_locator(
        store_root, descriptor.locator, expected_digest=descriptor.digest, kind="environments")
    try:
        _assert_sealed(target)
        verify_environment(target, descriptor.interpreter, descriptor.inventory_digest)
    except (OSError, EnvironmentIncompatible, ValueError) as exc:
        raise EnvironmentStoreError("frozen environment is missing or corrupt") from exc
    return target


def publish_environment(
    store_root: Path, built_environment: Path, descriptor: EnvironmentDescriptor,
) -> Path:
    try:
        if inventory_environment(built_environment).digest != descriptor.inventory_digest:
            raise EnvironmentStoreError("built environment inventory disagrees")
    except EnvironmentIncompatible as exc:
        raise EnvironmentStoreError("built environment is unsafe") from exc
    root = store_root.resolve()
    target = resolve_locator(
        root, descriptor.locator, expected_digest=descriptor.digest, kind="environments")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    lock_parent = root / "frozen/.locks/environments"
    lock_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    hidden = Path(tempfile.mkdtemp(prefix=".reserve-", dir=target.parent))
    hidden.rmdir()
    os.rename(built_environment, hidden)
    published = False
    try:
        _seal_and_sync(hidden)
        verify_environment(hidden, descriptor.interpreter, descriptor.inventory_digest)
        with file_lock(lock_parent / f"{descriptor.digest}.lock"):
            if target.exists() or target.is_symlink():
                verify_published_environment(root, descriptor)
            else:
                os.rename(hidden, target)
                published = True
                fsync_parents(target, stop_at=root)
        return verify_published_environment(root, descriptor)
    finally:
        if not published:
            _cleanup(hidden)
