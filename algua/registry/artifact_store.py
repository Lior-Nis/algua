"""Atomic no-overwrite publication for content-addressed planner bundles."""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from pathlib import Path

from algua.primitives.atomic_io import fsync_parents, fsync_tree
from algua.primitives.flock import file_lock
from algua.registry.artifact_contract import ArtifactFile, BundleDescriptor
from algua.registry.frozen_source import FrozenFile


class ArtifactStoreError(ValueError):
    """Published immutable content is unsafe, corrupt, or inconsistent."""


def resolve_locator(
    store_root: Path, locator: str, *, expected_digest: str, kind: str,
) -> Path:
    expected = f"frozen/{kind}/sha256/{expected_digest[:2]}/{expected_digest}"
    if locator != expected or Path(locator).is_absolute() or ".." in Path(locator).parts:
        raise ArtifactStoreError("artifact locator is not canonical for its digest")
    root = store_root.resolve()
    candidate = root.joinpath(*locator.split("/"))
    if root not in candidate.parents:
        raise ArtifactStoreError("artifact locator escapes the trusted store root")
    return candidate


def _remove_owned_stage(stage: Path) -> None:
    if not stage.exists() and not stage.is_symlink():
        return
    for dirpath, dirnames, _filenames in os.walk(stage, topdown=False, followlinks=False):
        for name in dirnames:
            path = Path(dirpath) / name
            if not path.is_symlink():
                path.chmod(0o700)
        Path(dirpath).chmod(0o700)
    shutil.rmtree(stage)


def _write_stage(stage: Path, files: tuple[FrozenFile, ...]) -> None:
    for item in files:
        destination = stage.joinpath(*item.path.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=False) as handle:
                handle.write(item.data)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(fd)
        destination.chmod(0o555 if item.mode == "100755" else 0o444)
    directories = [Path(path) for path, _dirs, _files in os.walk(stage, topdown=False)]
    for directory in directories:
        directory.chmod(0o555)
    fsync_tree(stage)


def _inventory(root: Path) -> tuple[ArtifactFile, ...]:
    if root.is_symlink() or not root.is_dir():
        raise ArtifactStoreError("bundle root is not a real directory")
    if stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise ArtifactStoreError("bundle root permissions drifted")
    entries: list[ArtifactFile] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        directory = Path(dirpath)
        if directory.is_symlink() or stat.S_IMODE(directory.stat().st_mode) != 0o555:
            raise ArtifactStoreError("bundle directory is linked or writable")
        for dirname in dirnames:
            child = directory / dirname
            if child.is_symlink():
                raise ArtifactStoreError("bundle contains a directory symlink")
        for filename in filenames:
            path = directory / filename
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ArtifactStoreError("bundle contains a link or non-regular file")
            permissions = stat.S_IMODE(info.st_mode)
            if permissions not in {0o444, 0o555}:
                raise ArtifactStoreError("bundle file permissions drifted")
            data = path.read_bytes()
            entries.append(ArtifactFile(
                path=path.relative_to(root).as_posix(),
                mode="100755" if permissions == 0o555 else "100644",
                size=len(data), sha256=hashlib.sha256(data).hexdigest(),
            ))
    return tuple(sorted(entries, key=lambda item: item.path.encode()))


def verify_bundle(store_root: Path, descriptor: BundleDescriptor) -> Path:
    target = resolve_locator(
        store_root, descriptor.locator, expected_digest=descriptor.digest, kind="bundles")
    try:
        actual = BundleDescriptor.from_files(_inventory(target))
    except (OSError, ValueError) as exc:
        raise ArtifactStoreError("frozen bundle is missing or corrupt") from exc
    if actual != descriptor:
        raise ArtifactStoreError("frozen bundle inventory disagrees with its descriptor")
    return target


def publish_bundle(
    store_root: Path, files: tuple[FrozenFile, ...], descriptor: BundleDescriptor,
) -> Path:
    expected = BundleDescriptor.from_files(tuple(item.contract_entry for item in files))
    if expected != descriptor:
        raise ArtifactStoreError("bundle files disagree with the expected descriptor")
    root = store_root.resolve()
    lock_parent = root / "frozen/.locks/bundles"
    target = resolve_locator(root, descriptor.locator, expected_digest=descriptor.digest,
                             kind="bundles")
    lock_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=target.parent))
    stage.chmod(0o700)
    published = False
    try:
        _write_stage(stage, files)
        staged = BundleDescriptor.from_files(_inventory(stage))
        if staged != descriptor:
            raise ArtifactStoreError("staged bundle verification failed")
        lock_path = lock_parent / f"{descriptor.digest}.lock"
        with file_lock(lock_path):
            if target.exists() or target.is_symlink():
                verify_bundle(root, descriptor)
            else:
                try:
                    os.rename(stage, target)
                except FileExistsError:
                    verify_bundle(root, descriptor)
                else:
                    published = True
                    fsync_parents(target, stop_at=root)
        return verify_bundle(root, descriptor)
    finally:
        if not published:
            _remove_owned_stage(stage)
