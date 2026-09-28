"""Atomic no-overwrite publication for content-addressed planner bundles."""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

from algua.primitives.atomic_io import fsync_parents, fsync_tree
from algua.primitives.bounded_walk import (
    TraversalLimitExceeded,
    TreeEntry,
    WalkCleanupError,
    scoped_walk,
)
from algua.primitives.flock import file_lock
from algua.primitives.no_replace import rename_noreplace
from algua.registry.artifact_contract import (
    MAX_BUNDLE_DIRECTORIES,
    MAX_BUNDLE_FILES,
    MAX_PATH_BYTES,
    ArtifactFile,
    BundleDescriptor,
)
from algua.registry.frozen_source import MAX_BUNDLE_BYTES, MAX_FILE_BYTES, FrozenFile


class ArtifactStoreError(ValueError):
    """Published immutable content is unsafe, corrupt, or inconsistent."""


@contextmanager
def _tree(root: Path) -> Iterator[Generator[TreeEntry, None, None]]:
    """Walk a bundle tree; a listing that cannot be closed is a typed store error."""
    try:
        with scoped_walk(
            root, max_files=MAX_BUNDLE_FILES, max_directories=MAX_BUNDLE_DIRECTORIES,
            max_path_bytes=MAX_PATH_BYTES,
        ) as tree:
            yield tree
    except WalkCleanupError as exc:
        raise ArtifactStoreError("a bundle directory listing could not be closed") from exc


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
    current = root
    for component in locator.split("/")[:-1]:
        current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ArtifactStoreError("artifact locator has an unsafe ancestor")
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
    with _tree(stage) as tree:
        directories = [entry.path for entry in tree if entry.is_dir]
    for directory in (*reversed(directories), stage):
        directory.chmod(0o555)
    fsync_tree(stage)


def _inventory(root: Path) -> tuple[ArtifactFile, ...]:
    if root.is_symlink() or not root.is_dir():
        raise ArtifactStoreError("bundle root is not a real directory")
    if stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise ArtifactStoreError("bundle root permissions drifted")
    entries: list[ArtifactFile] = []
    directories: set[str] = set()
    total = 0
    try:
        with _tree(root) as tree:
            for entry in tree:
                info = entry.path.lstat()
                if entry.is_dir:
                    if stat.S_IMODE(info.st_mode) != 0o555:
                        raise ArtifactStoreError("bundle directory is linked or writable")
                    directories.add(entry.relative)
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ArtifactStoreError("bundle contains a link or non-regular file")
                permissions = stat.S_IMODE(info.st_mode)
                if permissions not in {0o444, 0o555}:
                    raise ArtifactStoreError("bundle file permissions drifted")
                if info.st_size > MAX_FILE_BYTES:
                    raise ArtifactStoreError("bundle file exceeds the per-file bound")
                digest = hashlib.sha256()
                size = 0
                with entry.path.open("rb") as handle:
                    # Never request more than one byte past the bound, even if the file grew.
                    while chunk := handle.read(min(1024 * 1024, MAX_FILE_BYTES - size + 1)):
                        size += len(chunk)
                        if size > MAX_FILE_BYTES:
                            raise ArtifactStoreError("bundle file exceeds the per-file bound")
                        digest.update(chunk)
                total += size
                if total > MAX_BUNDLE_BYTES:
                    raise ArtifactStoreError("bundle exceeds the aggregate size bound")
                entries.append(ArtifactFile(
                    path=entry.relative,
                    mode="100755" if permissions == 0o555 else "100644",
                    size=size, sha256=digest.hexdigest(),
                ))
    except TraversalLimitExceeded as exc:
        raise ArtifactStoreError(f"bundle exceeds the {exc.kind} bound") from exc
    implied = {
        parent.as_posix()
        for entry in entries
        for parent in Path(entry.path).parents
        if parent.as_posix() != "."
    }
    if directories != implied:
        raise ArtifactStoreError("bundle contains an undeclared or empty directory")
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
                    rename_noreplace(stage, target)
                except FileExistsError:
                    verify_bundle(root, descriptor)
                else:
                    published = True
                    fsync_parents(target, stop_at=root)
        return verify_bundle(root, descriptor)
    finally:
        if not published:
            _remove_owned_stage(stage)
