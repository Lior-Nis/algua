from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path

import pytest

from algua.registry import artifact_store
from algua.registry.artifact_contract import BundleDescriptor
from algua.registry.artifact_store import (
    ArtifactStoreError,
    publish_bundle,
    resolve_locator,
    verify_bundle,
)
from algua.registry.frozen_source import FrozenFile


def _files() -> tuple[FrozenFile, ...]:
    return (
        FrozenFile("_algua/protocol.json", "100644", b"{}"),
        FrozenFile("algua/__init__.py", "100644", b"VALUE = 1\n"),
        FrozenFile("algua/tool.py", "100755", b"#!/usr/bin/env python\n"),
    )


def _descriptor() -> BundleDescriptor:
    return BundleDescriptor.from_files(tuple(item.contract_entry for item in _files()))


def test_publish_verify_and_reuse_bundle(tmp_path: Path) -> None:
    first = publish_bundle(tmp_path, _files(), _descriptor())
    second = publish_bundle(tmp_path, _files(), _descriptor())

    assert first == second == tmp_path / _descriptor().locator
    verify_bundle(tmp_path, _descriptor())
    assert (first / "algua/__init__.py").stat().st_mode & 0o777 == 0o444
    assert (first / "algua/tool.py").stat().st_mode & 0o777 == 0o555
    assert first.stat().st_mode & 0o777 == 0o555


def test_concurrent_same_digest_builders_publish_one_object(tmp_path: Path) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(
            lambda _: publish_bundle(tmp_path, _files(), _descriptor()), range(8)))
    assert len(set(paths)) == 1
    verify_bundle(tmp_path, _descriptor())


def test_concurrent_different_digests_publish_separate_objects(tmp_path: Path) -> None:
    other_files = tuple(sorted(
        _files() + (FrozenFile("algua/other.py", "100644", b"x = 2\n"),),
        key=lambda item: item.path.encode(),
    ))
    other = BundleDescriptor.from_files(tuple(item.contract_entry for item in other_files))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = (
            pool.submit(publish_bundle, tmp_path, _files(), _descriptor()),
            pool.submit(publish_bundle, tmp_path, other_files, other),
        )
    assert futures[0].result() != futures[1].result()
    verify_bundle(tmp_path, _descriptor())
    verify_bundle(tmp_path, other)


@pytest.mark.parametrize(
    "boundary", ["write", "file_fsync", "seal", "tree_fsync", "rename", "parent_fsync"],
)
def test_faults_never_leave_partial_published_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    if boundary == "write":
        original_open = artifact_store.os.open

        def fail_write(path, flags, *args, **kwargs):
            if flags & artifact_store.os.O_WRONLY:
                raise OSError("injected write")
            return original_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(
            artifact_store.os, "open", fail_write,
        )
    elif boundary == "file_fsync":
        monkeypatch.setattr(
            artifact_store.os, "fsync",
            lambda *_args: (_ for _ in ()).throw(OSError("injected file fsync")),
        )
    elif boundary == "seal":
        original_chmod = Path.chmod

        def fail_seal(path, mode, *args, **kwargs):
            if mode in {0o444, 0o555}:
                raise OSError("injected seal")
            return original_chmod(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "chmod", fail_seal)
    elif boundary == "tree_fsync":
        monkeypatch.setattr(
            artifact_store, "fsync_tree",
            lambda *_args: (_ for _ in ()).throw(OSError("injected tree fsync")),
        )
    elif boundary == "rename":
        monkeypatch.setattr(artifact_store.os, "rename", lambda *_: (_ for _ in ()).throw(
            OSError("injected rename")))
    else:
        monkeypatch.setattr(artifact_store, "fsync_parents", lambda *_args, **_kwargs: (
            (_ for _ in ()).throw(OSError("injected parent fsync"))))

    with pytest.raises(OSError):
        publish_bundle(tmp_path, _files(), _descriptor())
    target = tmp_path / _descriptor().locator
    if target.exists():
        verify_bundle(tmp_path, _descriptor())
    assert not list(target.parent.glob(".stage-*"))


def test_existing_corrupt_or_writable_object_fails_without_repair(tmp_path: Path) -> None:
    root = publish_bundle(tmp_path, _files(), _descriptor())
    target = root / "algua/__init__.py"
    target.chmod(0o644)
    target.write_bytes(b"corrupt\n")

    with pytest.raises(ArtifactStoreError):
        publish_bundle(tmp_path, _files(), _descriptor())
    assert target.read_bytes() == b"corrupt\n"


def test_verify_rejects_extra_file_symlink_and_hardlink(tmp_path: Path) -> None:
    root = publish_bundle(tmp_path, _files(), _descriptor())
    root.chmod(0o755)
    (root / "extra").write_text("x")
    with pytest.raises(ArtifactStoreError):
        verify_bundle(tmp_path, _descriptor())
    (root / "extra").unlink()
    os.link(root / "algua/__init__.py", root / "hardlink")
    with pytest.raises(ArtifactStoreError):
        verify_bundle(tmp_path, _descriptor())


@pytest.mark.parametrize("locator", ["/tmp/x", "../x", "frozen/bundles/sha256/aa/wrong"])
def test_locator_resolution_rejects_untrusted_paths(tmp_path: Path, locator: str) -> None:
    with pytest.raises(ArtifactStoreError):
        resolve_locator(tmp_path, locator, expected_digest="a" * 64, kind="bundles")


def test_locator_resolution_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "frozen").symlink_to(outside, target_is_directory=True)
    locator = "frozen/bundles/sha256/aa/" + "a" * 64
    with pytest.raises(ArtifactStoreError, match="ancestor"):
        resolve_locator(tmp_path, locator, expected_digest="a" * 64, kind="bundles")


def test_verify_rejects_empty_directory_and_oversized_file(tmp_path: Path, monkeypatch) -> None:
    root = publish_bundle(tmp_path, _files(), _descriptor())
    root.chmod(0o755)
    empty = root / "empty"
    empty.mkdir(mode=0o555)
    root.chmod(0o555)
    with pytest.raises(ArtifactStoreError):
        verify_bundle(tmp_path, _descriptor())
    root.chmod(0o755)
    empty.rmdir()
    root.chmod(0o555)

    monkeypatch.setattr(artifact_store, "MAX_FILE_BYTES", 1)
    with pytest.raises(ArtifactStoreError):
        verify_bundle(tmp_path, _descriptor())
