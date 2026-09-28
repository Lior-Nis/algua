from __future__ import annotations

import concurrent.futures
import os
import tempfile
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
from tests._walk_faults import count_scandir_pulls, fail_scandir_once


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
        monkeypatch.setattr(artifact_store, "rename_noreplace", lambda *_: (_ for _ in ()).throw(
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


def _insert_destination_before_final_rename(
    monkeypatch: pytest.MonkeyPatch, target: Path, occupy,
) -> list[str]:
    """Occupy ``target`` after the existence check, immediately before the final rename.

    The racer is injected at whichever rename the store uses for final publication, then the
    real operation runs against the now-existing destination.
    """
    from algua.primitives import no_replace

    original_rename = os.rename
    real_noreplace = no_replace.rename_noreplace
    raced: list[str] = []

    def racing(real):
        def rename(source, destination, *args, **kwargs):
            if Path(destination) == target and not raced:
                raced.append(str(source))
                occupy(original_rename)
            return real(source, destination, *args, **kwargs)
        return rename

    monkeypatch.setattr(os, "rename", racing(original_rename))
    monkeypatch.setattr(
        artifact_store, "rename_noreplace", racing(real_noreplace), raising=False)
    return raced


def test_winner_inserted_after_existence_check_is_verified_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / _descriptor().locator
    winner_inodes: list[int] = []

    def publish_winner(original_rename) -> None:
        winner = Path(tempfile.mkdtemp(prefix="winner-", dir=target.parent))
        artifact_store._write_stage(winner, _files())
        original_rename(winner, target)
        winner_inodes.append(target.stat().st_ino)

    raced = _insert_destination_before_final_rename(monkeypatch, target, publish_winner)

    published = publish_bundle(tmp_path, _files(), _descriptor())

    assert raced, "the destination race was never injected"
    assert published == target
    verify_bundle(tmp_path, _descriptor())
    assert [target.stat().st_ino] == winner_inodes
    assert not Path(raced[0]).exists()
    assert not list(target.parent.glob(".stage-*"))


def test_empty_destination_inserted_after_existence_check_is_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / _descriptor().locator

    def occupy_empty(_original_rename) -> None:
        target.mkdir()

    raced = _insert_destination_before_final_rename(monkeypatch, target, occupy_empty)

    with pytest.raises(ArtifactStoreError):
        publish_bundle(tmp_path, _files(), _descriptor())

    assert raced, "the destination race was never injected"
    assert target.is_dir() and not any(target.iterdir())
    assert not Path(raced[0]).exists()
    assert not list(target.parent.glob(".stage-*"))


def test_publication_fails_closed_without_a_no_replace_primitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from algua.primitives import no_replace

    monkeypatch.setattr(no_replace, "_load_renameat2", lambda: None)
    original_rename = os.rename

    def no_final_replacement(source, destination, *args, **kwargs):
        if Path(destination) == tmp_path / _descriptor().locator:
            pytest.fail("final publication used a replacement-capable rename")
        return original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "rename", no_final_replacement)

    with pytest.raises(OSError):
        publish_bundle(tmp_path, _files(), _descriptor())

    target = tmp_path / _descriptor().locator
    assert not target.exists()
    assert not list(target.parent.glob(".stage-*"))


def test_verify_propagates_traversal_errors_instead_of_skipping_a_subtree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = publish_bundle(tmp_path, _files(), _descriptor())
    root.chmod(0o755)
    extra = root / "zz-extra"
    extra.mkdir()
    (extra / "smuggled.py").write_text("x = 1\n")
    (extra / "smuggled.py").chmod(0o444)
    extra.chmod(0o555)
    root.chmod(0o555)
    failed = fail_scandir_once(monkeypatch, lambda path: path == extra)

    with pytest.raises(ArtifactStoreError) as caught:
        verify_bundle(tmp_path, _descriptor())

    assert failed == [extra]
    assert isinstance(caught.value.__cause__, PermissionError)


def test_staging_seal_propagates_traversal_errors_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = fail_scandir_once(
        monkeypatch,
        lambda path: path.name == "algua" and path.parent.name.startswith(".stage-"),
    )

    with pytest.raises(PermissionError):
        publish_bundle(tmp_path, _files(), _descriptor())

    assert failed
    target = tmp_path / _descriptor().locator
    assert not target.exists()
    assert not list(target.parent.glob(".stage-*"))


def test_inventory_enforces_the_file_count_bound_before_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = publish_bundle(tmp_path, _files(), _descriptor())
    monkeypatch.setattr(artifact_store, "MAX_BUNDLE_FILES", len(_files()))
    verify_bundle(tmp_path, _descriptor())

    root.chmod(0o755)
    for name in ("zz-empty-1", "zz-empty-2"):
        (root / name).touch(mode=0o444)
    root.chmod(0o555)
    constructed: list[str] = []
    real_file = artifact_store.ArtifactFile

    def counted(*args, **kwargs):
        entry = real_file(*args, **kwargs)
        constructed.append(entry.path)
        return entry

    monkeypatch.setattr(artifact_store, "ArtifactFile", counted)

    with pytest.raises(ArtifactStoreError) as caught:
        verify_bundle(tmp_path, _descriptor())

    assert "file-count" in str(caught.value.__cause__)
    assert len(constructed) <= len(_files())


def _writable(root: Path):
    import contextlib

    @contextlib.contextmanager
    def opened():
        root.chmod(0o755)
        try:
            yield root
        finally:
            root.chmod(0o555)

    return opened()


def test_verify_bounds_empty_directory_fanout_before_retaining_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = publish_bundle(tmp_path, _files(), _descriptor())
    with _writable(root):
        fanout = root / "zz-fanout"
        fanout.mkdir()
        for index in range(300):
            (fanout / f"d{index:03d}").mkdir(mode=0o555)
        fanout.chmod(0o555)
    monkeypatch.setattr(artifact_store, "MAX_BUNDLE_DIRECTORIES", 5, raising=False)
    pulls = count_scandir_pulls(monkeypatch)

    with pytest.raises(ArtifactStoreError) as caught:
        verify_bundle(tmp_path, _descriptor())

    assert "directory-count" in str(caught.value.__cause__)
    assert pulls.get(fanout, 0) <= 6


def test_verify_bounds_one_directory_with_huge_file_fanout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = publish_bundle(tmp_path, _files(), _descriptor())
    with _writable(root):
        for index in range(300):
            (root / f"zz-{index:03d}").touch(mode=0o444)
    monkeypatch.setattr(artifact_store, "MAX_BUNDLE_FILES", len(_files()))
    pulls = count_scandir_pulls(monkeypatch)

    with pytest.raises(ArtifactStoreError) as caught:
        verify_bundle(tmp_path, _descriptor())

    assert "file-count" in str(caught.value.__cause__)
    assert pulls[root] <= len(_files()) + 3


def test_protected_bundle_directory_bound_is_explicit() -> None:
    from algua.registry.artifact_contract import MAX_BUNDLE_DIRECTORIES

    assert MAX_BUNDLE_DIRECTORIES == 10_000


def test_verify_rejects_a_writable_bundle_directory(tmp_path: Path) -> None:
    root = publish_bundle(tmp_path, _files(), _descriptor())
    (root / "algua").chmod(0o755)

    with pytest.raises(ArtifactStoreError):
        verify_bundle(tmp_path, _descriptor())


def _foreign_stage(parent: Path) -> Path:
    """Another builder's in-progress stage; this attempt must never remove it."""
    parent.mkdir(parents=True, exist_ok=True)
    foreign = parent / ".stage-foreign"
    foreign.mkdir(mode=0o700)
    (foreign / "partial").write_text("another builder")
    return foreign


def test_lock_acquisition_fault_publishes_nothing_and_cleans_only_its_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contextlib
    import errno

    target = tmp_path / _descriptor().locator
    foreign = _foreign_stage(target.parent)
    attempted: list[Path] = []

    @contextlib.contextmanager
    def failing_lock(path: Path, **_kwargs):
        attempted.append(path)
        raise OSError(errno.ENOLCK, "injected lock acquisition fault")
        yield

    monkeypatch.setattr(artifact_store, "file_lock", failing_lock)

    with pytest.raises(OSError) as caught:
        publish_bundle(tmp_path, _files(), _descriptor())

    assert caught.value.errno == errno.ENOLCK
    assert attempted == [tmp_path / "frozen/.locks/bundles" / f"{_descriptor().digest}.lock"]
    assert not target.exists() and not target.is_symlink()
    assert sorted(path.name for path in target.parent.iterdir()) == [".stage-foreign"]
    assert (foreign / "partial").read_text() == "another builder"


def test_post_publication_verification_fault_leaves_a_valid_immutable_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / _descriptor().locator
    foreign = _foreign_stage(target.parent)
    real_verify = artifact_store.verify_bundle

    def failing_final_verify(*_args, **_kwargs):
        assert target.is_dir(), "final verification must follow publication"
        raise ArtifactStoreError("injected post-publication verification fault")

    monkeypatch.setattr(artifact_store, "verify_bundle", failing_final_verify)

    with pytest.raises(ArtifactStoreError, match="post-publication"):
        publish_bundle(tmp_path, _files(), _descriptor())

    assert real_verify(tmp_path, _descriptor()) == target
    assert target.stat().st_mode & 0o777 == 0o555
    assert sorted(path.name for path in target.parent.iterdir()) == [
        ".stage-foreign", target.name]
    assert (foreign / "partial").read_text() == "another builder"
