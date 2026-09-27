from __future__ import annotations

import concurrent.futures
import os
import shutil
import sys
import venv
from pathlib import Path

import pytest

from algua.registry.environment_contract import EnvironmentDescriptor, EnvironmentKey
from algua.registry.environment_store import (
    EnvironmentStoreError,
    publish_environment,
    verify_published_environment,
)
from algua.registry.planner_environment import current_interpreter_identity, inventory_environment


def _stage(root: Path, name: str = "build-environment") -> Path:
    stage = root / name
    venv.EnvBuilder(with_pip=False, symlinks=True).create(stage)
    lib64 = stage / "lib64"
    if lib64.is_symlink():
        lib64.unlink()
    return stage


def _descriptor(stage: Path) -> EnvironmentDescriptor:
    identity = current_interpreter_identity()
    key = EnvironmentKey(
        build_inputs_digest="a" * 64, dependency_hash="b" * 64, interpreter=identity,
        uv_version="uv 0.9.26", create_argv=("uv", "venv"), sync_argv=("uv", "sync"),
    )
    return EnvironmentDescriptor(key, inventory_environment(stage).digest, identity)


def test_publish_verify_reuse_and_final_interpreter(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    descriptor = _descriptor(stage)
    published = publish_environment(tmp_path / "store", stage, descriptor)

    assert published == tmp_path / "store" / descriptor.locator
    assert publish_environment(tmp_path / "store", _stage(tmp_path), descriptor) == published
    verify_published_environment(tmp_path / "store", descriptor)
    assert published.stat().st_mode & 0o777 == 0o555
    assert os.access(published / "bin/python", os.X_OK)


def test_corrupt_environment_is_not_repaired(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    descriptor = _descriptor(stage)
    published = publish_environment(tmp_path / "store", stage, descriptor)
    config = published / "pyvenv.cfg"
    config.chmod(0o644)
    config.write_text("corrupt")

    with pytest.raises(EnvironmentStoreError):
        publish_environment(tmp_path / "store", _stage(tmp_path), descriptor)
    assert config.read_text() == "corrupt"


def test_inventory_drift_and_unexpected_link_fail_closed(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    descriptor = _descriptor(stage)
    published = publish_environment(tmp_path / "store", stage, descriptor)
    published.chmod(0o755)
    os.symlink(sys.executable, published / "unexpected")
    with pytest.raises(EnvironmentStoreError):
        verify_published_environment(tmp_path / "store", descriptor)


def test_concurrent_equivalent_environments_publish_one_object(tmp_path: Path) -> None:
    first = _stage(tmp_path, "first")
    second = tmp_path / "second"
    shutil.copytree(first, second, symlinks=True)
    descriptor = _descriptor(first)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(publish_environment, tmp_path / "store", first, descriptor),
            pool.submit(publish_environment, tmp_path / "store", second, descriptor),
        ]
    assert futures[0].result() == futures[1].result()
    verify_published_environment(tmp_path / "store", descriptor)


@pytest.mark.parametrize(
    "boundary", ["seal", "file_fsync", "dir_fsync", "verify", "rename", "parent_fsync"],
)
def test_environment_publication_faults_leave_no_partial_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    from algua.registry import environment_store
    from algua.registry.planner_environment import EnvironmentIncompatible

    stage = _stage(tmp_path)
    descriptor = _descriptor(stage)
    if boundary == "seal":
        monkeypatch.setattr(
            environment_store, "_seal_and_sync",
            lambda *_args: (_ for _ in ()).throw(OSError("seal fault")),
        )
    elif boundary == "file_fsync":
        monkeypatch.setattr(
            environment_store, "fsync_file",
            lambda *_args: (_ for _ in ()).throw(OSError("file fsync fault")),
        )
    elif boundary == "dir_fsync":
        monkeypatch.setattr(
            environment_store, "fsync_dir",
            lambda *_args: (_ for _ in ()).throw(OSError("dir fsync fault")),
        )
    elif boundary == "verify":
        monkeypatch.setattr(
            environment_store, "verify_environment",
            lambda *_args: (_ for _ in ()).throw(EnvironmentIncompatible("verify fault")),
        )
    elif boundary == "rename":
        original = environment_store.os.rename
        calls = 0

        def fail_final(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("rename fault")
            return original(source, target)

        monkeypatch.setattr(environment_store.os, "rename", fail_final)
    else:
        monkeypatch.setattr(
            environment_store, "fsync_parents",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("parent fsync fault")),
        )

    with pytest.raises((OSError, EnvironmentStoreError, EnvironmentIncompatible)):
        publish_environment(tmp_path / "store", stage, descriptor)
    target = tmp_path / "store" / descriptor.locator
    if target.exists():
        verify_published_environment(tmp_path / "store", descriptor)
    assert not list(target.parent.glob(".reserve-*"))
