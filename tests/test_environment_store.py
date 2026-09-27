from __future__ import annotations

import os
import sys
import venv
from pathlib import Path

import pytest

from algua.registry.artifact_contract import EnvironmentDescriptor, EnvironmentKey
from algua.registry.environment_store import (
    EnvironmentStoreError,
    publish_environment,
    verify_published_environment,
)
from algua.registry.planner_environment import current_interpreter_identity, inventory_environment


def _stage(root: Path) -> Path:
    stage = root / "build-environment"
    venv.EnvBuilder(with_pip=False).create(stage)
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
