from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from algua.registry.frozen_source import (
    FrozenAssetsUnsupported,
    FrozenSourceError,
    assert_clean_head,
    export_build_inputs,
    export_source,
    parse_tree,
    require_source_only,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "algua").mkdir()
    (tmp_path / "algua/__init__.py").write_text("VALUE = 'committed'\n")
    tool = tmp_path / "algua/tool.py"
    tool.write_text("#!/usr/bin/env python\n")
    tool.chmod(0o755)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='0'\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / ".python-version").write_text("3.12\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "fixture")
    return tmp_path


def test_export_reads_exact_commit_blobs_not_checkout(repo: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    (repo / "algua/__init__.py").write_text("VALUE = 'mutable'\n")

    source = export_source(repo, head)
    by_path = {item.path: item for item in source}

    assert by_path["algua/__init__.py"].data == b"VALUE = 'committed'\n"
    assert by_path["algua/__init__.py"].mode == "100644"
    assert by_path["algua/tool.py"].mode == "100755"


def test_build_inputs_come_from_same_commit(repo: Path) -> None:
    head = assert_clean_head(repo)
    inputs = export_build_inputs(repo, head)
    assert [item.path for item in inputs] == [".python-version", "pyproject.toml", "uv.lock"]
    assert inputs[0].data == b"3.12\n"


@pytest.mark.parametrize(
    "entry",
    [
        b"120000 blob " + b"a" * 40 + b"\talgua/link\0",
        b"160000 commit " + b"a" * 40 + b"\talgua/submodule\0",
        b"100644 blob " + b"a" * 40 + b"\talgua/../escape.py\0",
        b"100644 blob " + b"a" * 40 + b"\talgua\\evil.py\0",
        b"100644 blob " + b"a" * 40 + b"\talgua/bad. \0",
        b"100644 blob " + b"a" * 40 + b"\talgua/\xff.py\0",
    ],
)
def test_tree_parser_rejects_unsafe_entries(entry: bytes) -> None:
    with pytest.raises(FrozenSourceError):
        parse_tree(entry)


def test_tree_parser_rejects_casefold_and_normalization_collisions() -> None:
    oid = b"a" * 40
    raw = (
        b"100644 blob " + oid + b"\talgua/Caf\xc3\xa9.py\0"
        b"100644 blob " + oid + b"\talgua/caf\xc3\xa9.py\0"
    )
    with pytest.raises(FrozenSourceError, match="collision"):
        parse_tree(raw)


def test_clean_head_rejects_tracked_drift_and_untracked_shadow(repo: Path) -> None:
    (repo / "algua/__init__.py").write_text("changed\n")
    with pytest.raises(FrozenSourceError, match="tracked"):
        assert_clean_head(repo)
    _git(repo, "checkout", "--", "algua/__init__.py")
    (repo / "algua/shadow.py").write_text("x = 1\n")
    with pytest.raises(FrozenSourceError, match="untracked"):
        assert_clean_head(repo)


def test_clean_head_rejects_untracked_symlink(repo: Path) -> None:
    os.symlink("__init__.py", repo / "algua/shadow.py")
    with pytest.raises(FrozenSourceError, match="untracked"):
        assert_clean_head(repo)


def test_asset_rejection_does_not_dereference_model_handle() -> None:
    class ExplosiveHandle:
        @property
        def version(self):
            raise AssertionError("model path/bytes were dereferenced")

    with pytest.raises(FrozenAssetsUnsupported):
        require_source_only(ExplosiveHandle())
    require_source_only(None)
