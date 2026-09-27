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


def test_clean_head_rejects_untracked_empty_directory(repo: Path) -> None:
    (repo / "algua/shadow").mkdir()
    with pytest.raises(FrozenSourceError, match="untracked"):
        assert_clean_head(repo)


def test_clean_head_allows_only_recognized_generated_cache(repo: Path) -> None:
    cache = repo / "algua/__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"generated")
    assert assert_clean_head(repo) == _git(repo, "rev-parse", "HEAD")
    (cache / "foreign.pyc").write_bytes(b"generated")
    with pytest.raises(FrozenSourceError, match="untracked"):
        assert_clean_head(repo)


def test_export_checks_blob_size_before_reading(monkeypatch, tmp_path: Path) -> None:
    oid = "a" * 40
    calls: list[tuple[str, ...]] = []

    def fake_git(_root, *args, max_bytes):
        calls.append(args)
        if args[0] == "ls-tree":
            return f"100644 blob {oid}\talgua/x.py\0".encode()
        if args[:2] == ("cat-file", "-s"):
            return str(64 * 1024 * 1024 + 1).encode()
        raise AssertionError("oversized blob bytes were read")

    monkeypatch.setattr("algua.registry.frozen_source._git", fake_git)
    with pytest.raises(FrozenSourceError, match="per-file"):
        export_source(tmp_path, "b" * 40)
    assert not any(args[:2] == ("cat-file", "blob") for args in calls)


def test_asset_rejection_does_not_dereference_model_handle() -> None:
    class ExplosiveHandle:
        @property
        def version(self):
            raise AssertionError("model path/bytes were dereferenced")

    with pytest.raises(FrozenAssetsUnsupported):
        require_source_only(ExplosiveHandle())
    require_source_only(None)


def test_git_output_is_stopped_once_it_exceeds_the_protected_bound(
    monkeypatch, tmp_path: Path,
) -> None:
    import time

    from algua.registry.frozen_source import _git as bounded_git

    marker = tmp_path / "finished"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "git"
    fake.write_text(f"#!/bin/sh\nhead -c 4096 /dev/zero\nsleep 5\ntouch {marker}\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    started = time.monotonic()
    with pytest.raises(FrozenSourceError, match="protected bound"):
        bounded_git(tmp_path, "cat-file", "blob", "a" * 40, max_bytes=1024)
    assert time.monotonic() - started < 4
    time.sleep(0.1)
    assert not marker.exists()


def test_git_output_within_the_bound_is_returned_exactly(repo: Path) -> None:
    from algua.registry.frozen_source import _git as bounded_git

    oid = _git(repo, "rev-parse", "HEAD:algua/__init__.py")
    data = b"VALUE = 'committed'\n"
    assert bounded_git(repo, "cat-file", "blob", oid, max_bytes=len(data)) == data
    with pytest.raises(FrozenSourceError, match="protected bound"):
        bounded_git(repo, "cat-file", "blob", oid, max_bytes=len(data) - 1)
    with pytest.raises(FrozenSourceError, match="could not be read"):
        bounded_git(repo, "cat-file", "blob", "f" * 40, max_bytes=1024)


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
@pytest.mark.parametrize("path", ["algua/__init__.py", "pyproject.toml"])
def test_clean_head_rejects_index_flags_that_hide_tracked_drift(
    repo: Path, flag: str, path: str,
) -> None:
    _git(repo, "update-index", flag, path)
    (repo / path).write_text("hidden drift\n")
    assert _git(repo, "status", "--porcelain") == ""
    with pytest.raises(FrozenSourceError, match="hide tracked"):
        assert_clean_head(repo)


def test_clean_head_rejects_hidden_flags_even_without_content_drift(repo: Path) -> None:
    _git(repo, "update-index", "--skip-worktree", "algua/tool.py")
    with pytest.raises(FrozenSourceError, match="hide tracked"):
        assert_clean_head(repo)
