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


@pytest.mark.parametrize("name", [
    "gone.cpython-312.pyc", "gone.cpython-311.opt-1.pyc", "_tmp_probe.cpython-312.opt-2.pyc",
])
def test_clean_head_accepts_orphaned_bytecode_caches(repo: Path, name: str) -> None:
    """Python never imports a ``__pycache__`` entry whose source is gone (PEP 3147), so the caches
    a deleted module (or a test's temporary module) leaves behind cannot shadow anything."""
    cache = repo / "algua/__pycache__"
    cache.mkdir()
    (cache / name).write_bytes(b"orphaned")
    assert assert_clean_head(repo) == _git(repo, "rev-parse", "HEAD")


def test_clean_head_accepts_a_deleted_package_that_left_only_its_caches(repo: Path) -> None:
    """Git removes a deleted package's tracked files but not its untracked ``__pycache__``, so the
    directory survives holding caches alone; it can only ever import as an empty namespace."""
    cache = repo / "algua/removed/__pycache__"
    cache.mkdir(parents=True)
    (cache / "__init__.cpython-312.pyc").write_bytes(b"orphaned")
    (cache / "helper.cpython-312.pyc").write_bytes(b"orphaned")
    assert assert_clean_head(repo) == _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize(("path", "kind"), [
    ("algua/stray.py", "untracked Python source"),
    ("algua/tool.pyc", "sourceless bytecode"),  # legacy sourceless bytecode IS importable
    ("algua/legacy.pyc", "sourceless bytecode"),
    ("algua/removed/helper.cpython-312.pyc", "sourceless bytecode"),  # not inside __pycache__
    ("algua/__pycache__/foreign.pyc", "malformed bytecode cache"),
    ("algua/__pycache__/gone.cpython-312.pyc.140234", "malformed bytecode cache"),
    ("algua/__pycache__/gone.cpython.pyc", "malformed bytecode cache"),
    ("algua/__pycache__/gone.cpython-312.opt-.pyc", "malformed bytecode cache"),
    ("algua/__pycache__/gone-x.cpython-312.pyc", "malformed bytecode cache"),
    ("algua/__pycache__/shadow.py", "untracked Python source"),  # importable as a namespace
    ("algua/__pycache__/nested/gone.cpython-312.pyc", "sourceless bytecode"),
    ("algua/notes.txt", "untracked file"),
])
def test_clean_head_still_refuses_everything_but_well_formed_caches(
    repo: Path, path: str, kind: str,
) -> None:
    """Only a well-formed ``<name>.<cache_tag>[.opt-N].pyc`` directly inside ``__pycache__`` is
    accepted; everything else is refused and the refusal names the offending path's class."""
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x")
    with pytest.raises(FrozenSourceError, match="untracked") as info:
        assert_clean_head(repo)
    assert kind in str(info.value)
    assert repr(path) in str(info.value)


def test_clean_head_refuses_an_orphaned_cache_beside_a_stray_source(repo: Path) -> None:
    cache = repo / "algua/__pycache__"
    cache.mkdir()
    (cache / "gone.cpython-312.pyc").write_bytes(b"orphaned")
    (repo / "algua/gone.py").write_text("x = 1\n")
    with pytest.raises(FrozenSourceError, match="untracked Python source 'algua/gone.py'"):
        assert_clean_head(repo)


def test_clean_head_names_an_untracked_symlink_and_directory(repo: Path) -> None:
    os.symlink("__init__.py", repo / "algua/shadow.py")
    with pytest.raises(FrozenSourceError, match="untracked symlink 'algua/shadow.py'"):
        assert_clean_head(repo)
    (repo / "algua/shadow.py").unlink()
    (repo / "algua/empty").mkdir()
    with pytest.raises(FrozenSourceError, match="untracked directory 'algua/empty'"):
        assert_clean_head(repo)


def test_clean_head_refuses_a_symlink_even_with_a_cache_name(repo: Path) -> None:
    cache = repo / "algua/__pycache__"
    cache.mkdir()
    (repo / "elsewhere.pyc").write_bytes(b"x")
    os.symlink(repo / "elsewhere.pyc", cache / "__init__.cpython-312.pyc")
    with pytest.raises(FrozenSourceError, match="untracked symlink"):
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


def _commit_long_paths(repo: Path, count: int, *, top: str = "vendor") -> list[str]:
    """Commit ``count`` non-source files whose paths are each ~500 UTF-8 bytes."""
    paths = []
    for index in range(count):
        relative = f"{top}/{'d' * 240}/{index:03d}{'f' * 240}.txt"
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        (repo / relative).write_text("data\n")
        paths.append(relative)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "long non-source paths")
    return paths


def test_index_flag_scan_is_not_bounded_by_the_source_listing_aggregate(
    repo: Path, monkeypatch,
) -> None:
    """The repository-wide index may be far larger than the source-entry aggregate bound; the
    hidden-flag scan must still inspect all of it rather than fail on its total size."""
    import algua.registry.frozen_source as frozen_source

    paths = _commit_long_paths(repo, 12)
    monkeypatch.setattr(frozen_source, "MAX_SOURCE_FILES", 2)
    aggregate = (frozen_source.MAX_PATH_BYTES + 100) * (frozen_source.MAX_SOURCE_FILES + 1)
    assert len(_git(repo, "ls-files", "-z", "-v").encode()) > aggregate
    assert assert_clean_head(repo) == _git(repo, "rev-parse", "HEAD")

    _git(repo, "update-index", "--skip-worktree", paths[-1])
    with pytest.raises(FrozenSourceError, match="hide tracked"):
        assert_clean_head(repo)


def test_index_flag_scan_accepts_non_source_paths_beyond_the_source_path_bound(
    repo: Path,
) -> None:
    deep = "vendor/" + "/".join(["p" * 200] * 7) + "/file.txt"
    assert len(deep.encode()) > 1_024
    (repo / deep).parent.mkdir(parents=True)
    (repo / deep).write_text("data\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "deep non-source path")
    assert assert_clean_head(repo) == _git(repo, "rev-parse", "HEAD")


def test_index_flag_scan_rejects_tracked_source_beyond_its_bounds(repo: Path, monkeypatch) -> None:
    import algua.registry.frozen_source as frozen_source

    monkeypatch.setattr(frozen_source, "MAX_SOURCE_FILES", 1)
    with pytest.raises(FrozenSourceError, match="file-count"):
        assert_clean_head(repo)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 7, 4096])
def test_index_record_scan_is_independent_of_chunk_boundaries(chunk_size: int) -> None:
    from algua.registry.frozen_source import _scan_index_records

    listing = b"H algua/a.py\0H README.md\0H algua/sub/b.py\0H " + b"x" * 5000 + b"\0"
    chunks = (listing[i:i + chunk_size] for i in range(0, len(listing), chunk_size))
    assert _scan_index_records(chunks) == {"algua/a.py", "algua/sub/b.py"}


@pytest.mark.parametrize(
    "listing,message",
    [
        (b"H algua/a.py\0h README.md\0", "hide tracked"),
        (b"H algua/a.py\0S docs/x.md\0", "hide tracked"),
        (b"H algua/a.py", "NUL terminated"),
        (b"H algua/" + b"x" * 1_100 + b"\0", "path bound"),
        (b"H algua/\xff.py\0", "UTF-8"),
    ],
    ids=["assume-unchanged", "skip-worktree", "unterminated", "long-source-path", "invalid-utf8"],
)
def test_index_record_scan_fails_closed(listing: bytes, message: str) -> None:
    from algua.registry.frozen_source import _scan_index_records

    with pytest.raises(FrozenSourceError, match=message):
        _scan_index_records(iter([listing]))


@pytest.mark.parametrize("delivery", ["streamed", "single-chunk"])
def test_index_record_scan_retains_bounded_memory_for_huge_records(delivery: str) -> None:
    import tracemalloc

    from algua.registry.frozen_source import _scan_index_records

    def huge_listing():
        yield b"H vendor/"
        for _ in range(512):  # one 32 MiB non-source record
            yield b"x" * 65_536
        yield b"\0H algua/a.py\0"

    # A single pre-built chunk is allocated before tracing starts; only retention is measured.
    listing = iter([b"".join(huge_listing())]) if delivery == "single-chunk" else huge_listing()
    tracemalloc.start()
    try:
        assert _scan_index_records(listing) == {"algua/a.py"}
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1024 * 1024
