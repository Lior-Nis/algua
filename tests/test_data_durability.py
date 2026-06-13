from __future__ import annotations

import os
from pathlib import Path

import pytest

from algua.data import files


def test_fsync_file_and_dir_run_on_real_paths(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    files.fsync_file(f)  # must not raise
    files.fsync_dir(tmp_path)  # must not raise


def test_fsync_dir_rejects_non_directory(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"x")
    with pytest.raises(OSError):  # O_DIRECTORY on a file -> ENOTDIR
        files.fsync_dir(f)


def test_fsync_tree_visits_files_before_their_parent_dirs(tmp_path: Path, monkeypatch) -> None:
    for sym in ("A", "B"):
        d = tmp_path / f"symbol={sym}"
        d.mkdir()
        (d / "part-0.parquet").write_bytes(b"data")

    order: list[tuple[str, bool]] = []
    real_open = os.open

    def spy_open(path, flags, *a, **k):
        fd = real_open(path, flags, *a, **k)
        order.append((str(path), bool(flags & os.O_DIRECTORY)))
        return fd

    monkeypatch.setattr(os, "open", spy_open)
    files.fsync_tree(tmp_path)

    # root itself is fsynced last; every symbol dir fsync comes after its own part file
    assert order[-1] == (str(tmp_path), True)
    for sym in ("A", "B"):
        d = str(tmp_path / f"symbol={sym}")
        f = str(tmp_path / f"symbol={sym}" / "part-0.parquet")
        file_idx = next(i for i, (p, is_dir) in enumerate(order) if p == f and not is_dir)
        dir_idx = next(i for i, (p, is_dir) in enumerate(order) if p == d and is_dir)
        assert file_idx < dir_idx


def test_fsync_parents_walks_up_to_stop_at_inclusive(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    leaf = root / "snapshots" / "bars" / "abc123"
    leaf.mkdir(parents=True)
    payload = leaf / "symbol=A"
    payload.mkdir()

    fsynced: list[str] = []
    real_open = os.open

    def spy_open(path, flags, *a, **k):
        if flags & os.O_DIRECTORY:
            fsynced.append(str(path))
        return real_open(path, flags, *a, **k)

    monkeypatch.setattr(os, "open", spy_open)
    files.fsync_parents(payload, stop_at=root)

    assert str(leaf) in fsynced
    assert str(root / "snapshots" / "bars") in fsynced
    assert str(root / "snapshots") in fsynced
    assert str(root) in fsynced
    assert str(root.parent) not in fsynced
