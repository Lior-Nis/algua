"""algua.primitives.atomic_io — one atomic/durable write implementation (spec §4.2)."""
from __future__ import annotations

import pytest

from algua.primitives.atomic_io import (
    fsync_dir,
    fsync_file,
    fsync_parents,
    fsync_tree,
    write_bytes_atomic,
    write_bytes_durable,
    write_text_atomic,
)


def test_write_bytes_atomic_creates_parent_and_no_temp_residue(tmp_path):
    dest = tmp_path / "sub" / "out.bin"
    write_bytes_atomic(b"payload", dest)
    assert dest.read_bytes() == b"payload"
    assert [p.name for p in dest.parent.iterdir()] == ["out.bin"]  # temp cleaned up


def test_write_text_atomic_roundtrip(tmp_path):
    dest = tmp_path / "note.md"
    write_text_atomic("hello", dest)
    write_text_atomic("world", dest)  # overwrite is atomic too
    assert dest.read_text() == "world"


def test_write_bytes_durable_with_root_chain(tmp_path):
    root = tmp_path / "store"
    dest = root / "a" / "b" / "x.bin"
    write_bytes_durable(b"d", dest, durable_root=root)
    assert dest.read_bytes() == b"d"


def test_fsync_parents_rejects_path_outside_root(tmp_path):
    inside = tmp_path / "root"
    inside.mkdir()
    outside = tmp_path / "elsewhere" / "f"
    outside.parent.mkdir()
    outside.touch()
    with pytest.raises(ValueError):
        fsync_parents(outside, stop_at=inside)


def test_fsync_helpers_run_on_real_objects(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x")
    fsync_file(f)
    fsync_dir(tmp_path)
    fsync_tree(tmp_path)
    with pytest.raises(OSError):
        fsync_dir(f)  # O_DIRECTORY on a file fails loudly
