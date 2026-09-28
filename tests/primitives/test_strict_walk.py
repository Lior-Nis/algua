from __future__ import annotations

import os
from pathlib import Path

import pytest

from algua.primitives.strict_walk import strict_walk
from tests._walk_faults import fail_scandir_once


def _tree(root: Path) -> None:
    (root / "a/b").mkdir(parents=True)
    (root / "a/b/file").write_text("x")
    (root / "c").mkdir()
    (root / "c/link").symlink_to(root / "a", target_is_directory=True)


@pytest.mark.parametrize("topdown", [True, False])
def test_strict_walk_matches_os_walk_without_following_links(
    tmp_path: Path, topdown: bool,
) -> None:
    _tree(tmp_path)

    assert list(strict_walk(tmp_path, topdown=topdown)) == list(
        os.walk(tmp_path, topdown=topdown, followlinks=False))


@pytest.mark.parametrize("topdown", [True, False])
def test_strict_walk_propagates_a_subtree_listing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, topdown: bool,
) -> None:
    _tree(tmp_path)
    failed = fail_scandir_once(monkeypatch, lambda path: path == tmp_path / "a/b")

    with pytest.raises(PermissionError):
        list(strict_walk(tmp_path, topdown=topdown))
    assert failed == [tmp_path / "a/b"]


def test_strict_walk_propagates_a_root_listing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fail_scandir_once(monkeypatch, lambda path: path == tmp_path)

    with pytest.raises(PermissionError):
        list(strict_walk(tmp_path))
