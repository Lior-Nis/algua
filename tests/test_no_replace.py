from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from algua.primitives import no_replace
from algua.primitives.no_replace import NoReplaceUnsupported, rename_noreplace


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "stage"
    source.mkdir()
    (source / "payload").write_text("owned")
    return source


def test_rename_into_an_absent_destination(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "published"

    rename_noreplace(source, destination)

    assert not source.exists()
    assert (destination / "payload").read_text() == "owned"


@pytest.mark.parametrize("occupant", ["empty-directory", "directory", "file", "dangling-link"])
def test_existing_destination_is_never_replaced(tmp_path: Path, occupant: str) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "published"
    if occupant == "empty-directory":
        destination.mkdir()
    elif occupant == "directory":
        destination.mkdir()
        (destination / "winner").write_text("winner")
    elif occupant == "file":
        destination.write_text("winner")
    else:
        destination.symlink_to(tmp_path / "missing")

    with pytest.raises(FileExistsError) as caught:
        rename_noreplace(source, destination)

    assert caught.value.errno == errno.EEXIST
    assert (source / "payload").read_text() == "owned"
    if occupant == "empty-directory":
        assert destination.is_dir() and not any(destination.iterdir())
    elif occupant == "directory":
        assert (destination / "winner").read_text() == "winner"
    elif occupant == "file":
        assert destination.read_text() == "winner"
    else:
        assert destination.is_symlink()


def test_missing_kernel_primitive_fails_closed_without_replacement_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    monkeypatch.setattr(no_replace, "_load_renameat2", lambda: None)
    for name in ("rename", "replace", "renames"):
        monkeypatch.setattr(
            os, name, lambda *_args, **_kwargs: pytest.fail("replacement-capable fallback"))

    with pytest.raises(NoReplaceUnsupported):
        rename_noreplace(source, tmp_path / "published")

    assert (source / "payload").read_text() == "owned"
    assert not (tmp_path / "published").exists()


@pytest.mark.parametrize("code", [errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP])
def test_unsupported_flag_or_filesystem_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int,
) -> None:
    source = _source(tmp_path)
    monkeypatch.setattr(no_replace, "_load_renameat2", lambda: lambda _src, _dst: code)

    with pytest.raises(NoReplaceUnsupported) as caught:
        rename_noreplace(source, tmp_path / "published")

    assert caught.value.errno == code
    assert (source / "payload").read_text() == "owned"


def test_other_kernel_failures_keep_their_errno(tmp_path: Path) -> None:
    with pytest.raises(OSError) as caught:
        rename_noreplace(tmp_path / "absent", tmp_path / "published")

    assert caught.value.errno == errno.ENOENT
    assert not isinstance(caught.value, (FileExistsError, NoReplaceUnsupported))


def test_non_linux_platforms_are_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(no_replace.sys, "platform", "darwin")

    assert no_replace._load_renameat2() is None
