from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from algua.primitives.bounded_walk import TraversalLimitExceeded, bounded_walk
from tests._walk_faults import count_scandir_pulls, fail_scandir_once, track_closes


def _walk(root: Path, *, files: int = 100, directories: int = 100, path_bytes: int = 1024):
    return list(bounded_walk(
        root, max_files=files, max_directories=directories, max_path_bytes=path_bytes))


def _tree(root: Path) -> None:
    (root / "a/b").mkdir(parents=True)
    (root / "a/b/file").write_text("x")
    (root / "a/top").write_text("y")
    (root / "c").mkdir()
    (root / "c/link").symlink_to(root / "a", target_is_directory=True)
    (root / "c/dangling").symlink_to(root / "missing")


def test_yields_every_entry_once_directory_first_without_following_links(
    tmp_path: Path,
) -> None:
    _tree(tmp_path)

    entries = _walk(tmp_path)
    by_path = {entry.relative: entry for entry in entries}

    assert sorted(by_path) == ["a", "a/b", "a/b/file", "a/top", "c", "c/dangling", "c/link"]
    assert len(entries) == len(by_path)
    assert [name for name, entry in sorted(by_path.items()) if entry.is_dir] == ["a", "a/b", "c"]
    assert {name for name, entry in by_path.items() if entry.is_symlink} == {
        "c/dangling", "c/link"}
    assert by_path["c/link"].is_dir is False
    assert all(entry.path == tmp_path / entry.relative for entry in entries)
    order = [entry.relative for entry in entries]
    assert order.index("a") < order.index("a/b") < order.index("a/b/file")


@pytest.mark.parametrize("failing", ["", "a", "a/b"])
def test_listing_errors_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: str,
) -> None:
    _tree(tmp_path)
    target = tmp_path / failing if failing else tmp_path
    failed = fail_scandir_once(monkeypatch, lambda path: path == target)

    with pytest.raises(PermissionError):
        _walk(tmp_path)
    assert failed == [target]


class _Failing:
    """An iterator that yields its first real entry, then fails like a mid-listing EIO."""

    def __init__(self, inner, fail_on: str) -> None:
        self._inner = inner
        self._fail_on = fail_on
        self._pulled = 0

    def __iter__(self):
        return self

    def __next__(self):
        self._pulled += 1
        if self._pulled == 2 and self._fail_on == "iteration":
            raise OSError(errno.EIO, "injected listing fault")
        entry = next(self._inner)
        if self._fail_on == "type":
            return _BrokenEntry(entry)
        return entry

    def close(self) -> None:
        self._inner.close()


class _BrokenEntry:
    def __init__(self, entry) -> None:
        self.name = entry.name
        self.path = entry.path

    def is_symlink(self) -> bool:
        raise OSError(errno.EIO, "injected lstat fault")

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        raise OSError(errno.EIO, "injected lstat fault")


@pytest.mark.parametrize("fail_on", ["iteration", "type"])
def test_iteration_and_type_check_errors_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_on: str,
) -> None:
    for name in ("one", "two", "three"):
        (tmp_path / name).write_text(name)
    original = os.scandir
    monkeypatch.setattr(os, "scandir", lambda path: _Failing(original(path), fail_on))

    with pytest.raises(OSError) as caught:
        _walk(tmp_path)
    assert caught.value.errno == errno.EIO


def test_exactly_the_bounds_are_accepted(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"d{index}").mkdir()
        (tmp_path / f"d{index}/f").write_text("x")

    assert len(_walk(tmp_path, files=3, directories=3)) == 6


def test_one_directory_with_huge_file_fanout_is_refused_while_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(2000):
        (tmp_path / f"f{index:04d}").touch()
    pulls = count_scandir_pulls(monkeypatch)
    received: list[str] = []

    with pytest.raises(TraversalLimitExceeded) as caught:
        for entry in bounded_walk(tmp_path, max_files=10, max_directories=10,
                                  max_path_bytes=1024):
            received.append(entry.relative)

    assert caught.value.kind == "file-count"
    assert pulls[tmp_path] == 11
    assert len(received) == 10  # the entry past the bound is never handed to the consumer


def test_empty_directory_fanout_is_refused_while_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(2000):
        (tmp_path / f"d{index:04d}").mkdir()
    pulls = count_scandir_pulls(monkeypatch)

    with pytest.raises(TraversalLimitExceeded) as caught:
        _walk(tmp_path, directories=10)

    assert caught.value.kind == "directory-count"
    assert pulls[tmp_path] == 11
    assert sum(1 for path in pulls if path != tmp_path) <= 10


def test_deep_empty_directory_chain_is_refused(tmp_path: Path) -> None:
    deepest = tmp_path.joinpath(*["d"] * 50)
    deepest.mkdir(parents=True)

    with pytest.raises(TraversalLimitExceeded) as caught:
        _walk(tmp_path, directories=10)
    assert caught.value.kind == "directory-count"


def test_path_length_is_bounded_before_descending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    long = tmp_path / ("a" * 40) / ("b" * 40)
    long.mkdir(parents=True)
    (long / "file").touch()
    pulls = count_scandir_pulls(monkeypatch)

    with pytest.raises(TraversalLimitExceeded) as caught:
        _walk(tmp_path, path_bytes=60)

    assert caught.value.kind == "path-length"
    assert long not in pulls
    assert len(_walk(tmp_path, path_bytes=len("a" * 40 + "/" + "b" * 40 + "/file"))) == 3


def test_path_length_counts_encoded_bytes_not_characters(tmp_path: Path) -> None:
    (tmp_path / ("é" * 30)).touch()  # 30 characters, 60 UTF-8 bytes

    with pytest.raises(TraversalLimitExceeded):
        _walk(tmp_path, path_bytes=50)
    assert len(_walk(tmp_path, path_bytes=60)) == 1


def test_an_abandoned_walk_closes_every_directory_handle(tmp_path: Path) -> None:
    import gc
    import warnings

    deepest = tmp_path.joinpath(*["d"] * 20)
    deepest.mkdir(parents=True)
    (deepest / "f").touch()
    before = len(os.listdir("/proc/self/fd"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        walk = bounded_walk(tmp_path, max_files=10, max_directories=100, max_path_bytes=1024)
        for entry in walk:
            if entry.relative.count("/") == 15:
                break
        walk.close()
        del walk
        gc.collect()

    assert len(os.listdir("/proc/self/fd")) == before
    assert not [item for item in caught if issubclass(item.category, ResourceWarning)]


def _chain(root: Path) -> None:
    (root / "d1/d2/d3").mkdir(parents=True)
    for index in range(5):
        (root / f"d1/d2/f{index}").touch()


def test_an_abandoned_walk_closes_every_handle_and_reports_the_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1")
    walk = bounded_walk(tmp_path, max_files=100, max_directories=100, max_path_bytes=1024)
    for entry in walk:
        if entry.relative == "d1/d2/d3":
            break

    with pytest.raises(OSError) as caught:
        walk.close()

    assert caught.value.errno == errno.EIO
    assert closed and all(closed.values()), closed


def test_the_deepest_cleanup_failure_is_reported_when_several_closes_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(
        monkeypatch, faulty=tmp_path / "d1", also={tmp_path / "d1/d2": errno.ENOSPC})
    walk = bounded_walk(tmp_path, max_files=100, max_directories=100, max_path_bytes=1024)
    for entry in walk:
        if entry.relative == "d1/d2/d3":
            break

    with pytest.raises(OSError) as caught:
        walk.close()

    assert caught.value.errno == errno.ENOSPC  # d1/d2 is closed before d1
    assert all(closed.values()), closed


def _abandon_at_d3(tmp_path: Path):
    walk = bounded_walk(tmp_path, max_files=100, max_directories=100, max_path_bytes=1024)
    for entry in walk:
        if entry.relative == "d1/d2/d3":
            break
    return walk


@pytest.mark.parametrize("fault", [RuntimeError, ValueError, KeyboardInterrupt, SystemExit])
def test_an_abandoned_walk_closes_every_handle_whatever_a_close_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: type[BaseException],
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=fault)
    walk = _abandon_at_d3(tmp_path)

    with pytest.raises(fault):
        walk.close()

    assert len(closed) == 3 and all(closed.values()), closed


@pytest.mark.parametrize("fault", [RuntimeError, ValueError, errno.EIO])
def test_an_active_traversal_error_survives_any_ordinary_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: object,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=fault)

    with pytest.raises(TraversalLimitExceeded):
        _walk(tmp_path, files=1)

    assert all(closed.values()), closed


def test_an_interrupt_during_cleanup_is_never_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=SystemExit)

    with pytest.raises(SystemExit) as caught:
        _walk(tmp_path, files=1)

    assert isinstance(caught.value.__context__, TraversalLimitExceeded)
    assert all(closed.values()), closed


def test_an_interrupt_outranks_an_earlier_ordinary_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(
        monkeypatch, faulty=tmp_path / "d1", fault=SystemExit,
        also={tmp_path / "d1/d2": RuntimeError})
    walk = _abandon_at_d3(tmp_path)

    with pytest.raises(SystemExit):  # d1/d2 (RuntimeError) closes first, d1 interrupts later
        walk.close()

    assert all(closed.values()), closed


def test_an_exhausted_listing_whose_close_fails_once_is_released_by_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(
        monkeypatch, faulty=tmp_path / "d1/d2/d3", release=False, times=1)

    with pytest.raises(OSError) as caught:
        _walk(tmp_path)

    assert caught.value.errno == errno.EIO
    assert all(closed.values()), closed  # the failed close was retried, not dropped


def test_an_active_traversal_error_survives_a_failing_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1")

    with pytest.raises(TraversalLimitExceeded):
        _walk(tmp_path, files=1)

    assert closed and all(closed.values()), closed


def test_an_active_listing_error_survives_a_failing_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1")
    tracked_scandir = os.scandir

    def failing(path):
        if Path(os.fsdecode(path)) == tmp_path / "d1/d2/d3":
            raise PermissionError(errno.EACCES, "injected listing fault")
        return tracked_scandir(path)

    monkeypatch.setattr(os, "scandir", failing)

    with pytest.raises(PermissionError):
        _walk(tmp_path)

    assert closed and all(closed.values()), closed


def test_a_failing_close_of_an_exhausted_directory_still_closes_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1/d2/d3")

    with pytest.raises(OSError) as caught:
        _walk(tmp_path)

    assert caught.value.errno == errno.EIO
    assert closed and all(closed.values()), closed
