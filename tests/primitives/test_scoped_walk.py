"""A consumer's own error stays primary when closing its walk also fails."""
from __future__ import annotations

import errno
import inspect
import re
from pathlib import Path

import pytest

from algua.primitives import bounded_walk as walk_module
from algua.primitives.bounded_walk import TraversalLimitExceeded
from tests._walk_faults import track_closes

REPO = Path(__file__).resolve().parents[2]


class TypedRefusal(ValueError):
    """Stands in for a consumer's typed validation error."""


def _scoped(root: Path, *, files: int = 100):
    return walk_module.scoped_walk(
        root, max_files=files, max_directories=100, max_path_bytes=1024)


def _chain(root: Path) -> None:
    (root / "d1/d2/d3").mkdir(parents=True)
    for index in range(5):
        (root / f"d1/d2/f{index}").touch()


@pytest.mark.parametrize("fault", [RuntimeError, ValueError, errno.EIO])
def test_a_consumer_error_stays_primary_when_closing_the_walk_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: object,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=fault)

    with pytest.raises(TypedRefusal), _scoped(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                raise TypedRefusal("consumer refused this entry")

    assert len(closed) == 3 and all(closed.values()), closed


def test_a_generator_exit_from_a_close_is_visible_after_an_early_scope_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=GeneratorExit)

    with pytest.raises(RuntimeError) as caught, _scoped(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                break

    assert type(caught.value) is getattr(walk_module, "WalkCleanupError", None)
    assert isinstance(caught.value.__cause__, GeneratorExit)
    assert all(closed.values()), closed


def test_a_generator_exit_from_a_close_never_displaces_a_consumer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=GeneratorExit)

    with pytest.raises(TypedRefusal), _scoped(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                raise TypedRefusal("consumer refused this entry")

    assert all(closed.values()), closed


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_a_cleanup_interrupt_is_never_swallowed_by_a_consumer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: type[BaseException],
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=interrupt)

    with pytest.raises(interrupt) as caught, _scoped(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                raise TypedRefusal("consumer refused this entry")

    assert isinstance(caught.value.__cause__, TypedRefusal)
    assert all(closed.values()), closed


def test_an_early_exit_still_reports_the_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=RuntimeError)

    with pytest.raises(walk_module.WalkCleanupError) as caught, _scoped(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                break

    assert type(caught.value.__cause__) is RuntimeError
    assert all(closed.values()), closed


def test_a_traversal_failure_stays_primary_inside_the_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=RuntimeError)

    with pytest.raises(TraversalLimitExceeded), _scoped(tmp_path, files=1) as tree:
        for _entry in tree:
            pass

    assert all(closed.values()), closed


def test_a_completed_scope_closes_cleanly(tmp_path: Path) -> None:
    _chain(tmp_path)

    with _scoped(tmp_path) as tree:
        relatives = [entry.relative for entry in tree]

    assert "d1/d2/f4" in relatives


class TypedCleanupFailure(RuntimeError):
    """Stands in for a consumer's typed cleanup error."""


def _translating(root: Path):
    return walk_module.scoped_walk(
        root, max_files=100, max_directories=100, max_path_bytes=1024,
        cleanup_error=lambda: TypedCleanupFailure("listing could not be closed"))


def test_the_walks_own_exhausted_close_failure_is_translated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1/d2/d3", fault=RuntimeError)

    with pytest.raises(TypedCleanupFailure) as caught, _translating(tmp_path) as tree:
        for _entry in tree:
            pass

    assert type(caught.value.__cause__) is walk_module.WalkCleanupError
    assert all(closed.values()), closed


def test_the_walks_own_close_failure_after_an_early_exit_is_translated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=RuntimeError)

    with pytest.raises(TypedCleanupFailure), _translating(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                break

    assert all(closed.values()), closed


def test_a_walk_cleanup_error_raised_by_the_body_is_never_translated(tmp_path: Path) -> None:
    _chain(tmp_path)
    body_error = walk_module.WalkCleanupError("raised by the consumer body itself")

    with pytest.raises(walk_module.WalkCleanupError) as caught, _translating(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                raise body_error

    assert caught.value is body_error


def test_an_early_scope_exit_retries_a_listing_that_failed_before_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", release=False, times=1)

    with pytest.raises(walk_module.WalkCleanupError), _scoped(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                break

    assert len(closed) == 3 and all(closed.values()), closed


def test_a_consumer_error_retries_a_listing_that_failed_before_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", release=False, times=1)

    with pytest.raises(TypedRefusal), _scoped(tmp_path) as tree:
        for entry in tree:
            if entry.relative == "d1/d2/d3":
                raise TypedRefusal("consumer refused this entry")

    assert len(closed) == 3 and all(closed.values()), closed


def _consume(scope):
    """A consumer that is itself a generator, yielding from inside its scope."""
    with scope as tree:
        for entry in tree:
            yield entry.relative


def _abandon(scope, stop: str = "d1/d2/d3") -> None:
    consumer = _consume(scope)
    for relative in consumer:
        if relative == stop:
            break
    consumer.close()  # the consumer, and with it the entered scope, is abandoned


@pytest.mark.parametrize("fault", [RuntimeError, ValueError, errno.EIO])
def test_an_abandoned_scope_reports_the_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: object,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=fault)

    with pytest.raises(walk_module.WalkCleanupError):
        _abandon(_scoped(tmp_path))

    assert len(closed) == 3 and all(closed.values()), closed


def test_an_abandoned_scope_translates_its_own_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1")

    with pytest.raises(TypedCleanupFailure) as caught:
        _abandon(_translating(tmp_path))

    assert type(caught.value.__cause__) is walk_module.WalkCleanupError
    assert len(closed) == 3 and all(closed.values()), closed


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_while_closing_an_abandoned_scope_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: type[BaseException],
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=tmp_path / "d1", fault=interrupt)

    with pytest.raises(interrupt):
        _abandon(_translating(tmp_path))

    assert len(closed) == 3 and all(closed.values()), closed


def test_an_abandoned_scope_closes_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _chain(tmp_path)
    closed = track_closes(monkeypatch, faulty=lambda _path: False)

    _abandon(_scoped(tmp_path))

    assert len(closed) == 3 and all(closed.values()), closed


def test_a_walk_used_after_its_scope_closed_fails_loudly(tmp_path: Path) -> None:
    _chain(tmp_path)
    with _scoped(tmp_path) as tree:
        pass

    with pytest.raises(RuntimeError, match="after its scope closed"):
        next(tree)


# The raw walk is module-private, so consumers can only reach it through `scoped_walk`. This
# guards against accidental direct use only; deliberately obfuscated access (string-built names,
# `vars()` lookups) is out of scope and left to code review.
WALK_SOURCE = REPO / "algua/primitives/bounded_walk.py"
RAW_WALK = re.compile(r"\b_bounded_walk\b")


@pytest.mark.parametrize("source", [
    "from algua.primitives.bounded_walk import _bounded_walk\n",
    "from algua.primitives import bounded_walk as w\nw._bounded_walk(root)\n",
    "import algua.primitives.bounded_walk\nalgua.primitives.bounded_walk._bounded_walk(root)\n",
    "walk = getattr(module, '_bounded_walk')\n",
])
def test_the_guard_flags_direct_raw_walk_spellings(source: str) -> None:
    assert RAW_WALK.search(source)


@pytest.mark.parametrize("source", [
    "from algua.primitives.bounded_walk import scoped_walk\n",
    "from algua.primitives import bounded_walk\nbounded_walk.scoped_walk(root)\n",
    "not_bounded_walk = 1\n",
])
def test_the_guard_allows_scoped_use_and_the_module_name(source: str) -> None:
    assert not RAW_WALK.search(source)


def test_scoped_walk_is_the_walk_modules_only_public_function() -> None:
    public = {
        name for name, value in vars(walk_module).items()
        if inspect.isfunction(value) and not name.startswith("_")
        and value.__module__ == walk_module.__name__
    }
    assert public == {"scoped_walk"}


def test_no_other_module_names_the_raw_walk() -> None:
    direct = [
        str(path.relative_to(REPO))
        for root in ("algua", "scripts") for path in sorted((REPO / root).rglob("*.py"))
        if path != WALK_SOURCE and RAW_WALK.search(path.read_text(encoding="utf-8"))
    ]
    assert direct == []
