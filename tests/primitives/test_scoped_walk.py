"""A consumer's own error stays primary when closing its walk also fails."""
from __future__ import annotations

import ast
import errno
import importlib.util
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


WALK_MODULE = "algua.primitives.bounded_walk"
RAW_WALK = f"{WALK_MODULE}.bounded_walk"


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _direct_walk_uses(source: str, module: str) -> list[str]:
    """Every way ``source`` (the module ``module``) reaches `bounded_walk` directly.

    Import-aware: every local name bound by an import (absolute or relative, aliased or not) is
    resolved to what it names, so an aliased function import, a star import, a module or package
    alias followed by attribute access, and `getattr(module, "bounded_walk")` all resolve to the
    raw walk. Text mentions and unrelated local names do not.
    """
    tree = ast.parse(source)
    package = module.rpartition(".")[0]
    bindings: dict[str, str] = {}
    uses: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            for alias in node.names:
                if alias.name == "*":
                    if base == WALK_MODULE:
                        uses.append(f"line {node.lineno}: star import of {WALK_MODULE}")
                    continue
                target = f"{base}.{alias.name}"
                bindings[alias.asname or alias.name] = target
                if target == RAW_WALK:
                    uses.append(f"line {node.lineno}: imports {RAW_WALK}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bindings[alias.asname] = alias.name
                else:
                    head = alias.name.partition(".")[0]
                    bindings[head] = head

    def resolved(node: ast.expr) -> str | None:
        dotted = _dotted(node)
        if dotted is None:
            return None
        head, _, rest = dotted.partition(".")
        if head not in bindings:
            return None  # a local or builtin name, not something imported
        return f"{bindings[head]}.{rest}" if rest else bindings[head]

    for node in ast.walk(tree):
        # A bare name can only resolve to the raw walk through a function import, which is
        # already flagged above; module-qualified access is the remaining direct spelling.
        if isinstance(node, ast.Attribute) and resolved(node) == RAW_WALK:
            uses.append(f"line {node.lineno}: references {RAW_WALK}")
        if (
            isinstance(node, ast.Call) and _dotted(node.func) == "getattr"
            and len(node.args) >= 2 and resolved(node.args[0]) == WALK_MODULE
            and isinstance(node.args[1], ast.Constant) and node.args[1].value == "bounded_walk"
        ):
            uses.append(f"line {node.lineno}: getattr of {RAW_WALK}")
    return uses


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def test_every_production_consumer_walks_through_the_scoped_seam() -> None:
    direct = {}
    for path in sorted((REPO / "algua").rglob("*.py")):
        module = _module_name(path)
        if module == WALK_MODULE:
            continue  # the defining module is the only place the raw walk may be used
        uses = _direct_walk_uses(path.read_text(), module)
        if uses:
            direct[module] = uses
    assert direct == {}


DIRECT_WALK_SPELLINGS = {
    "direct-import": (
        "from algua.primitives.bounded_walk import bounded_walk\nbounded_walk(root)\n"),
    "aliased-import": (
        "from algua.primitives.bounded_walk import bounded_walk as bw\nbw(root)\n"),
    "aliased-import-unused-call": (
        "from algua.primitives.bounded_walk import bounded_walk as bw\n"),
    "star-import": "from algua.primitives.bounded_walk import *\n",
    "module-alias-call": (
        "from algua.primitives import bounded_walk as walks\nwalks.bounded_walk(root)\n"),
    "module-alias-reference": (
        "from algua.primitives import bounded_walk as w\nfn = w.bounded_walk\nfn(root)\n"),
    "import-as": "import algua.primitives.bounded_walk as w\nw.bounded_walk(root)\n",
    "fully-qualified": (
        "import algua.primitives.bounded_walk\n"
        "algua.primitives.bounded_walk.bounded_walk(root)\n"),
    "package-alias": "import algua.primitives as p\np.bounded_walk.bounded_walk(root)\n",
    "from-package": "from algua import primitives\nprimitives.bounded_walk.bounded_walk(root)\n",
    "relative-import": "from ..primitives.bounded_walk import bounded_walk as bw\nbw(root)\n",
    "relative-module": "from ..primitives import bounded_walk as w\nw.bounded_walk(root)\n",
    "getattr": (
        "import algua.primitives.bounded_walk as w\ngetattr(w, 'bounded_walk')(root)\n"),
}


@pytest.mark.parametrize(
    "source", DIRECT_WALK_SPELLINGS.values(), ids=DIRECT_WALK_SPELLINGS.keys())
def test_the_guard_catches_every_direct_walk_spelling(source: str) -> None:
    assert _direct_walk_uses(source, "algua.registry.consumer")


SCOPED_OR_UNRELATED = {
    "scoped-import": (
        "from algua.primitives.bounded_walk import WalkCleanupError, scoped_walk\n"
        "with scoped_walk(root, max_files=1, max_directories=1, max_path_bytes=1) as tree:\n"
        "    pass\n"),
    "module-scoped-call": (
        "from algua.primitives import bounded_walk as w\nw.scoped_walk(root)\n"),
    "mention-in-text": '"""Never call bounded_walk(root) directly."""\n# bounded_walk(root)\n',
    "unrelated-local-name": "def bounded_walk(root):\n    return root\nbounded_walk(1)\n",
}


@pytest.mark.parametrize(
    "source", SCOPED_OR_UNRELATED.values(), ids=SCOPED_OR_UNRELATED.keys())
def test_the_guard_allows_scoped_use_and_unrelated_mentions(source: str) -> None:
    assert _direct_walk_uses(source, "algua.registry.consumer") == []
