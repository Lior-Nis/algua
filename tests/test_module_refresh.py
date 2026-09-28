"""Warm package-closure refresh: current source only, acyclic, fresh globals, all-or-nothing."""
from __future__ import annotations

import importlib
import importlib.util
import os
import py_compile
import subprocess
import sys
import textwrap
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from algua.primitives import module_refresh, module_source_scan
from algua.primitives.module_refresh import ModuleRefreshError, refresh_package_closure
from algua.primitives.module_source_scan import require_acyclic, static_closure


@pytest.fixture
def family(tmp_path, monkeypatch):
    """A throwaway ``<top>.fam`` package on ``sys.path`` (nested, so the parent binding is real)."""
    top = f"mrtop_{uuid.uuid4().hex[:10]}"
    fam_dir = tmp_path / top / "fam"
    fam_dir.mkdir(parents=True)
    (tmp_path / top / "__init__.py").write_text("")
    (fam_dir / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))

    def write(module: str, source: str) -> Path:
        path = fam_dir / f"{module}.py"
        path.write_text(textwrap.dedent(source))
        return path

    def entries() -> dict[str, object]:
        return {k: v for k, v in sys.modules.items() if k == package or k.startswith(package + ".")}

    package = f"{top}.fam"
    yield SimpleNamespace(
        package=package, top=top, dir=fam_dir, write=write, entries=entries,
        mod=lambda name: sys.modules[f"{package}.{name}"],
    )
    for key in [k for k in sys.modules if k == top or k.startswith(top + ".")]:
        del sys.modules[key]


def _refresh(family) -> None:
    refresh_package_closure(family.package, f"{family.package}.strat")


def test_newly_imported_helper_bypasses_timestamp_valid_stale_bytecode(family) -> None:
    """A helper first imported BY the refresh must also compile from current source: its cached
    bytecode is purged before any member executes, not only the bytecode of loaded members."""
    family.write("strat", "VALUE = 'root'\n")
    importlib.import_module(f"{family.package}.strat")
    helper = family.write("helper", "VALUE = 'AAPL'\n")
    cached = Path(importlib.util.cache_from_source(str(helper)))
    py_compile.compile(
        str(helper), cfile=str(cached), invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    stamp = helper.stat()
    family.write("helper", "VALUE = 'MSFT'\n")
    os.utime(helper, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert helper.stat().st_size == stamp.st_size
    family.write("strat", "from .helper import VALUE\n")

    _refresh(family)

    assert family.mod("strat").VALUE == "MSFT"
    assert family.mod("helper").VALUE == "MSFT"


@pytest.mark.parametrize(
    "sources",
    [
        {"a": "from . import b\n", "b": "from . import a\n"},
        {"a": "from . import b\n", "b": "import {pkg}.c\n", "c": "from .a import VALUE\n"},
        {"a": "def lazy():\n    from . import b\n", "b": "from .a import lazy\n"},
        {"__init__": "from . import a\n", "a": "VALUE = 2\n"},
    ],
    ids=["two-cycle", "three-cycle", "lazy-import-cycle", "package-init-cycle"],
)
def test_cyclic_static_family_imports_fail_closed_with_no_effect(family, sources) -> None:
    family.write("a", "VALUE = 1\n")
    family.write("strat", "from .a import VALUE\n")
    importlib.import_module(f"{family.package}.strat")
    before = family.entries()
    parent_binding = sys.modules[family.top].fam

    for module, source in sources.items():
        family.write(module, source.replace("{pkg}", family.package))
    cached = Path(py_compile.compile(str(family.dir / "strat.py")))
    with pytest.raises(ModuleRefreshError, match="cycl") as caught:
        _refresh(family)

    assert isinstance(caught.value, ImportError)
    assert cached.is_file(), "a refused refresh must not purge or execute anything"
    assert family.entries() == before
    assert all(family.entries()[k] is v for k, v in before.items())
    assert sys.modules[family.top].fam is parent_binding
    assert family.mod("strat").VALUE == 1


def test_refresh_drops_globals_removed_from_current_source(family) -> None:
    family.write("helper", "VALUE = 1\nLEGACY = 2\n")
    family.write("strat", "from .helper import VALUE\nEXTRA = 3\n")
    importlib.import_module(f"{family.package}.strat")
    family.write("helper", "VALUE = 1\n")
    family.write("strat", "from .helper import VALUE\n")

    _refresh(family)

    assert not hasattr(family.mod("strat"), "EXTRA")
    assert not hasattr(family.mod("helper"), "LEGACY")


def test_a_name_removed_from_a_helper_is_not_silently_re_imported(family) -> None:
    family.write("helper", "LEGACY = ['AAPL']\n")
    family.write("strat", "from .helper import LEGACY\n")
    importlib.import_module(f"{family.package}.strat")
    family.write("helper", "CURRENT = ['MSFT']\n")

    with pytest.raises(ImportError, match="LEGACY"):
        _refresh(family)


@pytest.mark.parametrize(
    ("statement", "error"),
    [("raise RuntimeError('boom')", RuntimeError), ("raise KeyboardInterrupt", KeyboardInterrupt)],
)
def test_any_member_failure_rolls_back_the_complete_closure(family, statement, error) -> None:
    family.write("a_ok", "VALUE = 1\n")
    family.write("b_bad", "VALUE = 1\n")
    family.write("strat", "from . import a_ok, b_bad\nVALUE = a_ok.VALUE + b_bad.VALUE\n")
    importlib.import_module(f"{family.package}.strat")
    before = family.entries()
    parent_binding = sys.modules[family.top].fam
    family.write("a_ok", "VALUE = 2\n")
    family.write("b_bad", f"VALUE = 2\n{statement}\n")
    family.write("new_helper", "VALUE = 9\n")
    family.write("strat", "from . import a_ok, new_helper, b_bad\nVALUE = 0\n")

    with pytest.raises(error):
        _refresh(family)

    after = family.entries()
    assert set(after) == set(before)
    assert all(after[k] is v for k, v in before.items())
    assert sys.modules[family.top].fam is parent_binding
    assert (family.mod("a_ok").VALUE, family.mod("b_bad").VALUE) == (1, 1)
    assert family.mod("strat").VALUE == 2


def test_failed_first_refresh_leaves_no_family_module_or_parent_binding(family) -> None:
    importlib.import_module(family.top)
    family.write("strat", "raise RuntimeError('boom')\n")

    with pytest.raises(RuntimeError, match="boom"):
        _refresh(family)

    assert family.entries() == {}
    assert not hasattr(sys.modules[family.top], "fam")


def test_successful_refresh_rebinds_the_parent_to_the_fresh_closure(family) -> None:
    family.write("helper", "VALUE = 1\n")
    family.write("strat", "from . import helper\nVALUE = helper.VALUE\n")
    importlib.import_module(f"{family.package}.strat")
    old_package = sys.modules[family.package]
    family.write("helper", "VALUE = 2\n")

    _refresh(family)

    fresh = sys.modules[family.package]
    assert fresh is not old_package
    assert sys.modules[family.top].fam is fresh
    assert fresh.strat is family.mod("strat") and fresh.helper is family.mod("helper")
    assert family.mod("strat").VALUE == 2


def test_static_closure_resolves_absolute_relative_nested_and_parent_edges(family) -> None:
    pkg = family.package
    (family.dir / "sub").mkdir()
    (family.dir / "sub" / "__init__.py").write_text("")
    (family.dir / "sub" / "deep.py").write_text("VALUE = 1\n")
    for name in ("_a", "_b", "_c", "_d", "_e", "unreached"):
        family.write(name, "VALUE = 1\n")
    family.write("strat", (
        f"import {pkg}._a\n"
        "from . import _b\n"
        "from ._c import VALUE\n"
        f"from {pkg} import _d as alias\n"
        f"def lazy():\n    from {pkg}._e import VALUE\n"
        "from .sub import deep\n"
        "import os\n"
        f"from {pkg}.strat import self_reference\n"
        "from . import not_a_module\n"
    ))

    graph = static_closure(pkg, str(family.dir), f"{pkg}.strat")

    assert graph[f"{pkg}.strat"] == {
        pkg, f"{pkg}._a", f"{pkg}._b", f"{pkg}._c", f"{pkg}._d", f"{pkg}._e",
        f"{pkg}.sub", f"{pkg}.sub.deep",
    }
    assert graph[f"{pkg}.sub.deep"] == {f"{pkg}.sub"}
    assert graph[f"{pkg}.sub"] == {pkg}
    assert graph[pkg] == set()
    assert f"{pkg}.unreached" not in graph


def test_require_acyclic_names_the_cycle_deterministically() -> None:
    require_acyclic({"f": {"f._a"}, "f._a": {"f._b"}, "f._b": set()})
    graph = {"f": set(), "f._c": {"f._d"}, "f._d": {"f._e"}, "f._e": {"f._c"}}
    with pytest.raises(ModuleRefreshError) as first:
        require_acyclic(graph)
    with pytest.raises(ModuleRefreshError) as second:
        require_acyclic(dict(reversed(list(graph.items()))))
    assert str(first.value) == str(second.value)
    assert "f._c -> f._d -> f._e -> f._c" in str(first.value)


def test_require_acyclic_is_iterative_beyond_the_recursion_limit() -> None:
    depth = sys.getrecursionlimit() + 500
    graph: dict[str, set[str]] = {f"f._{i}": {f"f._{i + 1}"} for i in range(depth)}
    graph[f"f._{depth}"] = set()
    require_acyclic(graph)

    graph[f"f._{depth}"] = {"f._0"}
    with pytest.raises(ModuleRefreshError, match="cycl"):
        require_acyclic(graph)


def test_require_acyclic_expands_each_module_once_across_shared_dependencies() -> None:
    """Diamonds are acyclic (a finished module leaves the active path) and a finished module is
    never expanded again, so layered shared dependencies stay linear rather than exponential."""
    expanded: list[str] = []

    class Recording(dict[str, set[str]]):
        def __getitem__(self, name: str) -> set[str]:
            expanded.append(name)
            return super().__getitem__(name)

        def get(self, name: str, default: object = None) -> set[str]:  # noqa: ARG002
            expanded.append(name)
            return super().__getitem__(name)

    layers = [[f"f._{layer}_{i}" for i in range(2)] for layer in range(12)]
    graph = Recording({"f": set(layers[0])})
    for upper, lower in zip(layers, layers[1:], strict=False):
        graph.update({name: set(lower) for name in upper})
    graph.update({name: set() for name in layers[-1]})

    require_acyclic(graph)

    assert sorted(expanded) == sorted(graph)


_LAZY_CHAIN_DEPTH = 6  # fixed: depth beyond the recursion limit is proven on the in-memory graph


def test_refresh_follows_a_fixed_lazy_acyclic_helper_chain(family) -> None:
    """Integration over a small FIXED lazy-import chain. The beyond-recursion-limit depth proof is
    ``test_require_acyclic_is_iterative_beyond_the_recursion_limit`` (in memory), so the root gate
    never writes thousands of files or scales with a mutable interpreter recursion limit."""
    for i in range(_LAZY_CHAIN_DEPTH):
        family.write(f"h{i}", f"def follow():\n    from . import h{i + 1}\n    return h{i + 1}\n")
    family.write(f"h{_LAZY_CHAIN_DEPTH}", "VALUE = 1\n")
    family.write("strat", "from . import h0\nVALUE = 1\n")

    graph = static_closure(family.package, str(family.dir), f"{family.package}.strat")
    _refresh(family)

    chain = [f"{family.package}.h{i}" for i in range(_LAZY_CHAIN_DEPTH + 1)]
    assert all(graph[upper] >= {lower} for upper, lower in zip(chain, chain[1:], strict=False))
    assert family.mod("strat").VALUE == 1
    module = family.mod("h0")
    for _ in range(_LAZY_CHAIN_DEPTH):
        module = module.follow()
    assert module is family.mod(f"h{_LAZY_CHAIN_DEPTH}") and module.VALUE == 1
    assert len(list(family.dir.glob("*.py"))) == _LAZY_CHAIN_DEPTH + 3


def test_rollback_never_installs_a_binding_manufactured_by_parent_getattr(family) -> None:
    (family.dir.parent / "__init__.py").write_text(
        "def __getattr__(name):\n    return 'manufactured'\n")
    importlib.import_module(family.top)
    family.write("strat", "raise RuntimeError('boom')\n")

    with pytest.raises(RuntimeError, match="boom"):
        _refresh(family)

    assert family.entries() == {}
    assert "fam" not in vars(sys.modules[family.top])


@pytest.mark.parametrize(
    "case", ["helper-file", "subpackage", "package-directory", "package-ancestor"])
def test_symlinked_family_source_paths_fail_closed_before_any_purge(
        tmp_path, monkeypatch, case) -> None:
    """A symlinked path escapes the package walk (a linked subdirectory's timestamp-valid stale
    bytecode would execute as current source), so every search location and reachable source
    whose lexical path or existing ancestor is a symlink is refused before anything is purged."""
    top = f"mrlink_{uuid.uuid4().hex[:10]}"
    real, elsewhere = tmp_path / "real", tmp_path / "elsewhere"
    fam_dir = (elsewhere if case == "package-directory" else real / top) / "fam"
    fam_dir.mkdir(parents=True)
    elsewhere.mkdir(exist_ok=True)
    (real / top).mkdir(parents=True, exist_ok=True)
    (real / top / "__init__.py").write_text("")
    (fam_dir / "__init__.py").write_text("")
    (fam_dir / "strat.py").write_text("VALUE = 1\n")
    if case == "package-directory":
        (real / top / "fam").symlink_to(fam_dir, target_is_directory=True)
    elif case == "helper-file":
        (elsewhere / "helper.py").write_text("VALUE = 1\n")
        (fam_dir / "helper.py").symlink_to(elsewhere / "helper.py")
        (fam_dir / "strat.py").write_text("from .helper import VALUE\n")
    elif case == "subpackage":
        (elsewhere / "sub").mkdir()
        (elsewhere / "sub" / "__init__.py").write_text("")
        (elsewhere / "sub" / "deep.py").write_text("VALUE = 1\n")
        (fam_dir / "sub").symlink_to(elsewhere / "sub", target_is_directory=True)
        (fam_dir / "strat.py").write_text("from .sub.deep import VALUE\n")
    entry = real
    if case == "package-ancestor":
        entry = tmp_path / "link"
        entry.symlink_to(real, target_is_directory=True)
    monkeypatch.syspath_prepend(str(entry))
    package = f"{top}.fam"
    try:
        importlib.import_module(f"{package}.strat")
        before = {k: v for k, v in sys.modules.items() if k.startswith(package)}
        cached = Path(py_compile.compile(str(entry / top / "fam" / "strat.py")))

        with pytest.raises(ModuleRefreshError, match="symlink"):
            refresh_package_closure(package, f"{package}.strat")

        assert cached.is_file(), "a refused refresh must not purge anything"
        after = {k: v for k, v in sys.modules.items() if k.startswith(package)}
        assert set(after) == set(before) and all(after[k] is v for k, v in before.items())
    finally:
        for key in [k for k in sys.modules if k == top or k.startswith(top + ".")]:
            del sys.modules[key]


@pytest.mark.parametrize(
    "case", ["dynamic-subpackage", "dynamic-module", "unreached-nested-file", "dangling"])
def test_any_symlink_in_the_family_tree_fails_closed_before_any_purge(
        family, tmp_path, monkeypatch, case) -> None:
    """A dynamic ``importlib``/``__import__`` edge is invisible to the static closure, so the
    COMPLETE family tree is scanned without following links and any symlink entry is refused
    before discovery, purge or execution."""
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "dyn").mkdir(parents=True)
    (elsewhere / "dyn" / "__init__.py").write_text("")
    (elsewhere / "dyn" / "mod.py").write_text("VALUE = 1\n")
    (elsewhere / "other.py").write_text("VALUE = 1\n")
    family.write("strat", "VALUE = 1\n")
    importlib.import_module(f"{family.package}.strat")
    if case == "dynamic-subpackage":
        (family.dir / "dyn").symlink_to(elsewhere / "dyn", target_is_directory=True)
        family.write("strat", (
            "import importlib\nVALUE = importlib.import_module(__package__ + '.dyn.mod').VALUE\n"))
    elif case == "dynamic-module":
        (family.dir / "other.py").symlink_to(elsewhere / "other.py")
        family.write("strat", "VALUE = __import__(__package__ + '.other', fromlist=['_']).VALUE\n")
    elif case == "unreached-nested-file":
        (family.dir / "sub").mkdir()
        (family.dir / "sub" / "__init__.py").write_text("")
        (family.dir / "sub" / "deep.py").symlink_to(elsewhere / "other.py")
    else:
        (family.dir / "broken").symlink_to(tmp_path / "missing")
    before = family.entries()
    cached = Path(py_compile.compile(str(family.dir / "strat.py")))

    def discovery(*_args: object) -> None:
        raise AssertionError("source discovery ran before the family-tree symlink scan")

    monkeypatch.setattr(module_refresh, "static_closure", discovery)
    with pytest.raises(ModuleRefreshError, match="symlink"):
        _refresh(family)

    assert cached.is_file(), "a refused refresh must not purge anything"
    after = family.entries()
    assert set(after) == set(before) and all(after[k] is v for k, v in before.items())


def test_lexical_parent_traversal_is_refused_before_normalization_erases_a_symlink(
        tmp_path, monkeypatch) -> None:
    """``link/..`` resolves physically to the link TARGET's parent, but lexical normalization
    erases ``link`` entirely, so the symlink check would inspect an unrelated path. A raw ``..``
    component is refused before any normalization."""
    top = f"mrdots_{uuid.uuid4().hex[:10]}"
    (tmp_path / "outer" / "inner").mkdir(parents=True)
    fam_dir = tmp_path / "outer" / top / "fam"
    fam_dir.mkdir(parents=True)
    (fam_dir.parent / "__init__.py").write_text("")
    (fam_dir / "__init__.py").write_text("")
    (fam_dir / "strat.py").write_text("VALUE = 1\n")
    (tmp_path / "link").symlink_to(tmp_path / "outer" / "inner", target_is_directory=True)
    monkeypatch.syspath_prepend(str(tmp_path / "link") + os.sep + "..")
    package = f"{top}.fam"
    try:
        importlib.import_module(f"{package}.strat")
        cached = Path(py_compile.compile(str(fam_dir / "strat.py")))

        with pytest.raises(ModuleRefreshError, match="traversal"):
            refresh_package_closure(package, f"{package}.strat")

        assert cached.is_file(), "a refused refresh must not purge anything"
    finally:
        for key in [k for k in sys.modules if k == top or k.startswith(top + ".")]:
            del sys.modules[key]


@pytest.mark.parametrize("suffix", ["..", "../sub", "x/../.."])
def test_any_raw_parent_component_is_refused(tmp_path, suffix) -> None:
    (tmp_path / "sub" / "x").mkdir(parents=True)
    with pytest.raises(ModuleRefreshError, match="traversal"):
        module_source_scan.require_unlinked(str(tmp_path / "sub") + os.sep + suffix, "fam")


def test_a_symlinked_search_location_is_refused_before_source_discovery(
        family, tmp_path, monkeypatch) -> None:
    linked = tmp_path / "linked"
    linked.symlink_to(family.dir.parent.parent, target_is_directory=True)
    monkeypatch.syspath_prepend(str(linked))
    importlib.invalidate_caches()
    family.write("strat", "VALUE = 1\n")

    def discovery(*_args: object) -> None:
        raise AssertionError("source discovery ran before the search-location check")

    monkeypatch.setattr(module_refresh, "static_closure", discovery)
    with pytest.raises(ModuleRefreshError, match="symlink"):
        _refresh(family)


def test_failed_refresh_discards_new_external_modules_and_their_parent_bindings(family) -> None:
    """A module outside the family first imported by a failed refresh may hold the discarded
    fresh family objects; it is dropped along with the parent attribute that binds it."""
    for name in ("ext", "newext"):
        (family.dir.parent / name).mkdir()
        (family.dir.parent / name / "__init__.py").write_text("KEEP = 'kept'\n")
        (family.dir.parent / name / "held.py").write_text(
            f"from {family.package} import a_ok as HELD\n")
    ext = importlib.import_module(f"{family.top}.ext")
    family.write("a_ok", "VALUE = 1\n")
    family.write("strat", "from . import a_ok\n")
    importlib.import_module(f"{family.package}.strat")
    before = family.entries()
    family.write("b_bad", "raise RuntimeError('boom')\n")
    family.write("strat", (
        "from . import a_ok\n"
        f"import {family.top}.ext.held\n"
        f"import {family.top}.newext.held\n"
        "from . import b_bad\n"
    ))

    with pytest.raises(RuntimeError, match="boom"):
        _refresh(family)

    for name in ("ext.held", "newext", "newext.held"):
        assert f"{family.top}.{name}" not in sys.modules
    assert sys.modules[f"{family.top}.ext"] is ext
    assert "held" not in vars(ext) and ext.KEEP == "kept"
    assert "newext" not in vars(sys.modules[family.top])
    after = family.entries()
    assert vars(sys.modules[family.package])["a_ok"] is before[f"{family.package}.a_ok"], (
        "a restored parent binding to a previous object is not a discarded-module binding")
    assert set(after) == set(before) and all(after[k] is v for k, v in before.items())


@pytest.mark.parametrize("case", ["replaced-child-module", "pre-existing-value"])
def test_failed_refresh_restores_each_affected_parent_binding_exactly(family, case) -> None:
    """Rollback restores every affected module's direct parent binding to its exact prior state:
    a binding the attempt overwrote (a replaced child module, or a non-module value shadowed by a
    newly imported child) is reinstated, not deleted."""
    (family.dir.parent / "ext").mkdir()
    (family.dir.parent / "ext" / "__init__.py").write_text("")
    if case == "pre-existing-value":
        (family.dir.parent / "__init__.py").write_text("ext = 'pre-existing'\n")
        top = importlib.import_module(family.top)
        attempt = f"import {family.top}.ext\n"
    else:
        top = importlib.import_module(family.top)
        importlib.import_module(f"{family.top}.ext")
        attempt = f"import sys\nsys.modules.pop('{family.top}.ext')\nimport {family.top}.ext\n"
    previous = vars(top)["ext"]
    previous_entry = sys.modules.get(f"{family.top}.ext")
    family.write("strat", attempt + "raise RuntimeError('boom')\n")

    with pytest.raises(RuntimeError, match="boom"):
        _refresh(family)

    assert vars(top)["ext"] is previous
    assert sys.modules.get(f"{family.top}.ext") is previous_entry


@pytest.mark.parametrize("failure", ["cycle", "keyboard-interrupt"])
@pytest.mark.parametrize("warm_top", [False, True], ids=["cold-top", "warm-top"])
def test_failed_preflight_rolls_back_cold_parent_imports(
        tmp_path, monkeypatch, failure, warm_top) -> None:
    """Package discovery imports cold parents (``find_spec``), so module state is snapshotted
    BEFORE discovery and every preflight failure, including a ``BaseException``, restores it:
    no cold nested parent entry or parent binding is left behind."""
    top = f"mrcold_{uuid.uuid4().hex[:10]}"
    fam_dir = tmp_path / top / "mid" / "fam"
    fam_dir.mkdir(parents=True)
    for directory in (fam_dir.parent.parent, fam_dir.parent, fam_dir):
        (directory / "__init__.py").write_text("")
    (fam_dir / "a.py").write_text("from . import strat\n")
    (fam_dir / "strat.py").write_text("from . import a\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    if failure == "keyboard-interrupt":
        (fam_dir / "strat.py").write_text("VALUE = 1\n")

        def interrupted(*_args: object) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(module_refresh, "static_closure", interrupted)
    package = f"{top}.mid.fam"
    try:
        if warm_top:
            importlib.import_module(top)
        before = dict(sys.modules)

        with pytest.raises(ModuleRefreshError if failure == "cycle" else KeyboardInterrupt):
            refresh_package_closure(package, f"{package}.strat")

        assert f"{top}.mid" not in sys.modules and package not in sys.modules
        assert set(sys.modules) == set(before)
        if warm_top:
            assert "mid" not in vars(sys.modules[top])
        else:
            assert top not in sys.modules
    finally:
        for key in [k for k in sys.modules if k == top or k.startswith(top + ".")]:
            del sys.modules[key]


def test_refresh_returns_the_fresh_root_module(family) -> None:
    family.write("strat", "VALUE = 1\n")

    root = refresh_package_closure(family.package, f"{family.package}.strat")

    assert root is family.mod("strat")


def test_a_supported_serialized_import_waits_for_an_in_progress_refresh(family) -> None:
    """Supported loader callers share the refresh serialization: a serialized import of a family
    member mid-refresh waits for the final state instead of importing into a partially rebuilt
    family that may still be discarded."""
    gate = f"{family.top}.gate"
    (family.dir.parent / "gate.py").write_text(
        "import threading\nentered = threading.Event()\nrelease = threading.Event()\n")
    signal = importlib.import_module(gate)
    family.write("other", "VALUE = 'other'\n")
    family.write("strat", f"import {gate} as _g\n_g.entered.set()\nassert _g.release.wait(10)\n")
    outcome: dict[str, object] = {}
    imported = threading.Event()

    def refresh() -> None:
        try:
            outcome["root"] = refresh_package_closure(family.package, f"{family.package}.strat")
        except BaseException as exc:  # surfaced by the assertion below
            outcome["refresh_error"] = exc

    def compete() -> None:
        outcome["other"] = module_refresh.serialized_import(f"{family.package}.other")
        imported.set()

    refresher = threading.Thread(target=refresh)
    competitor = threading.Thread(target=compete)
    refresher.start()
    try:
        assert signal.entered.wait(10)
        competitor.start()
        entered_mid_refresh = imported.wait(0.5)
    finally:
        signal.release.set()
        refresher.join(10)
        competitor.join(10)

    assert not entered_mid_refresh, "a serialized import entered the refresh transaction"
    assert "refresh_error" not in outcome
    assert outcome["root"] is family.mod("strat")
    assert outcome["other"] is family.mod("other") is sys.modules[family.package].other


_DEADLOCK_PROBE = """
import importlib, os, sys, threading
sys.path.insert(0, sys.argv[1])
from algua.primitives.module_refresh import refresh_package_closure
import gate_probe
errors = []
def run(target):
    try:
        target()
    except BaseException as exc:
        errors.append(exc)
importer = threading.Thread(
    target=run, args=(lambda: importlib.import_module("slow_probe"),), daemon=True)
importer.start()
assert gate_probe.entered.wait(10)
refresher = threading.Thread(
    target=run, args=(lambda: refresh_package_closure("top_probe.fam", "top_probe.fam.strat"),),
    daemon=True)
refresher.start()
# Handshake: the refresher sets at_import inside the refresh transaction, immediately before its
# contested import of slow_probe (still being initialized by the importer), and only then is the
# importer released to need its own new import.
assert gate_probe.at_import.wait(10)
gate_probe.go.set()
importer.join(10)
refresher.join(10)
alive = importer.is_alive() or refresher.is_alive()
print("deadlock" if alive else f"done errors={errors!r}", flush=True)
os._exit(3 if alive else (1 if errors else 0))
"""


def test_refresh_does_not_deadlock_against_an_import_already_in_progress(tmp_path) -> None:
    """The refresh must not hold the process-global import lock: a thread already initializing a
    module the refresh imports, that itself needs a NEW import, would wait for the global lock
    while the refresh waits for that module's lock. Run in a child process so a deadlock cannot
    wedge the test runner's own import system."""
    (tmp_path / "top_probe" / "fam").mkdir(parents=True)
    (tmp_path / "top_probe" / "__init__.py").write_text("")
    (tmp_path / "top_probe" / "fam" / "__init__.py").write_text("")
    (tmp_path / "top_probe" / "fam" / "strat.py").write_text(
        "import gate_probe\ngate_probe.at_import.set()\nimport slow_probe\nVALUE = 1\n")
    (tmp_path / "gate_probe.py").write_text(
        "import threading\nentered = threading.Event()\nat_import = threading.Event()\n"
        "go = threading.Event()\n")
    (tmp_path / "slow_probe.py").write_text(
        "import gate_probe\ngate_probe.entered.set()\nassert gate_probe.go.wait(10)\n"
        "import late_probe\n")
    (tmp_path / "late_probe.py").write_text("VALUE = 1\n")

    result = subprocess.run(
        [sys.executable, "-c", _DEADLOCK_PROBE, str(tmp_path)],
        capture_output=True, text=True, timeout=120, check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)


@pytest.mark.parametrize("case", ["absent-before", "existing-before"])
def test_failed_refresh_restores_parent_bindings_of_self_removed_transient_imports(
        family, case) -> None:
    """Attempted code may import a module (binding it on its parent) and then remove its own
    ``sys.modules`` entry before failing, so the entry never differs from the snapshot. The
    parent binding the import changed is still restored to its prior value or absence."""
    (family.dir.parent / "ext").mkdir()
    (family.dir.parent / "ext" / "__init__.py").write_text("")
    if case == "existing-before":
        (family.dir.parent / "__init__.py").write_text("ext = 'pre-existing'\n")
    top = importlib.import_module(family.top)
    transient = f"import {family.top}.ext\nsys.modules.pop('{family.top}.ext')\n"
    family.write("strat", "import sys\n" + transient * 2 + "raise RuntimeError('boom')\n")

    with pytest.raises(RuntimeError, match="boom"):
        _refresh(family)

    assert f"{family.top}.ext" not in sys.modules
    if case == "existing-before":
        assert vars(top)["ext"] == "pre-existing"
    else:
        assert "ext" not in vars(top)


@pytest.mark.parametrize(
    "case", ["sourceless-module", "extension-module", "bytecode-only-package", "pycache-module"])
def test_importable_non_source_family_entries_fail_closed_before_any_purge(
        family, monkeypatch, case) -> None:
    """Sourceless bytecode or an extension module anywhere in the family tree is importable by a
    dynamic edge the static closure cannot see, yet it is neither validated source nor purged, so
    the tree scan refuses it before discovery, purge or execution."""
    family.write("strat", "VALUE = 1\n")
    importlib.import_module(f"{family.package}.strat")
    source = family.dir.parent / "compiled_source.py"
    source.write_text("VALUE = 'stale'\n")
    if case == "sourceless-module":
        py_compile.compile(str(source), cfile=str(family.dir / "dyn.pyc"))
        family.write("strat", (
            "import importlib\nVALUE = importlib.import_module(__package__ + '.dyn').VALUE\n"))
    elif case == "extension-module":
        (family.dir / f"ext{importlib.machinery.EXTENSION_SUFFIXES[0]}").write_bytes(b"\0")
    elif case == "bytecode-only-package":
        (family.dir / "sub").mkdir()
        py_compile.compile(str(source), cfile=str(family.dir / "sub" / "__init__.pyc"))
    else:
        (family.dir / "__pycache__").mkdir(exist_ok=True)
        py_compile.compile(str(source), cfile=str(family.dir / "__pycache__" / "evil.pyc"))
    before = family.entries()
    cached = Path(py_compile.compile(str(family.dir / "strat.py")))

    def discovery(*_args: object) -> None:
        raise AssertionError("source discovery ran before the family-tree entry scan")

    monkeypatch.setattr(module_refresh, "static_closure", discovery)
    with pytest.raises(ModuleRefreshError, match="non-source"):
        _refresh(family)

    assert cached.is_file(), "a refused refresh must not purge anything"
    after = family.entries()
    assert set(after) == set(before) and all(after[k] is v for k, v in before.items())


def test_generated_bytecode_caches_and_data_files_do_not_block_a_refresh(family) -> None:
    """Normal ``__pycache__`` entries (tagged, so not importable by name) and non-importable data
    files stay admissible: the source-only scope refuses only importable non-source entries."""
    family.write("helper", "VALUE = 1\n")
    family.write("strat", "from .helper import VALUE\n")
    py_compile.compile(str(family.dir / "helper.py"), optimize=1)
    py_compile.compile(str(family.dir / "strat.py"))
    (family.dir / "notes.json").write_text("{}")
    importlib.import_module(f"{family.package}.strat")
    family.write("helper", "VALUE = 2\n")

    _refresh(family)

    assert family.mod("strat").VALUE == 2


@pytest.mark.parametrize(
    "case", ["package-appends", "package-prepends-shadow", "subpackage-appends",
             "module-becomes-package", "lazy-import-after-refresh"])
def test_fresh_search_paths_cannot_reach_beyond_the_prevalidated_family_roots(
        family, tmp_path, case) -> None:
    """A fresh package ``__init__`` (or any member) that extends ``__path__`` would let a child
    import load unscanned code, so children resolve only from the exact prevalidated roots and a
    refresh whose fresh search paths deviate from them fails closed, before any outside code runs
    and before a later lazy import could use the extended path."""
    outside, marker = tmp_path / "outside", tmp_path / "executed"
    outside.mkdir()
    for name in ("helper", "outside_helper"):
        (outside / f"{name}.py").write_text(f"open({str(marker)!r}, 'w').close()\nVALUE = 'out'\n")
    family.write("helper", "VALUE = 'in'\n")
    family.write("strat", "VALUE = 1\n")
    importlib.import_module(f"{family.package}.strat")
    before = family.entries()
    extend = f"__path__.append({str(outside)!r})\n"
    if case == "package-appends":
        (family.dir / "__init__.py").write_text(extend)
        family.write("strat", "from . import outside_helper\n")
    elif case == "package-prepends-shadow":
        (family.dir / "__init__.py").write_text(
            f"import importlib\n__path__.insert(0, {str(outside)!r})\n"
            "importlib.import_module(__name__ + '.helper')\n")
    elif case == "subpackage-appends":
        (family.dir / "sub").mkdir()
        (family.dir / "sub" / "__init__.py").write_text(extend)
        family.write("strat", "from .sub import outside_helper\n")
    elif case == "module-becomes-package":
        family.write("helper", f"__path__ = [{str(outside)!r}]\n")
        family.write("strat", "from .helper import outside_helper\n")
    else:
        (family.dir / "sub").mkdir()
        (family.dir / "sub" / "__init__.py").write_text(extend)
        family.write("strat", (
            "from . import sub\ndef later():\n    from .sub import outside_helper\n"))

    with pytest.raises(ModuleRefreshError, match="search path"):
        _refresh(family)

    assert not marker.exists(), "code outside the prevalidated family roots executed"
    after = family.entries()
    assert set(after) == set(before) and all(after[k] is v for k, v in before.items())


@pytest.mark.parametrize("case", ["later-finder", "namespace-directory"])
def test_family_modules_resolve_only_as_source_from_the_prevalidated_root(
        family, tmp_path, monkeypatch, case) -> None:
    """During the transaction no other finder may supply a family module the scanned root lacks,
    and a dynamically imported family entry must be Python source (namespace directories stay
    outside the source-only scope, as for the static closure)."""
    marker = tmp_path / "executed"
    ghost = tmp_path / "ghost.py"
    ghost.write_text(f"open({str(marker)!r}, 'w').close()\n")
    family.write("strat", "VALUE = 1\n")
    importlib.import_module(f"{family.package}.strat")
    before = family.entries()
    if case == "later-finder":
        class Elsewhere:
            @staticmethod
            def find_spec(fullname: str, path: object = None, target: object = None):
                if fullname != f"{family.package}.ghost":
                    return None
                return importlib.util.spec_from_file_location(fullname, ghost)

        monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, Elsewhere()])
        family.write("strat", "from . import ghost\n")
        expected, match = ImportError, "ghost"
    else:
        (family.dir / "nsdir").mkdir()
        (family.dir / "nsdir" / "mod.py").write_text("VALUE = 1\n")
        family.write("strat", (
            "import importlib\n"
            "VALUE = importlib.import_module(__package__ + '.nsdir.mod').VALUE\n"))
        expected, match = ModuleRefreshError, "not a Python source"

    with pytest.raises(expected, match=match):
        _refresh(family)

    assert not marker.exists()
    after = family.entries()
    assert set(after) == set(before) and all(after[k] is v for k, v in before.items())


def test_a_nested_refresh_on_the_same_thread_is_refused(family) -> None:
    """A refresh started from a module body the outer refresh is executing would drop and rebuild
    the family underneath the outer transaction, so the re-entrant lock must not admit it."""
    family.write("helper", "VALUE = 1\n")
    family.write("strat", "from .helper import VALUE\n")
    importlib.import_module(f"{family.package}.strat")
    before = family.entries()
    family.write("other", "VALUE = 2\n")
    family.write("strat", (
        "from algua.primitives.module_refresh import refresh_package_closure\n"
        f"refresh_package_closure({family.package!r}, {family.package + '.other'!r})\n"))

    with pytest.raises(ModuleRefreshError, match="nested"):
        _refresh(family)

    after = family.entries()
    assert set(after) == set(before) and all(after[k] is v for k, v in before.items())


def test_a_serialized_import_inside_a_refresh_on_the_same_thread_proceeds(family) -> None:
    family.write("other", "VALUE = 2\n")
    family.write("strat", (
        "from algua.primitives.module_refresh import serialized_import\n"
        f"VALUE = serialized_import({family.package + '.other'!r}).VALUE\n"))

    root = refresh_package_closure(family.package, f"{family.package}.strat")

    assert root.VALUE == 2 and family.mod("other").VALUE == 2


_FORK_PROBE = """
import os, threading
from algua.primitives import module_refresh
held, release = threading.Event(), threading.Event()
def owner():
    with module_refresh._REFRESH_LOCK:
        held.set()
        release.wait(60)
threading.Thread(target=owner, daemon=True).start()
assert held.wait(10)
pid = os.fork()
if pid == 0:
    acquired = module_refresh._REFRESH_LOCK.acquire(timeout=5)
    if acquired:
        module_refresh._REFRESH_LOCK.release()
        module_refresh.serialized_import("json")
    os._exit(0 if acquired else 7)
_, status = os.waitpid(pid, 0)
release.set()
print(os.waitstatus_to_exitcode(status), flush=True)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork-only platform behavior")
def test_a_forked_child_does_not_inherit_a_vanished_refresh_lock_owner() -> None:
    """A fork while another thread holds the refresh lock leaves the child a lock whose owner does
    not exist there; the child reinitializes it, so later strategy loads do not block forever.
    Run in a child process so the fork never touches the test runner's own threads."""
    result = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", _FORK_PROBE],
        capture_output=True, text=True, timeout=120, check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.strip() == "0", "the forked child could not acquire the refresh lock"


_HOST_PATH = "/host/secret/location"


def _failing_inspection(monkeypatch, case: str, root: Path) -> None:
    """Make family-tree inspection under ``root`` raise an ``OSError`` naming a host path."""
    real_scandir, real_is_symlink = os.scandir, Path.is_symlink

    def denied() -> OSError:
        return PermissionError(13, "Permission denied", _HOST_PATH)

    def within_root(path: object) -> bool:
        return str(path).startswith(str(root))

    if case == "scandir":
        def scandir(path: object = "."):  # type: ignore[no-untyped-def]
            if within_root(path):
                raise denied()
            return real_scandir(path)  # type: ignore[arg-type]
        monkeypatch.setattr(os, "scandir", scandir)
    elif case == "dir-entry":
        class Entry:
            def __init__(self, entry: os.DirEntry[str]) -> None:
                self.name, self.path = entry.name, entry.path

            def is_symlink(self) -> bool:
                raise denied()

        class Entries:
            def __init__(self, path: object) -> None:
                self._inner = real_scandir(path)  # type: ignore[arg-type]

            def __enter__(self):  # type: ignore[no-untyped-def]
                return (Entry(entry) for entry in self._inner)

            def __exit__(self, *exc: object) -> None:
                self._inner.close()

        monkeypatch.setattr(
            os, "scandir",
            lambda path=".": Entries(path) if within_root(path) else real_scandir(path))
    else:
        def is_symlink(self: Path) -> bool:
            if within_root(self):
                raise denied()
            return real_is_symlink(self)
        monkeypatch.setattr(Path, "is_symlink", is_symlink)


@pytest.mark.parametrize("case", ["scandir", "dir-entry", "path-lstat"])
def test_family_tree_inspection_os_errors_fail_closed_with_bounded_diagnostics(
        family, monkeypatch, case) -> None:
    family.write("strat", "VALUE = 1\n")
    importlib.import_module(f"{family.package}.strat")
    before = family.entries()
    _failing_inspection(monkeypatch, case, family.dir)

    with pytest.raises(ModuleRefreshError) as caught:
        _refresh(family)

    message = str(caught.value)
    assert "PermissionError" in message and len(message) <= 200
    assert _HOST_PATH not in message and str(family.dir.parent.parent) not in message
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    after = family.entries()
    assert set(after) == set(before) and all(after[k] is v for k, v in before.items())


def test_a_strategy_whose_family_tree_cannot_be_inspected_is_not_found(monkeypatch) -> None:
    from algua.strategies.loader import StrategyNotFound, _index, load_strategy_config

    dotted = _index()["cross_sectional_momentum"]
    family_dir = Path(importlib.util.find_spec(dotted.rsplit(".", 1)[0]).origin).parent
    _failing_inspection(monkeypatch, "scandir", family_dir)

    with pytest.raises(StrategyNotFound) as caught:
        load_strategy_config("cross_sectional_momentum")

    assert "PermissionError" in str(caught.value) and _HOST_PATH not in str(caught.value)


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_refresh_never_copies_unrelated_module_namespaces(family, monkeypatch, outcome) -> None:
    """Rollback state is ``sys.modules`` plus the direct parent bindings imports change, so the
    refresh cost does not scale with every global of every loaded (scientific) module."""
    from types import ModuleType

    class Unrelated(ModuleType):
        @property
        def __dict__(self):  # type: ignore[override]
            raise AssertionError("the refresh read an unrelated module's namespace")

    name = f"mrunrelated_{uuid.uuid4().hex[:10]}"
    monkeypatch.setitem(sys.modules, name, Unrelated(name))
    family.write("strat", "VALUE = 1\n" if outcome == "success" else "raise RuntimeError('boom')\n")

    if outcome == "success":
        assert refresh_package_closure(family.package, f"{family.package}.strat").VALUE == 1
    else:
        with pytest.raises(RuntimeError, match="boom"):
            _refresh(family)


class _Redirecting:
    """A poisoned path-entry finder: it hands out an OUTSIDE source spec for ``target`` and
    delegates every other name to a genuine source finder for the same directory."""

    def __init__(self, directory: str, target: str, outside: Path) -> None:
        self._real = importlib.machinery.FileFinder(
            directory, (importlib.machinery.SourceFileLoader, [".py"]))
        self._target, self._outside = target, outside

    def find_spec(self, fullname: str, target: object = None):  # type: ignore[no-untyped-def]
        if fullname == self._target:
            return importlib.util.spec_from_file_location(fullname, self._outside)
        return self._real.find_spec(fullname, target)  # type: ignore[arg-type]

    def invalidate_caches(self) -> None:
        self._real.invalidate_caches()


@pytest.mark.parametrize("case", ["path-importer-cache", "path-hook", "stale-package-spec"])
def test_family_specs_resolve_only_to_the_exact_in_root_source(
        family, tmp_path, monkeypatch, case) -> None:
    """Preflight and execution must both use the exact deterministic in-root source: neither a
    poisoned path-entry finder (cached or produced by a path hook) nor a stale loaded family
    ``__spec__`` may redirect a family module, so no outside code is inspected or executed."""
    marker = tmp_path / "executed"
    outside = tmp_path / "outside" / "fam"
    outside.mkdir(parents=True)
    body = f"open({str(marker)!r}, 'w').close()\nVALUE = 'out'\n"
    for name in ("__init__", "helper", "strat"):
        (outside / f"{name}.py").write_text(body)
    family.write("helper", "VALUE = 'in'\n")
    family.write("strat", "from .helper import VALUE\n")
    warm = importlib.import_module(f"{family.package}.strat")
    helper = f"{family.package}.helper"
    directories = {str(family.dir), *sys.modules[family.package].__path__}
    if case == "path-importer-cache":
        for directory in directories:
            monkeypatch.setitem(
                sys.path_importer_cache, directory,
                _Redirecting(directory, helper, outside / "helper.py"))
    elif case == "path-hook":
        def hook(directory: str) -> _Redirecting:
            if directory not in directories:
                raise ImportError(directory)
            return _Redirecting(directory, helper, outside / "helper.py")

        monkeypatch.setattr(sys, "path_hooks", [hook, *sys.path_hooks])
        for directory in directories:
            monkeypatch.delitem(sys.path_importer_cache, directory, raising=False)
    else:
        monkeypatch.setattr(sys.modules[family.package], "__spec__", (
            importlib.util.spec_from_file_location(
                family.package, outside / "__init__.py",
                submodule_search_locations=[str(outside)])))

    root = refresh_package_closure(family.package, f"{family.package}.strat")

    assert not marker.exists(), "code outside the exact family root executed"
    assert root is not warm and root.VALUE == "in"
    assert family.mod("helper").__file__ == str(family.dir / "helper.py")


@pytest.mark.parametrize("shape", ["namespace-directory", "plain-module", "absent"])
def test_a_family_that_is_not_a_regular_source_package_is_refused(
        tmp_path, monkeypatch, shape) -> None:
    top = f"mrshape_{uuid.uuid4().hex[:10]}"
    (tmp_path / top).mkdir()
    (tmp_path / top / "__init__.py").write_text("")
    if shape == "namespace-directory":
        (tmp_path / top / "fam").mkdir()
        (tmp_path / top / "fam" / "strat.py").write_text("VALUE = 1\n")
    elif shape == "plain-module":
        (tmp_path / top / "fam.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        with pytest.raises(ModuleRefreshError):
            refresh_package_closure(f"{top}.fam", f"{top}.fam.strat")
        assert f"{top}.fam" not in sys.modules and top not in sys.modules
    finally:
        for key in [k for k in sys.modules if k == top or k.startswith(top + ".")]:
            del sys.modules[key]
