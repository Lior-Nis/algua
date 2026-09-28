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
from types import ModuleType, SimpleNamespace

import pytest

from algua.primitives import module_commit_check, module_refresh, module_source_scan
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
        with pytest.raises(ModuleRefreshError, match="not a regular package"):
            refresh_package_closure(f"{top}.fam", f"{top}.fam.strat")
        assert f"{top}.fam" not in sys.modules and top not in sys.modules
    finally:
        for key in [k for k in sys.modules if k == top or k.startswith(top + ".")]:
            del sys.modules[key]


@pytest.fixture
def special_node(monkeypatch):
    """Create a FIFO or UNIX socket at a path. A FIFO gets a writer thread that is released at
    teardown, so any code that wrongly opens it for reading sees EOF instead of hanging."""
    import socket

    created: list[object] = []

    def make(path: Path, kind: str) -> None:
        if kind == "fifo":
            os.mkfifo(path)

            def feed() -> None:
                with open(path, "wb"):  # blocks until a reader opens, then closes: EOF
                    pass

            writer = threading.Thread(target=feed, daemon=True)
            writer.start()
            created.append((path, writer))
        else:
            monkeypatch.chdir(path.parent)  # a relative bind keeps the socket path short
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(path.name)
            created.append(server)

    yield make
    for item in created:
        if isinstance(item, tuple):
            path, writer = item
            os.close(os.open(path, os.O_RDONLY | os.O_NONBLOCK))
            writer.join(10)
        else:
            item.close()  # type: ignore[attr-defined]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special files")
@pytest.mark.parametrize(
    ("kind", "where"),
    [("fifo", "dynamic-module"), ("socket", "dynamic-module"), ("fifo", "subpackage-init"),
     ("fifo", "unreached-data")])
def test_non_regular_family_tree_nodes_fail_closed_before_any_read(
        family, monkeypatch, special_node, kind, where) -> None:
    """A FIFO, socket or device named like source is importable by a dynamic edge, yet reading it
    can block or yield unsupported content, so the family-tree scan refuses every non-regular,
    non-directory node before discovery, purge, read or execution."""
    family.write("strat", "VALUE = 1\n")
    importlib.import_module(f"{family.package}.strat")
    if where == "dynamic-module":
        special_node(family.dir / "dyn.py", kind)
        family.write("strat", (
            "import importlib\nVALUE = importlib.import_module(__package__ + '.dyn').VALUE\n"))
    elif where == "subpackage-init":
        (family.dir / "sub").mkdir()
        special_node(family.dir / "sub" / "__init__.py", kind)
    else:
        special_node(family.dir / "data", kind)
    before = family.entries()
    cached = Path(py_compile.compile(str(family.dir / "strat.py")))

    def discovery(*_args: object) -> None:
        raise AssertionError("source discovery ran before the family-tree node scan")

    monkeypatch.setattr(module_refresh, "static_closure", discovery)
    with pytest.raises(ModuleRefreshError, match="non-regular"):
        _refresh(family)

    assert cached.is_file(), "a refused refresh must not purge anything"
    after = family.entries()
    assert set(after) == set(before) and all(after[k] is v for k, v in before.items())


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special files")
def test_a_family_spec_never_resolves_to_a_non_regular_source_node(family, special_node) -> None:
    """Defense in depth behind the tree scan: exact spec construction itself refuses a source node
    that is not a regular file instead of treating it as absent or opening it."""
    special_node(family.dir / "dyn.py", "fifo")

    with pytest.raises(ModuleRefreshError, match="non-regular"):
        module_source_scan.source_spec(family.package, str(family.dir), f"{family.package}.dyn")


@pytest.mark.parametrize(
    ("case", "error"),
    [("syntax-error", "SyntaxError"), ("null-byte", "SyntaxError"),
     ("undecodable", "SyntaxError"), ("read-denied", "PermissionError"),
     ("stat-denied", "PermissionError")])
def test_source_stat_read_and_parse_failures_fail_closed_with_bounded_diagnostics(
        family, monkeypatch, case, error) -> None:
    """Preflight reads and parses every reachable source; a failure there is the stable bounded
    refusal naming only the module, the error class and a line or errno, never a host path, and
    the raw exception is not chained."""
    family.write("helper", "VALUE = 1\n")
    family.write("strat", "from .helper import VALUE\n")
    importlib.import_module(f"{family.package}.strat")
    before = family.entries()
    helper = family.dir / "helper.py"
    if case == "syntax-error":
        helper.write_text("def broken(:\n")
    elif case == "null-byte":
        helper.write_bytes(b"VALUE = 1\0\n")
    elif case == "undecodable":
        helper.write_bytes(b"VALUE = '\xff'\n")
    elif case == "read-denied":
        if os.geteuid() == 0:
            pytest.skip("root bypasses file permissions")
        helper.chmod(0)
    else:
        real_stat = os.stat

        def stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
            if os.fspath(path) == str(helper):  # type: ignore[arg-type]
                raise PermissionError(13, "Permission denied", _HOST_PATH)
            return real_stat(path, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(os, "stat", stat)

    try:
        with pytest.raises(ModuleRefreshError) as caught:
            _refresh(family)
    finally:
        helper.chmod(0o644)

    message = str(caught.value)
    assert error in message and len(message) <= 200
    assert _HOST_PATH not in message and str(family.dir.parent.parent) not in message
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    after = family.entries()
    assert set(after) == set(before) and all(after[k] is v for k, v in before.items())


def test_a_bytecode_purge_failure_fails_closed_without_touching_the_module_graph(family) -> None:
    """The purge walks the family tree top-down; here it deletes the top-level cache and then
    fails on a nested cache entry that cannot be unlinked. The partial deletion only removes
    caches: the refusal is the stable bounded error, nothing commits, and every module entry and
    parent binding is left exactly as it was."""
    (family.dir / "sub").mkdir()
    (family.dir / "sub" / "__init__.py").write_text("")
    (family.dir / "sub" / "deep.py").write_text("VALUE = 1\n")
    family.write("helper", "VALUE = 1\n")
    family.write("strat", "from .helper import VALUE\nfrom .sub import deep\n")
    importlib.import_module(f"{family.package}.strat")
    before = family.entries()
    parent_binding = vars(sys.modules[family.top])["fam"]
    helper_cache = Path(py_compile.compile(str(family.dir / "helper.py")))
    blocked = Path(importlib.util.cache_from_source(str(family.dir / "sub" / "deep.py")))
    blocked.unlink(missing_ok=True)
    blocked.mkdir(parents=True)
    (blocked / "keep").write_text("")
    family.write("helper", "VALUE = 2\n")

    with pytest.raises(ModuleRefreshError) as caught:
        _refresh(family)

    message = str(caught.value)
    assert "IsADirectoryError" in message and len(message) <= 200
    assert str(family.dir.parent.parent) not in message
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert not helper_cache.exists(), "the purge must have deleted a cache before failing"
    after = family.entries()
    assert set(after) == set(before) and all(after[k] is v for k, v in before.items())
    assert vars(sys.modules[family.top])["fam"] is parent_binding
    assert family.mod("strat").VALUE == 1


def test_a_purge_walk_that_cannot_enter_a_directory_is_refused_not_skipped(
        family, monkeypatch) -> None:
    """``os.walk`` silently skips a directory it cannot list, which would leave its stale caches
    in place; the purge surfaces that as the bounded refusal instead (the tree scan is bypassed
    here to reach the purge, as a directory can become unreadable after the scan)."""
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory permissions")
    (family.dir / "sub").mkdir()
    (family.dir / "sub" / "deep.py").write_text("VALUE = 1\n")
    family.write("strat", "VALUE = 1\n")
    monkeypatch.setattr(module_refresh, "require_source_tree", lambda *_args: None)
    (family.dir / "sub").chmod(0)
    try:
        with pytest.raises(ModuleRefreshError, match=r"bytecode.*PermissionError"):
            _refresh(family)
    finally:
        (family.dir / "sub").chmod(0o755)
    assert family.entries() == {}


_UNCOMMITTABLE = {
    "non-module-entry": (
        "import sys, types\nfrom . import helper\n"
        "sys.modules[__package__ + '.helper'] = types.SimpleNamespace(VALUE=1)\n"),
    "root-replaces-itself": (
        "import sys, types\nsys.modules[__name__] = types.SimpleNamespace(VALUE=1)\n"),
    "foreign-entry": (
        "import sys, types\n"
        "sys.modules[__package__ + '.fake'] = types.ModuleType(__package__ + '.fake')\n"),
    "self-removed-entry": (
        "import sys\nfrom . import helper\ndel sys.modules[__package__ + '.helper']\n"),
    "rebound-on-parent": (
        "import sys\nfrom . import helper\nsys.modules[__package__].helper = 'shadow'\n"),
    "package-unbound-from-parent": (
        "import sys\ntop, _, leaf = __package__.rpartition('.')\n"
        "vars(sys.modules[top]).pop(leaf)\n"),
    "parent-not-a-module": (
        "import sys, types\ntop, _, leaf = __package__.rpartition('.')\n"
        "sys.modules[top] = types.SimpleNamespace(**{leaf: sys.modules[__package__]})\n"),
    "foreign-spec": (
        "import importlib.util\nfrom . import helper\n"
        "helper.__spec__ = importlib.util.find_spec('json')\n"),
    "module-gains-a-path": (
        "import os\nfrom . import helper\n"
        "helper.__path__ = [os.path.join(os.path.dirname(__file__), 'helper')]\n"),
}


@pytest.mark.parametrize("case", sorted(_UNCOMMITTABLE))
def test_a_fresh_family_graph_that_is_not_exactly_bound_never_commits(family, case) -> None:
    """At commit every fresh family entry must be a source-backed ``ModuleType`` the guard
    resolved from the exact root, bound identically in ``sys.modules`` and on its direct parent,
    with a package keeping exactly its confined ``__path__`` and a module having none; anything
    else rolls the complete transaction back."""
    _assert_never_commits(family, _UNCOMMITTABLE[case])


def _assert_never_commits(family, source: str) -> None:
    """Refreshing into ``source`` fails closed and restores every prior family entry and binding."""
    (family.dir / "helper").mkdir()
    (family.dir / "helper" / "inner.py").write_text("VALUE = 1\n")
    family.write("helper", "VALUE = 1\n")
    family.write("strat", "from . import helper\nVALUE = 1\n")
    importlib.import_module(f"{family.package}.strat")
    before = family.entries()
    parent_binding = vars(sys.modules[family.top])["fam"]
    family.write("strat", source)

    with pytest.raises(ModuleRefreshError):
        _refresh(family)

    after = family.entries()
    assert set(after) == set(before) and all(after[k] is v for k, v in before.items())
    assert vars(sys.modules[family.top])["fam"] is parent_binding
    assert vars(parent_binding)["helper"] is before[f"{family.package}.helper"]


_MID_REFRESH_FORK_PROBE = """
import os, sys, threading, traceback
root = sys.argv[1]
sys.path.insert(0, root)
from algua.primitives import module_refresh
import gate_fork
import top_fork.fam.strat
top = sys.modules["top_fork"]
old = {k: v for k, v in sys.modules.items() if k.startswith("top_fork.fam")}
old_binding = vars(top)["fam"]
with open(os.path.join(root, "top_fork", "fam", "strat.py"), "w") as handle:
    handle.write("import top_fork.ext\\nimport gate_fork\\ngate_fork.entered.set()\\n"
                 "assert gate_fork.release.wait(30)\\nVALUE = 2\\n")
errors = []
def refresh():
    try:
        module_refresh.refresh_package_closure("top_fork.fam", "top_fork.fam.strat")
    except BaseException as exc:
        errors.append(exc)
refresher = threading.Thread(target=refresh, daemon=True)
refresher.start()
# Handshake: the refresher is parked inside the transaction (family dropped, a fresh package and a
# new external module imported, the guard installed, the lock held) before the fork.
assert gate_fork.entered.wait(10)
pid = os.fork()
if pid == 0:
    code = 0
    try:
        now = {k: v for k, v in sys.modules.items() if k.startswith("top_fork.fam")}
        if set(now) != set(old) or any(now[k] is not old[k] for k in old):
            code = 11
        elif vars(top).get("fam") is not old_binding:
            code = 12
        elif "top_fork.ext" in sys.modules or "ext" in vars(top):
            code = 13
        elif any(type(f).__name__ == "_ImportGuard" for f in sys.meta_path):
            code = 14
        elif not module_refresh._REFRESH_LOCK.acquire(timeout=5):
            code = 15
        elif module_refresh._ACTIVE is not None or getattr(
                module_refresh._TRANSACTION, "active", False):
            code = 16
        elif (module_refresh.serialized_import("top_fork.fam.strat")
              is not old["top_fork.fam.strat"]):
            code = 17
        else:
            module_refresh._REFRESH_LOCK.release()
            # CPython's own per-module import lock for strat stays held by the vanished thread
            # that was executing its body (import-quiescent precondition), so the later refresh
            # goes through a root that thread was not initializing.
            fresh = module_refresh.refresh_package_closure("top_fork.fam", "top_fork.fam.other")
            if fresh.VALUE != 3 or sys.modules["top_fork.fam.other"] is not fresh:
                code = 18
    except BaseException:
        traceback.print_exc()
        code = 19
    os._exit(code)
_, status = os.waitpid(pid, 0)
gate_fork.release.set()
refresher.join(30)
print(os.waitstatus_to_exitcode(status), flush=True)
os._exit(0 if not errors and not refresher.is_alive() else 5)
"""


def _fork_probe_tree(tmp_path: Path) -> None:
    (tmp_path / "top_fork" / "fam").mkdir(parents=True)
    (tmp_path / "top_fork" / "ext").mkdir()
    for package in ("top_fork", "top_fork/fam", "top_fork/ext"):
        (tmp_path / package / "__init__.py").write_text("")
    (tmp_path / "top_fork" / "fam" / "strat.py").write_text("VALUE = 1\n")
    (tmp_path / "top_fork" / "fam" / "other.py").write_text("VALUE = 3\n")
    (tmp_path / "gate_fork.py").write_text(
        "import threading\nentered = threading.Event()\nrelease = threading.Event()\n")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork-only platform behavior")
def test_a_fork_mid_refresh_recovers_the_pre_refresh_state_in_the_child(tmp_path) -> None:
    """A fork while ANOTHER thread is mid-refresh leaves the child a transaction nobody will
    finish: a partially rebuilt family, a new external module and its parent binding, the guard
    and a lock owned by a vanished thread. The child restores the pre-refresh modules and direct
    parent bindings, removes the guard, resets the transaction and replaces the lock, so warm loads
    and a later refresh in the child proceed; the parent's own refresh is unaffected."""
    _fork_probe_tree(tmp_path)

    result = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", _MID_REFRESH_FORK_PROBE,
         str(tmp_path)],
        capture_output=True, text=True, timeout=120, check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.strip() == "0", (result.stdout, result.stderr)


_OWNER_FORK_PROBE = """
import os, sys
sys.path.insert(0, sys.argv[1])
from algua.primitives.module_refresh import refresh_package_closure
import top_fork.fam.strat
with open(os.path.join(sys.argv[1], "top_fork", "fam", "strat.py"), "w") as handle:
    handle.write("import os\\nimport top_fork.ext\\nPID = os.fork()\\nVALUE = 2\\n")
fresh = refresh_package_closure("top_fork.fam", "top_fork.fam.strat")
committed = sys.modules["top_fork.fam.strat"] is fresh and fresh.VALUE == 2
if fresh.PID == 0:
    os._exit(0 if committed else 21)
_, status = os.waitpid(fresh.PID, 0)
print(os.waitstatus_to_exitcode(status), flush=True)
os._exit(0 if committed else 22)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork-only platform behavior")
def test_a_fork_by_the_refreshing_thread_itself_continues_the_transaction(tmp_path) -> None:
    """When the thread running the refresh forks (from a module body), that same thread goes on
    in the child and finishes the transaction there, so the child must NOT roll it back."""
    _fork_probe_tree(tmp_path)

    result = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", _OWNER_FORK_PROBE,
         str(tmp_path)],
        capture_output=True, text=True, timeout=120, check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.strip() == "0", (result.stdout, result.stderr)


_AFTER_REFRESH_FORK_PROBE = """
import os, sys, threading
sys.path.insert(0, sys.argv[1])
from algua.primitives.module_refresh import refresh_package_closure
import top_fork.fam.strat
with open(os.path.join(sys.argv[1], "top_fork", "fam", "strat.py"), "w") as handle:
    handle.write("VALUE = 2\\n")
done = {}
worker = threading.Thread(target=lambda: done.update(
    root=refresh_package_closure("top_fork.fam", "top_fork.fam.strat")))
worker.start()
worker.join(30)
pid = os.fork()
if pid == 0:
    kept = sys.modules["top_fork.fam.strat"] is done["root"] and done["root"].VALUE == 2
    os._exit(0 if kept else 31)
_, status = os.waitpid(pid, 0)
print(os.waitstatus_to_exitcode(status), flush=True)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork-only platform behavior")
def test_a_fork_after_a_completed_refresh_keeps_the_committed_family(tmp_path) -> None:
    """A finished transaction leaves no recovery record behind, so a later fork never rolls a
    committed refresh back in the child."""
    _fork_probe_tree(tmp_path)

    result = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", _AFTER_REFRESH_FORK_PROBE,
         str(tmp_path)],
        capture_output=True, text=True, timeout=120, check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.strip() == "0", (result.stdout, result.stderr)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special files")
@pytest.mark.parametrize("swap", ["fifo", "symlink"])
def test_a_source_node_swapped_after_the_scan_is_never_read_through(
        tmp_path, special_node, swap) -> None:
    """The preflight read opens without blocking or following a final link and requires a regular
    file, so a node swapped in after the tree scan cannot block the read or redirect it."""
    origin = tmp_path / "helper.py"
    if swap == "fifo":
        special_node(origin, "fifo")
        match = "non-regular"
    else:
        (tmp_path / "elsewhere.py").write_text("VALUE = 1\n")
        origin.symlink_to(tmp_path / "elsewhere.py")
        match = "OSError, errno"

    with pytest.raises(ModuleRefreshError, match=match) as caught:
        module_source_scan._parse(str(origin), "fam.helper")

    assert str(tmp_path) not in str(caught.value)


@pytest.mark.parametrize("entry", ["sys-path", "dotted-sys-path", "parent-search-path"])
def test_a_relative_search_entry_is_frozen_before_refreshed_code_can_change_directory(
        tmp_path, monkeypatch, entry) -> None:
    """A relative ``sys.path`` or parent ``__path__`` entry is resolved against the working
    directory once, before preflight: refreshed code that changes directory mid-import cannot make
    a later family import resolve, and execute, a same-shaped tree under the new directory. The
    frozen location is canonical, so a ``./`` entry yields the same module paths."""
    top = f"mrrel_{uuid.uuid4().hex[:10]}"
    package = f"{top}.fam" if entry == "parent-search-path" else top
    marker = tmp_path / "alt-executed"
    for base in (tmp_path, tmp_path / "alt"):
        directory = base.joinpath("src", *package.split("."))
        directory.mkdir(parents=True)
        (base / "src" / top / "__init__.py").write_text("")
        (directory / "__init__.py").write_text("")
        (directory / "helper.py").write_text(
            "VALUE = 'real'\n" if base == tmp_path
            else f"open({str(marker)!r}, 'w').close()\nVALUE = 'alt'\n")
    real = tmp_path.joinpath("src", *package.split("."))
    (real / "strat.py").write_text(
        f"import os\nos.chdir({str(tmp_path / 'alt')!r})\n"
        "from . import helper\nVALUE = helper.VALUE\n")
    monkeypatch.chdir(tmp_path)
    if entry != "parent-search-path":
        monkeypatch.syspath_prepend("src" if entry == "sys-path" else "./src")
    else:
        monkeypatch.syspath_prepend(str(tmp_path / "src"))
        importlib.import_module(top).__path__ = [os.path.join("src", top)]
    try:
        root = refresh_package_closure(package, f"{package}.strat")

        assert not marker.exists(), "a same-shaped tree under the new working directory executed"
        assert root.VALUE == "real"
        assert sys.modules[f"{package}.helper"].__file__ == str(real / "helper.py")
    finally:
        for key in [k for k in sys.modules if k == top or k.startswith(top + ".")]:
            del sys.modules[key]


def test_a_relative_search_entry_under_a_vanished_working_directory_is_a_bounded_refusal(
        tmp_path, monkeypatch) -> None:
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    monkeypatch.syspath_prepend("src")

    with pytest.raises(ModuleRefreshError, match=r"cannot be inspected \(FileNotFoundError"):
        refresh_package_closure("mrgone_pkg", "mrgone_pkg.strat")


_SPEC_MUTATIONS = {
    "loader-type": (
        "import importlib.machinery\nfrom . import helper\n"
        "Loader = type('Loader', (importlib.machinery.SourceFileLoader,), {})\n"
        "helper.__loader__ = helper.__spec__.loader = Loader(helper.__name__, helper.__file__)\n"),
    "loader-name": "from . import helper\nhelper.__spec__.loader.name = 'json'\n",
    "loader-path": "from . import helper\nhelper.__spec__.loader.path = '/elsewhere/helper.py'\n",
    "spec-name": "from . import helper\nhelper.__spec__.name = 'json'\n",
    "spec-origin": "from . import helper\nhelper.__spec__.origin = '/elsewhere/helper.py'\n",
    "spec-cached": "from . import helper\nhelper.__spec__.cached = '/elsewhere/helper.pyc'\n",
    "own-spec-origin": "__spec__.origin = '/elsewhere/strat.py'\n",
    "package-spec-search-rebound": (
        "import sys\n"
        "sys.modules[__package__].__spec__.submodule_search_locations = ['/elsewhere']\n"),
    "module-spec-gains-search": (
        "from . import helper\nhelper.__spec__.submodule_search_locations = []\n"),
    "module-name": "from . import helper\nhelper.__name__ = 'json'\n",
    "module-package": "from . import helper\nhelper.__package__ = 'json'\n",
    "package-package": "import sys\nsys.modules[__package__].__package__ = 'json'\n",
    "module-loader": (
        "import importlib.machinery\nfrom . import helper\n"
        "helper.__loader__ = importlib.machinery.SourceFileLoader(helper.__name__, helper.__file__)"
        "\n"),
    "module-file": "from . import helper\nhelper.__file__ = '/elsewhere/helper.py'\n",
    "module-file-removed": "from . import helper\ndel helper.__file__\n",
    "module-file-str-subclass": (
        "from . import helper\nhelper.__file__ = type('S', (str,), {})(helper.__file__)\n"),
    "module-cached": "from . import helper\nhelper.__cached__ = '/elsewhere/helper.pyc'\n",
    "package-path-str-subclass": (
        "import sys\nsearch = sys.modules[__package__].__path__\n"
        "search[0] = type('S', (str,), {})(search[0])\n"),
}


@pytest.mark.parametrize("case", sorted(_SPEC_MUTATIONS))
def test_in_place_mutation_of_an_executed_source_spec_never_commits(family, case) -> None:
    """The guard records immutable facts of every source spec it hands out; at commit the SAME
    spec object must still carry them (exact ``SourceFileLoader`` type, loader name and path, spec
    name, origin, cache and search locations) and its module the corresponding ``__name__``,
    ``__package__``, ``__loader__``, ``__file__`` and ``__cached__``, so code that mutated the
    spec in place, which an identity check alone accepts, rolls the transaction back."""
    _assert_never_commits(family, _SPEC_MUTATIONS[case])


_IDENTITY_FORGERIES = {
    "spec-class-mutated": (
        "import importlib.machinery\nfrom . import helper\n"
        "helper.__spec__.__class__ = type('Spec', (importlib.machinery.ModuleSpec,), {})\n"),
    "own-spec-class-mutated": (
        "import importlib.machinery\n"
        "__spec__.__class__ = type('Spec', (importlib.machinery.ModuleSpec,), {})\n"),
    "loader-replaced-same-shape": (
        "import importlib.machinery\nfrom . import helper\n"
        "helper.__loader__ = helper.__spec__.loader = "
        "importlib.machinery.SourceFileLoader(helper.__name__, helper.__file__)\n"),
    "spec-loader-replaced-same-shape": (
        "import importlib.machinery\nfrom . import helper\n"
        "helper.__spec__.loader = "
        "importlib.machinery.SourceFileLoader(helper.__name__, helper.__file__)\n"),
    "loader-class-mutated": (
        "import importlib.machinery\nfrom . import helper\n"
        "helper.__loader__.__class__ = type('L', (importlib.machinery.SourceFileLoader,), {})\n"),
    "loader-get-code-injected": (
        "from . import helper\nhelper.__loader__.get_code = lambda name: None\n"),
    "loader-get-data-injected": (
        "from . import helper\nhelper.__spec__.loader.get_data = lambda path: b'VALUE = 2\\n'\n"),
}


@pytest.mark.parametrize("case", sorted(_IDENTITY_FORGERIES))
def test_a_forged_spec_class_or_loader_identity_or_state_never_commits(family, case) -> None:
    """At commit a family spec must still be exactly a ``ModuleSpec`` (not a subclass, even one
    swapped in through ``__class__``) whose loader is the ORIGINAL ``SourceFileLoader`` object the
    guard handed out, with unchanged concrete type and instance state. A same-shaped replacement
    loader (same name and path, bound to both the spec and ``__loader__``) or an instance attribute
    overriding a loader operation rolls the transaction back."""
    _assert_never_commits(family, _IDENTITY_FORGERIES[case])


_OWNER_FORK_CONTENDER_PROBE = """
import os, sys
sys.path.insert(0, sys.argv[1])
from algua.primitives import module_refresh
import fork_contender, top_fork.fam.strat
old = sys.modules["top_fork.fam.strat"]
body = "import fork_contender\\nfork_contender.fork_and_contend()\\nVALUE = 2\\n"
with open(os.path.join(sys.argv[1], "top_fork", "fam", "strat.py"), "w") as handle:
    handle.write(body + ("raise RuntimeError('roll back')\\n" if sys.argv[2] == "rollback" else ""))
try:
    fresh = module_refresh.refresh_package_closure("top_fork.fam", "top_fork.fam.strat")
    finished = sys.argv[2] == "commit" and sys.modules["top_fork.fam.strat"] is fresh
except RuntimeError:
    finished = sys.argv[2] == "rollback" and sys.modules["top_fork.fam.strat"] is old
if fork_contender.pid == 0:
    contender = fork_contender.contenders[0]
    contender.join(30)
    code = (41 if fork_contender.seen["entered_mid_transaction"] is not False
            else 42 if contender.is_alive()
            else 43 if sys.modules["turn_probe"].ACTIVE_WHEN_ENTERED
            else 0 if finished else 44)
    os._exit(code)
_, status = os.waitpid(fork_contender.pid, 0)
print(os.waitstatus_to_exitcode(status), flush=True)
os._exit(0 if finished else 45)
"""

_FORK_CONTENDER = """
import os, threading
from algua.primitives import module_refresh
pid, seen, contenders = None, {}, []
def probe():
    acquired = module_refresh._REFRESH_LOCK.acquire(blocking=False)
    if acquired:
        module_refresh._REFRESH_LOCK.release()
    seen["entered_mid_transaction"] = acquired
def fork_and_contend():
    global pid
    pid = os.fork()
    if pid == 0:
        prober = threading.Thread(target=probe)
        prober.start()
        prober.join(30)
        contender = threading.Thread(
            target=module_refresh.serialized_import, args=("turn_probe",), daemon=True)
        contender.start()
        contenders.append(contender)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork-only platform behavior")
@pytest.mark.parametrize("outcome", ["commit", "rollback"])
def test_a_second_child_thread_waits_for_a_transaction_its_forking_owner_continues(
        tmp_path, outcome) -> None:
    """When the refreshing thread itself forks, it still owns the transaction and the lock in the
    child, so the lock is kept rather than replaced: a second child thread cannot acquire it and
    a supported caller there stays blocked until the owner commits or rolls back. Deterministic:
    a non-blocking acquire proves the lock is held mid-transaction, and the blocked caller records
    whether a transaction was still active when it finally entered."""
    _fork_probe_tree(tmp_path)
    (tmp_path / "fork_contender.py").write_text(_FORK_CONTENDER)
    (tmp_path / "turn_probe.py").write_text(
        "from algua.primitives import module_refresh\n"
        "ACTIVE_WHEN_ENTERED = module_refresh._ACTIVE is not None\n")

    result = subprocess.run(
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", _OWNER_FORK_CONTENDER_PROBE,
         str(tmp_path), outcome],
        capture_output=True, text=True, timeout=120, check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.strip() == "0", (result.stdout, result.stderr)


_MODULE_CLASS_FORGERIES = {
    "member-class-swapped": (
        "import types\nfrom . import helper\n"
        "helper.__class__ = type('M', (types.ModuleType,), {})\n"),
    "root-class-swapped": (
        "import sys, types\n"
        "sys.modules[__name__].__class__ = type('M', (types.ModuleType,), {})\n"),
    "package-class-swapped": (
        "import sys, types\n"
        "sys.modules[__package__].__class__ = type('M', (types.ModuleType,), {})\n"),
    "direct-parent-class-swapped": (
        "import sys, types\ntop = __package__.rpartition('.')[0]\n"
        "sys.modules[top].__class__ = type('M', (types.ModuleType,), {})\n"),
    "direct-parent-replaced-by-subclass": (
        "import sys, types\ntop = __package__.rpartition('.')[0]\n"
        "twin = type('M', (types.ModuleType,), {})(top)\n"
        "vars(twin).update(vars(sys.modules[top]))\nsys.modules[top] = twin\n"),
}


@pytest.mark.parametrize("case", sorted(_MODULE_CLASS_FORGERIES))
def test_a_family_module_or_direct_parent_that_is_not_exactly_a_module_never_commits(
        family, case) -> None:
    """Every committed family entry and every direct parent it is bound on must be exactly a
    ``ModuleType``: a subclass, including one swapped in through ``__class__``, can conceal import
    metadata from the dictionary-based commit validation, so it rolls the transaction back."""
    _assert_never_commits(family, _MODULE_CLASS_FORGERIES[case])


def test_a_package_subclass_concealing_an_alternate_search_path_never_commits(
        family, tmp_path) -> None:
    """A fresh package whose class is swapped to a ``ModuleType`` subclass answering ``__path__``
    with an alternate tree keeps its confined ``__path__`` in its dictionary, so it would pass a
    dictionary-based check, and after commit (with the guard gone) a lazy import of a new family
    name would load from outside the root. The exact-type check rolls it back instead, and the
    restored family cannot reach the alternate tree."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "evil.py").write_text("VALUE = 'out-of-root'\n")
    _assert_never_commits(family, (
        "import sys, types\n"
        f"ALTERNATE = [{str(elsewhere)!r}]\n"
        "class Concealing(types.ModuleType):\n"
        "    def __getattribute__(self, name):\n"
        "        if name == '__path__':\n"
        "            return ALTERNATE\n"
        "        return super().__getattribute__(name)\n"
        "sys.modules[__package__].__class__ = Concealing\n"
        "VALUE = 1\n"))

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"{family.package}.evil")
    assert f"{family.package}.evil" not in sys.modules


_SECOND_RESOLUTIONS = {
    "reimported-by-import-module": (
        "import importlib, sys\nfrom . import helper\n"
        "del sys.modules[helper.__name__]\nimportlib.import_module(helper.__name__)\nVALUE = 1\n"),
    "reimported-by-import-statement": (
        "import sys\nfrom . import helper\n"
        "del sys.modules[helper.__name__]\n__import__(helper.__name__)\nVALUE = 1\n"),
    "refusal-swallowed": (
        "import importlib, sys\nfrom . import helper\ndel sys.modules[helper.__name__]\n"
        "try:\n    importlib.import_module(helper.__name__)\nexcept ImportError:\n    pass\n"
        "VALUE = 1\n"),
    "reimported-subpackage-member": (
        "import importlib, sys\nfrom .sub import inner\n"
        "del sys.modules[inner.__name__]\nimportlib.import_module(inner.__name__)\nVALUE = 1\n"),
    "refusal-swallowed-and-first-object-restored": (
        "import importlib, sys\nfrom algua.primitives.module_refresh import ModuleRefreshError\n"
        "from . import helper\nfirst = helper\ndel sys.modules[helper.__name__]\n"
        "try:\n    importlib.import_module(helper.__name__)\nexcept ModuleRefreshError:\n    pass\n"
        "sys.modules[first.__name__] = first\n"
        "setattr(sys.modules[__package__], 'helper', first)\nVALUE = 1\n"),
    "subpackage-refusal-swallowed-and-first-object-restored": (
        "import importlib, sys\nfrom algua.primitives.module_refresh import ModuleRefreshError\n"
        "from .sub import inner\nfirst = inner\ndel sys.modules[inner.__name__]\n"
        "try:\n    importlib.import_module(inner.__name__)\nexcept ModuleRefreshError:\n    pass\n"
        "sys.modules[first.__name__] = first\n"
        "setattr(sys.modules[__package__ + '.sub'], 'inner', first)\nVALUE = 1\n"),
}


@pytest.mark.parametrize("case", sorted(_SECOND_RESOLUTIONS))
def test_a_second_resolution_of_the_same_family_name_never_commits(family, case) -> None:
    """Refreshed code that imports a family module, drops its ``sys.modules`` entry and imports
    it again would keep the stale first object while the replacement is the one certified in
    ``sys.modules`` and on its parent. The guard refuses to hand out a second spec for a name it
    already resolved in the transaction, so the complete transaction rolls back. The attempt is
    latched for the whole transaction: code that swallows the refusal, even one that then restores
    the first object's ``sys.modules`` entry and direct parent binding, still cannot commit."""
    (family.dir / "sub").mkdir()
    (family.dir / "sub" / "__init__.py").write_text("")
    (family.dir / "sub" / "inner.py").write_text("VALUE = 1\n")
    _assert_never_commits(family, _SECOND_RESOLUTIONS[case])


def test_distinct_names_and_repeated_misses_resolve_normally_in_every_transaction(family) -> None:
    """Only a second HAND-OUT of the same name is refused: distinct family names (a subpackage and
    its member included) resolve once each, a repeated probe for a missing name is never a
    duplicate, and a later transaction resolves the same names afresh."""
    (family.dir / "sub").mkdir()
    (family.dir / "sub" / "__init__.py").write_text("")
    (family.dir / "sub" / "inner.py").write_text("VALUE = 2\n")
    family.write("helper", "VALUE = 1\n")
    family.write("strat", (
        "import importlib\nfor _ in range(2):\n    try:\n"
        "        importlib.import_module(__package__ + '.optional')\n"
        "    except ModuleNotFoundError:\n        pass\n"
        "from . import helper\nfrom .sub import inner\nVALUE = helper.VALUE + inner.VALUE\n"))

    for _ in range(2):
        fresh = refresh_package_closure(family.package, f"{family.package}.strat")
        assert fresh.VALUE == 3 and family.mod("strat") is fresh
        assert fresh.helper is family.mod("helper") and fresh.inner is family.mod("sub.inner")


def test_a_latched_duplicate_resolution_does_not_outlive_its_transaction(family) -> None:
    """The duplicate-resolution latch belongs to one transaction: after a refresh whose code
    swallowed the refusal and restored the first object is rolled back, a later refresh of the
    same names from clean source commits normally."""
    family.write("helper", "VALUE = 1\n")
    family.write("strat", _SECOND_RESOLUTIONS["refusal-swallowed-and-first-object-restored"])
    with pytest.raises(ModuleRefreshError, match="second time"):
        _refresh(family)
    assert not family.entries()

    family.write("strat", "from . import helper\nVALUE = helper.VALUE + 1\n")
    fresh = refresh_package_closure(family.package, f"{family.package}.strat")
    assert fresh.VALUE == 2 and family.mod("strat") is fresh
    assert fresh.helper is family.mod("helper")
    assert vars(sys.modules[family.package])["helper"] is family.mod("helper")


_SPEC_BOOKKEEPING_FORGERIES = {
    "loader-state-set": (
        "from . import helper\nhelper.__spec__.loader_state = {'origin': '/elsewhere'}\n"),
    "loader-state-removed": "from . import helper\ndel helper.__spec__.loader_state\n",
    "own-loader-state-set": "__spec__.loader_state = 'forged'\n",
    "cached-removed": "from . import helper\ndel helper.__spec__._cached\n",
    "origin-removed": "from . import helper\ndel helper.__spec__.origin\n",
    "has-location-cleared": "from . import helper\nhelper.__spec__.has_location = False\n",
    "set-fileattr-truthy-int": "from . import helper\nhelper.__spec__._set_fileattr = 1\n",
    "set-fileattr-removed": "from . import helper\ndel helper.__spec__._set_fileattr\n",
    "uninitialized-submodule-forged": (
        "from . import helper\nhelper.__spec__._uninitialized_submodules.append('ghost')\n"),
    "package-uninitialized-submodule-forged": (
        "import sys\n"
        "sys.modules[__package__].__spec__._uninitialized_submodules.append('ghost')\n"),
    "package-uninitialized-submodules-replaced": (
        "import sys\nspec = sys.modules[__package__].__spec__\n"
        "spec._uninitialized_submodules = list(spec._uninitialized_submodules)\n"),
    "uninitialized-submodules-equal-subclass": (
        "from . import helper\nhelper.__spec__._uninitialized_submodules = "
        "type('L', (list,), {'__contains__': lambda self, item: True})()\n"),
    "uninitialized-submodules-removed": (
        "from . import helper\ndel helper.__spec__._uninitialized_submodules\n"),
    "initializing-stuck": "from . import helper\nhelper.__spec__._initializing = True\n",
    "initializing-falsy-zero": "from . import helper\nhelper.__spec__._initializing = 0\n",
    "initializing-removed": "from . import helper\ndel helper.__spec__._initializing\n",
    "package-initializing-stuck": (
        "import sys\nsys.modules[__package__].__spec__._initializing = True\n"),
}


@pytest.mark.parametrize("case", sorted(_SPEC_BOOKKEEPING_FORGERIES))
def test_forged_module_spec_bookkeeping_never_commits(family, case) -> None:
    """The guard records a handed-out spec's import-machinery bookkeeping before importlib sees
    it: no loader state, its location flag and its own empty uninitialized-submodules list. At
    commit the spec must still carry exactly those, and the load's final ``_initializing`` must be
    exactly ``False``: importlib legitimately sets it ``True`` while executing and ``False`` once
    the load completes, so that completed state, not the pre-load absence, is the one required. A
    rebound, removed, forged or merely equal replacement value rolls the transaction back."""
    _assert_never_commits(family, _SPEC_BOOKKEEPING_FORGERIES[case])


_METADATA_DELETIONS = {
    "search-locations-removed": (
        "from . import helper\ndel helper.__spec__.submodule_search_locations\n"),
    "own-search-locations-removed": "del __spec__.submodule_search_locations\n",
    "package-search-locations-removed": (
        "import sys\ndel sys.modules[__package__].__spec__.submodule_search_locations\n"),
    "spec-name-removed": "from . import helper\ndel helper.__spec__.name\n",
    "spec-loader-removed": "from . import helper\ndel helper.__spec__.loader\n",
    "module-spec-removed": "from . import helper\ndel helper.__spec__\n",
    "module-loader-removed": "from . import helper\ndel helper.__loader__\n",
    "module-name-removed": "from . import helper\ndel helper.__name__\n",
    "module-package-removed": "from . import helper\ndel helper.__package__\n",
    "module-cached-removed": "from . import helper\ndel helper.__cached__\n",
    "package-path-removed": "import sys\ndel sys.modules[__package__].__path__\n",
}


@pytest.mark.parametrize("case", sorted(_METADATA_DELETIONS))
def test_deleted_spec_or_module_metadata_never_commits(family, case) -> None:
    """A recorded fact whose valid value is ``None`` (an ordinary module's
    ``submodule_search_locations``) is not satisfied by its absence: every spec and module
    metadata lookup at commit distinguishes a missing entry from ``None``, so deleting any of them
    rolls the complete transaction back."""
    _assert_never_commits(family, _METADATA_DELETIONS[case])


_NO_BYTECODE_CACHE_FORGERIES = {
    "spec-cached-removed": "from . import helper\ndel helper.__spec__._cached\n",
    "module-cached-forged-none": "from . import helper\nhelper.__cached__ = None\n",
    "own-spec-cached-removed": "del __spec__._cached\n",
}


@pytest.mark.parametrize("case", sorted(_NO_BYTECODE_CACHE_FORGERIES))
def test_without_a_bytecode_cache_a_deleted_or_forged_cache_entry_never_commits(
        family, monkeypatch, case) -> None:
    """Without a cache tag a source spec's valid cache is ``None`` and its module has no
    ``__cached__``: deleting the spec's ``None`` cache, or adding a ``None`` ``__cached__`` the
    import system never sets, is a divergence from the recorded state and never commits."""
    monkeypatch.setattr(sys.implementation, "cache_tag", None)
    _assert_never_commits(family, _NO_BYTECODE_CACHE_FORGERIES[case])


def test_a_family_without_a_bytecode_cache_refreshes_normally(family, monkeypatch) -> None:
    """The valid ``None`` cache and absent ``__cached__`` of a cache-less import commit."""
    monkeypatch.setattr(sys.implementation, "cache_tag", None)
    family.write("helper", "VALUE = 1\n")
    family.write("strat", "from . import helper\nVALUE = helper.VALUE + 1\n")
    fresh = refresh_package_closure(family.package, f"{family.package}.strat")
    helper = family.mod("helper")
    assert fresh.VALUE == 2 and fresh.helper is helper and family.mod("strat") is fresh
    assert vars(helper.__spec__)["_cached"] is None and "__cached__" not in vars(helper)


_SENTINEL = "from algua.primitives import module_commit_check\n"
_ABSENT_SENTINEL_FORGERIES = {
    "member-cached": (
        _SENTINEL + "from . import helper\nhelper.__cached__ = module_commit_check._ABSENT\n"),
    "own-cached": _SENTINEL + "__cached__ = module_commit_check._ABSENT\n",
    "package-cached": (
        _SENTINEL + "import sys\n"
        "sys.modules[__package__].__cached__ = module_commit_check._ABSENT\n"),
    "member-path": (
        _SENTINEL + "from . import helper\nhelper.__path__ = module_commit_check._ABSENT\n"),
    "own-path": _SENTINEL + "__path__ = module_commit_check._ABSENT\n",
}


@pytest.mark.parametrize("case", sorted(_ABSENT_SENTINEL_FORGERIES))
def test_an_installed_absence_sentinel_never_satisfies_absent_module_metadata(
        family, monkeypatch, case) -> None:
    """Without a cache tag a module has no ``__cached__``, and an ordinary module never has a
    ``__path__``: those requirements are checked by key membership, so executed code that
    imports the commit check's own ``_ABSENT`` sentinel and installs it as a PRESENT value does
    not look absent, and the complete transaction rolls back."""
    monkeypatch.setattr(sys.implementation, "cache_tag", None)
    _assert_never_commits(family, _ABSENT_SENTINEL_FORGERIES[case])


# Values rollback must restore or drop by presence alone, never mistake for absence: ``None`` (a
# dictionary lookup's default), an ordinary object, and every bare ``object()`` sentinel the refresh
# seam exposes, which executed code can import and install as an entry's or binding's value.
_PRESENCE_VALUES = [
    pytest.param(None, id="none"),
    pytest.param(object(), id="object"),
    *(pytest.param(value, id=f"{module.__name__.rpartition('.')[2]}.{name}")
      for module in (module_refresh, module_commit_check)
      for name, value in vars(module).items() if type(value) is object),
]


def _ghost(family, scope: str) -> str:
    return f"{family.package if scope == 'family' else family.top}.ghost"


@pytest.mark.parametrize("value", _PRESENCE_VALUES)
@pytest.mark.parametrize("scope", ["family", "external"])
def test_rollback_drops_an_introduced_sys_modules_entry_whatever_its_value(
        family, monkeypatch, scope, value) -> None:
    """Rollback finds changed ``sys.modules`` entries by key membership, then identity: an entry
    the failed attempt introduced is dropped even when its value is ``None`` or a seam sentinel."""
    importlib.import_module(family.top)
    carrier = ModuleType(f"mrcarrier_{uuid.uuid4().hex[:10]}")
    vars(carrier)["value"] = value
    monkeypatch.setitem(sys.modules, carrier.__name__, carrier)
    name = _ghost(family, scope)
    family.write("strat", (
        f"import sys\nsys.modules[{name!r}] = sys.modules[{carrier.__name__!r}].value\n"
        "raise RuntimeError('boom')\n"))

    with pytest.raises(RuntimeError, match="boom"):
        _refresh(family)

    assert name not in sys.modules


@pytest.mark.parametrize("value", _PRESENCE_VALUES)
@pytest.mark.parametrize("scope", ["family", "external"])
def test_rollback_reinstates_a_removed_sys_modules_entry_whatever_its_value(
        family, scope, value) -> None:
    """A pre-existing entry the failed attempt removed (a family entry the refresh drops, or an
    external one executed code pops) is reinstated with its exact value, whatever that value."""
    importlib.import_module(family.top)
    name = _ghost(family, scope)
    sys.modules[name] = value
    family.write("strat", (
        f"import sys\nsys.modules.pop({name!r}, None)\nraise RuntimeError('boom')\n"))

    with pytest.raises(RuntimeError, match="boom"):
        _refresh(family)

    assert name in sys.modules and sys.modules[name] is value


@pytest.mark.parametrize("value", _PRESENCE_VALUES)
@pytest.mark.parametrize("child", ["fam", "ext"])
def test_rollback_reinstates_a_pre_existing_parent_binding_whatever_its_value(
        family, child, value) -> None:
    """The guard snapshots each direct parent binding as an explicit presence bit plus its exact
    value, so a pre-existing binding the failed attempt overwrote (by importing the family package
    or a new external module of that name) is reinstated, never deleted, whatever its value."""
    (family.dir.parent / "ext").mkdir()
    (family.dir.parent / "ext" / "__init__.py").write_text("")
    top = importlib.import_module(family.top)
    vars(top)[child] = value
    family.write("strat", f"import {family.top}.ext\nraise RuntimeError('boom')\n")

    with pytest.raises(RuntimeError, match="boom"):
        _refresh(family)

    assert child in vars(top) and vars(top)[child] is value
    assert f"{family.top}.ext" not in sys.modules and family.entries() == {}
