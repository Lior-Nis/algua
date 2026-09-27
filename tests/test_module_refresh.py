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

from algua.primitives import module_refresh
from algua.primitives.module_refresh import (
    ModuleRefreshError,
    _require_acyclic,
    _static_closure,
    refresh_package_closure,
)


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

    graph = _static_closure(pkg, f"{pkg}.strat")

    assert graph[f"{pkg}.strat"] == {
        pkg, f"{pkg}._a", f"{pkg}._b", f"{pkg}._c", f"{pkg}._d", f"{pkg}._e",
        f"{pkg}.sub", f"{pkg}.sub.deep",
    }
    assert graph[f"{pkg}.sub.deep"] == {f"{pkg}.sub"}
    assert graph[f"{pkg}.sub"] == {pkg}
    assert graph[pkg] == set()
    assert f"{pkg}.unreached" not in graph


def test_require_acyclic_names_the_cycle_deterministically() -> None:
    _require_acyclic({"f": {"f._a"}, "f._a": {"f._b"}, "f._b": set()})
    graph = {"f": set(), "f._c": {"f._d"}, "f._d": {"f._e"}, "f._e": {"f._c"}}
    with pytest.raises(ModuleRefreshError) as first:
        _require_acyclic(graph)
    with pytest.raises(ModuleRefreshError) as second:
        _require_acyclic(dict(reversed(list(graph.items()))))
    assert str(first.value) == str(second.value)
    assert "f._c -> f._d -> f._e -> f._c" in str(first.value)


def test_require_acyclic_is_iterative_beyond_the_recursion_limit() -> None:
    depth = sys.getrecursionlimit() + 500
    graph: dict[str, set[str]] = {f"f._{i}": {f"f._{i + 1}"} for i in range(depth)}
    graph[f"f._{depth}"] = set()
    _require_acyclic(graph)

    graph[f"f._{depth}"] = {"f._0"}
    with pytest.raises(ModuleRefreshError, match="cycl"):
        _require_acyclic(graph)


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

    _require_acyclic(graph)

    assert sorted(expanded) == sorted(graph)


def test_refresh_accepts_a_lazy_acyclic_chain_deeper_than_the_recursion_limit(family) -> None:
    depth = sys.getrecursionlimit() + 200
    for i in range(depth):
        family.write(f"h{i}", f"def follow():\n    from . import h{i + 1}\n")
    family.write(f"h{depth}", "VALUE = 1\n")
    family.write("strat", "from . import h0\nVALUE = 1\n")

    _refresh(family)

    assert family.mod("strat").VALUE == 1


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

    monkeypatch.setattr(module_refresh, "_static_closure", discovery)
    with pytest.raises(ModuleRefreshError, match="symlink"):
        _refresh(family)

    assert cached.is_file(), "a refused refresh must not purge anything"
    after = family.entries()
    assert set(after) == set(before) and all(after[k] is v for k, v in before.items())


def test_a_symlinked_search_location_is_refused_before_source_discovery(
        family, tmp_path, monkeypatch) -> None:
    linked = tmp_path / "linked"
    linked.symlink_to(family.dir.parent.parent, target_is_directory=True)
    monkeypatch.syspath_prepend(str(linked))
    importlib.invalidate_caches()
    family.write("strat", "VALUE = 1\n")

    def discovery(*_args: object) -> None:
        raise AssertionError("source discovery ran before the search-location check")

    monkeypatch.setattr(module_refresh, "_static_closure", discovery)
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
import importlib, os, sys, threading, time
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
time.sleep(0.5)  # the refresh now waits for slow_probe, which another thread is initializing
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
    (tmp_path / "top_probe" / "fam" / "strat.py").write_text("import slow_probe\nVALUE = 1\n")
    (tmp_path / "gate_probe.py").write_text(
        "import threading\nentered = threading.Event()\ngo = threading.Event()\n")
    (tmp_path / "slow_probe.py").write_text(
        "import gate_probe\ngate_probe.entered.set()\nassert gate_probe.go.wait(10)\n"
        "import late_probe\n")
    (tmp_path / "late_probe.py").write_text("VALUE = 1\n")

    result = subprocess.run(
        [sys.executable, "-c", _DEADLOCK_PROBE, str(tmp_path)],
        capture_output=True, text=True, timeout=120, check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
