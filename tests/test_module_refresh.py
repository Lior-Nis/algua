"""Warm package-closure refresh: current source only, acyclic, fresh globals, all-or-nothing."""
from __future__ import annotations

import importlib
import importlib.util
import os
import py_compile
import sys
import textwrap
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

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
