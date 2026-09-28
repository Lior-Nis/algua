"""Re-import a package closure from its CURRENT source, all-or-nothing (stdlib only).

``importlib.reload`` is not a faithful refresh: it re-executes into the SAME module dictionary (a
global deleted from source survives, and a helper's removed name is silently re-imported), it
trusts timestamp-validated cached bytecode (a same-size, same-mtime edit keeps a stale ``.pyc``
valid), and a failure part-way leaves a mixed-version closure. The refresh therefore re-imports
the root as FRESH module objects after purging the package's bytecode, and restores the previous
module state if anything fails. A cyclic static import component has no dependency-safe
execution order, and a symlink or importable non-source module anywhere in the package tree (which
a dynamic import can reach even outside the static closure) escapes the package-wide bytecode
purge, so all fail closed before anything is purged or executed (``module_source_scan``). During
the fresh import a first ``sys.meta_path`` guard resolves family modules from the exact scanned
root only, so a fresh ``__init__`` cannot extend ``__path__`` into an unscanned tree.

Concurrency: the complete transaction runs under a private module lock shared ONLY by the
supported Algua callers (``refresh_package_closure`` and ``serialized_import``, which the strategy
loader uses for every cold, warm and reload path). It deliberately does not take CPython's global
import lock, whose ordering against per-module import locks deadlocks with an import already in
progress. This is not a claim of arbitrary ``importlib`` concurrency safety: the process must be
import-quiescent for everything else. A direct import of a family module (or of anything the
refresh imports) from another thread, or a supported call made from inside a module body another
thread is initializing, while a refresh runs is unsupported and may observe a partially rebuilt
family or deadlock.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import threading
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

from algua.primitives.module_source_scan import (
    ModuleRefreshError,
    require_acyclic,
    require_source_tree,
    require_unlinked,
    static_closure,
    within,
)

__all__ = ["ModuleRefreshError", "refresh_package_closure", "serialized_import"]

_ABSENT = object()
# Re-entrant: a supported caller that runs inside a refresh on the same thread must not self-block.
_REFRESH_LOCK = threading.RLock()


def serialized_import(name: str) -> ModuleType:
    """``importlib.import_module(name)`` under the refresh serialization, so a supported caller
    never imports into, or returns a module from, a family another caller is rebuilding."""
    with _REFRESH_LOCK:
        return importlib.import_module(name)


def refresh_package_closure(package: str, root: str) -> ModuleType:
    """Replace every loaded ``package`` module with a fresh import of ``root``'s current closure,
    returning the fresh ``root`` module.

    ``root`` (a module inside ``package``) is imported anew and pulls in, as fresh module objects,
    exactly the ``package`` modules its current source reaches; other previously loaded
    ``package`` modules are dropped and re-import fresh on next use. Modules outside ``package``
    stay warm. On any failure every ``sys.modules`` entry the attempt introduced or replaced is
    dropped, every previous entry is reinstated, and each affected module's direct parent binding
    is restored to its exact prior value or absence; the previous family objects' dictionaries
    were never touched.

    The private refresh lock is held from preflight to commit or rollback, so a supported caller
    waits for the final state; other imports must be quiescent (see the module docstring)."""
    if not root.startswith(package + "."):
        raise ValueError(f"{root!r} is not inside package {package!r}")
    with _REFRESH_LOCK:
        return _refresh_locked(package, root)


def _refresh_locked(package: str, root: str) -> ModuleType:
    """Snapshot module state BEFORE package discovery (``find_spec`` imports cold parents), so any
    failure from preflight to the fresh import, including a ``BaseException``, restores it."""
    before = dict(sys.modules)
    namespaces = _namespaces(before)
    guard = _ImportGuard()
    sys.meta_path.insert(0, guard)
    try:
        return _refresh_attempt(package, root, guard)
    except BaseException:
        _restore_modules(before, namespaces, guard.bindings)
        raise
    finally:
        if guard in sys.meta_path:
            sys.meta_path.remove(guard)


class _ImportGuard:
    """A first ``sys.meta_path`` finder held for the whole transaction. Before the import system
    loads (and then binds) any module, it records that module's direct parent binding as first
    seen, read from the parent's own dictionary; rollback restores it even if attempted code later
    removed the child's own ``sys.modules`` entry.

    Once the family is prevalidated, the guard also RESOLVES every family module itself, from the
    exact scanned root only: a child is refused before it executes when its parent's search path
    (which a fresh ``__init__`` or member could extend) is not exactly that root, and a family
    module that is missing from the root or is not Python source is never found elsewhere."""

    def __init__(self) -> None:
        self.bindings: dict[str, tuple[ModuleType, object]] = {}
        self.family: tuple[str, str] | None = None  # (package, prevalidated location)

    def find_spec(
            self, fullname: str, path: object = None, target: object = None) -> ModuleSpec | None:
        parent_name, _, child = fullname.rpartition(".")
        holder = sys.modules.get(parent_name) if parent_name else None
        if fullname not in self.bindings and isinstance(holder, ModuleType):
            self.bindings[fullname] = (holder, holder.__dict__.get(child, _ABSENT))
        if self.family is None or not within(fullname, self.family[0]):
            return None
        package, location = self.family
        if fullname == package:
            search = [os.path.dirname(location)]
        else:
            search = _exact_search_path(package, location, parent_name)
            if path != search:
                raise ModuleRefreshError(
                    f"{parent_name!r} search path deviates from its prevalidated family root",
                    name=fullname)
        spec = importlib.machinery.PathFinder.find_spec(fullname, search)
        if spec is None:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        if not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            raise ModuleRefreshError(f"{fullname!r} is not a Python source module", name=fullname)
        return spec


def _exact_search_path(package: str, location: str, name: str) -> list[str]:
    """The only search path a family package ``name`` may have: its directory under the root."""
    return [os.path.join(location, *name.split(".")[package.count(".") + 1:])]


def _require_confined_search_paths(package: str, location: str) -> None:
    """Before commit, every fresh family module that is a package must still search exactly its
    prevalidated directory, so a later lazy import cannot use a path extended during the refresh."""
    for name, module in list(sys.modules.items()):
        search = getattr(module, "__dict__", {}).get("__path__", _ABSENT)
        if within(name, package) and search is not _ABSENT and (
                search != _exact_search_path(package, location, name)):
            raise ModuleRefreshError(
                f"{name!r} search path deviates from its prevalidated family root", name=name)


def _refresh_attempt(package: str, root: str, guard: _ImportGuard) -> ModuleType:
    importlib.invalidate_caches()
    package_spec = importlib.util.find_spec(package)
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise ModuleRefreshError(f"{package!r} is not a package", name=package)
    locations = list(package_spec.submodule_search_locations)
    if len(locations) != 1:
        raise ModuleRefreshError(f"{package!r} is not a regular package", name=package)
    location = locations[0]
    require_unlinked(location, package)
    require_source_tree(location, package)
    require_acyclic(static_closure(package, root))
    _purge_package_bytecode(location)
    for name in [name for name in sys.modules if within(name, package)]:
        del sys.modules[name]
    guard.family = (package, location)
    fresh = importlib.import_module(root)
    _require_confined_search_paths(package, location)
    return fresh


def _namespaces(modules: dict[str, ModuleType]) -> dict[str, dict[str, object]]:
    """A shallow copy of every module's own ``__dict__`` (read directly, so a module-level
    ``__getattr__`` cannot manufacture a binding), keyed by module name."""
    return {
        name: dict(module.__dict__) for name, module in modules.items()
        if isinstance(module, ModuleType)
    }


def _restore_modules(
        before: dict[str, ModuleType], namespaces: dict[str, dict[str, object]],
        bindings: dict[str, tuple[ModuleType, object]]) -> None:
    """Reinstate exactly ``before``'s ``sys.modules`` entries (dropping every entry the attempt
    introduced or replaced), then restore each affected module's direct parent binding to its
    snapshotted state: the previous value, or absence."""
    affected = {
        name for name in {*sys.modules, *before}
        if sys.modules.get(name, _ABSENT) is not before.get(name, _ABSENT)
    }
    for name in affected:
        sys.modules.pop(name, None)
        if name in before:
            sys.modules[name] = before[name]
    for name in affected:
        parent_name, _, child = name.rpartition(".")
        holder = sys.modules.get(parent_name) if parent_name else None
        if not isinstance(holder, ModuleType) or parent_name not in namespaces:
            continue  # a parent the attempt introduced was discarded with its bindings
        _rebind(holder, child, namespaces[parent_name].get(child, _ABSENT))
    for name, (holder, previous) in bindings.items():
        _rebind(holder, name.rpartition(".")[2], previous)


def _rebind(holder: ModuleType, child: str, previous: object) -> None:
    if previous is _ABSENT:
        holder.__dict__.pop(child, None)
    else:
        holder.__dict__[child] = previous


def _purge_package_bytecode(location: str) -> None:
    """Unlink the cached bytecode of EVERY source file under the package, so no module the refresh
    imports (including one no process has loaded yet) can execute a stale timestamp-valid cache."""
    if sys.implementation.cache_tag is None:
        return
    for directory, _subdirs, files in os.walk(location):
        for file in files:
            if file.endswith(".py"):
                source = os.path.join(directory, file)
                Path(importlib.util.cache_from_source(source)).unlink(missing_ok=True)
