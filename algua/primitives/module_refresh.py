"""Re-import a package closure from its CURRENT source, all-or-nothing (stdlib only).

``importlib.reload`` is not a faithful refresh: it re-executes into the SAME module dictionary (a
global deleted from source survives, and a helper's removed name is silently re-imported), it
trusts timestamp-validated cached bytecode (a same-size, same-mtime edit keeps a stale ``.pyc``
valid), and a failure part-way leaves a mixed-version closure. The refresh therefore re-imports
the root as FRESH module objects after purging the package's bytecode, and restores the previous
module state if anything fails. A cyclic static import component has no dependency-safe
execution order, and a symlink anywhere in the package tree (which a dynamic import can reach
even outside the static closure) escapes the package-wide bytecode purge, so both fail closed
before anything is purged or executed.

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

import ast
import importlib
import importlib.machinery
import importlib.util
import os
import sys
import threading
from collections.abc import Iterator
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType


class ModuleRefreshError(ImportError):
    """The package closure cannot be refreshed safely from its current source."""


_ABSENT = object()
# Loader suffixes that execute something other than validated source (sourceless or compiled).
_NON_SOURCE_SUFFIXES = (
    *importlib.machinery.BYTECODE_SUFFIXES, *importlib.machinery.EXTENSION_SUFFIXES)
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
        return _refresh_attempt(package, root)
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
    removed the child's own ``sys.modules`` entry."""

    def __init__(self) -> None:
        self.bindings: dict[str, tuple[ModuleType, object]] = {}

    def find_spec(
            self, fullname: str, path: object = None, target: object = None) -> ModuleSpec | None:
        parent_name, _, child = fullname.rpartition(".")
        holder = sys.modules.get(parent_name) if parent_name else None
        if fullname not in self.bindings and isinstance(holder, ModuleType):
            self.bindings[fullname] = (holder, holder.__dict__.get(child, _ABSENT))
        return None


def _refresh_attempt(package: str, root: str) -> ModuleType:
    importlib.invalidate_caches()
    package_spec = importlib.util.find_spec(package)
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise ModuleRefreshError(f"{package!r} is not a package", name=package)
    locations = list(package_spec.submodule_search_locations)
    for location in locations:
        _require_unlinked(location, package)
        _require_source_tree(location, package)
    _require_acyclic(_static_closure(package, root))
    _purge_package_bytecode(locations)
    for name in [name for name in sys.modules if _within(name, package)]:
        del sys.modules[name]
    return importlib.import_module(root)


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


def _require_unlinked(path: str, name: str) -> None:
    """Fail closed if ``path`` has a raw ``..`` component (normalizing ``link/..`` would erase the
    link before it is inspected), or if its lexical form or any existing ancestor is a symlink."""
    if os.pardir in Path(path).parts:
        raise ModuleRefreshError(f"{name!r} source path contains a parent traversal", name=name)
    lexical = Path(os.path.abspath(path))
    if any(candidate.is_symlink() for candidate in (lexical, *lexical.parents)):
        raise ModuleRefreshError(f"{name!r} source lies at or under a symlink", name=name)


def _require_source_tree(location: str, name: str) -> None:
    """Fail closed if ANY entry of the complete tree under ``location`` is a symlink (file,
    directory or dangling) or an importable non-source module, scanning without following links:
    a dynamic import can reach any file, not only the static closure, the bytecode purge never
    descends into a link, and sourceless bytecode or an extension is never current source."""
    pending = [location]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ModuleRefreshError(
                        f"{name!r} source tree contains a symlink: {entry.name!r}", name=name)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                elif _importable_non_source(entry.name):
                    raise ModuleRefreshError(
                        f"{name!r} source tree contains an importable non-source module: "
                        f"{entry.name!r}", name=name)


def _importable_non_source(file_name: str) -> bool:
    """Whether a finder would load ``file_name`` as a module (its stem has no dot) through a
    sourceless or extension loader. Tagged ``__pycache__`` files such as ``m.cpython-312.pyc``
    are not importable by name, so normal cached bytecode stays admissible."""
    return any(
        file_name.endswith(suffix) and "." not in file_name[:-len(suffix)]
        for suffix in _NON_SOURCE_SUFFIXES
    )


def _within(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


def _static_closure(package: str, root: str) -> dict[str, set[str]]:
    """The ``package`` source modules reachable from ``root``'s current source, each mapped to the
    ``package`` modules it statically imports (at any nesting) plus its own parent package."""
    specs: dict[str, ModuleSpec | None] = {}

    def resolve(name: str) -> ModuleSpec | None:
        if name not in specs:
            if name == package:
                specs[name] = importlib.util.find_spec(package)
            else:
                parent = resolve(name.rpartition(".")[0])
                locations = parent.submodule_search_locations if parent is not None else None
                specs[name] = (
                    None if locations is None
                    else importlib.machinery.PathFinder.find_spec(name, list(locations))
                )
        return specs[name]

    graph: dict[str, set[str]] = {}
    pending = [root]
    while pending:
        name = pending.pop()
        if name in graph:
            continue
        spec = resolve(name)
        if spec is None:
            raise ModuleRefreshError(f"{name!r} has no importable source", name=name)
        origin = spec.origin
        if isinstance(origin, str):
            _require_unlinked(origin, name)
        if not isinstance(origin, str) or not origin.endswith(".py") or not Path(origin).is_file():
            raise ModuleRefreshError(f"{name!r} is not a Python source module", name=name)
        candidates = _source_imports(origin, spec.parent or "")
        if name != package:
            candidates.add(name.rpartition(".")[0])
        graph[name] = {
            target for target in candidates
            if _within(target, package) and target != name and resolve(target) is not None
        }
        pending.extend(sorted(graph[name]))
    return graph


def _source_imports(origin: str, parent: str) -> set[str]:
    """Every module name ``origin``'s source statically imports, relative forms resolved."""
    targets: set[str] = set()
    for node in ast.walk(ast.parse(Path(origin).read_bytes(), filename=origin)):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            try:
                base = importlib.util.resolve_name("." * node.level + (node.module or ""), parent)
            except (ImportError, ValueError):
                continue  # beyond the top-level package: the real import fails on its own
            targets.add(base)
            targets.update(f"{base}.{alias.name}" for alias in node.names)
    return targets


def _require_acyclic(graph: dict[str, set[str]]) -> None:
    """Fail closed on any cyclic import component, naming one cycle deterministically. The DFS is
    iterative (explicit ``(node, dependency iterator)`` frames), so depth is not bounded by the
    interpreter recursion limit."""
    done: set[str] = set()
    for start in sorted(graph):
        if start in done:
            continue
        frames: list[tuple[str, Iterator[str]]] = [(start, iter(sorted(graph[start])))]
        on_path = {start}
        while frames:
            name, dependencies = frames[-1]
            dependency = next(dependencies, None)
            if dependency is None:
                frames.pop()
                on_path.discard(name)
                done.add(name)
            elif dependency in on_path:
                path = [frame_name for frame_name, _ in frames]
                cycle = path[path.index(dependency):] + [dependency]
                raise ModuleRefreshError(
                    "cyclic package imports have no dependency-safe order: " + " -> ".join(cycle),
                    name=dependency,
                )
            elif dependency not in done:
                frames.append((dependency, iter(sorted(graph.get(dependency, ())))))
                on_path.add(dependency)


def _purge_package_bytecode(locations: list[str]) -> None:
    """Unlink the cached bytecode of EVERY source file under the package, so no module the refresh
    imports (including one no process has loaded yet) can execute a stale timestamp-valid cache."""
    if sys.implementation.cache_tag is None:
        return
    for location in locations:
        for directory, _subdirs, files in os.walk(location):
            for file in files:
                if file.endswith(".py"):
                    source = os.path.join(directory, file)
                    Path(importlib.util.cache_from_source(source)).unlink(missing_ok=True)
