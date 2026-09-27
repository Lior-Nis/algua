"""Re-import a package closure from its CURRENT source, all-or-nothing (stdlib only).

``importlib.reload`` is not a faithful refresh: it re-executes into the SAME module dictionary (a
global deleted from source survives, and a helper's removed name is silently re-imported), it
trusts timestamp-validated cached bytecode (a same-size, same-mtime edit keeps a stale ``.pyc``
valid), and a failure part-way leaves a mixed-version closure. The refresh therefore re-imports
the root as FRESH module objects after purging the package's bytecode, and restores the previous
module state if anything fails. A cyclic static import component has no dependency-safe
execution order, and a symlinked source path escapes the package-wide bytecode purge, so both
fail closed before anything is purged or executed.

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
    stay warm. On any failure every module the attempt introduced is dropped (with each parent
    attribute still bound to it), and every previous ``sys.modules`` entry and the ``package``
    parent binding are restored; the previous family objects' dictionaries were never touched.

    The private refresh lock is held from preflight to commit or rollback, so a supported caller
    waits for the final state; other imports must be quiescent (see the module docstring)."""
    if not root.startswith(package + "."):
        raise ValueError(f"{root!r} is not inside package {package!r}")
    with _REFRESH_LOCK:
        return _refresh_locked(package, root)


def _refresh_locked(package: str, root: str) -> ModuleType:
    importlib.invalidate_caches()
    package_spec = importlib.util.find_spec(package)
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise ModuleRefreshError(f"{package!r} is not a package", name=package)
    locations = list(package_spec.submodule_search_locations)
    for location in locations:
        _require_unlinked(location, package)
    _require_acyclic(_static_closure(package, root))
    _purge_package_bytecode(locations)

    before = dict(sys.modules)
    parent_name, _, child = package.rpartition(".")
    parent = sys.modules.get(parent_name) if parent_name else None
    parent_binding = parent.__dict__.get(child, _ABSENT) if parent is not None else _ABSENT
    for name in [name for name in before if _within(name, package)]:
        del sys.modules[name]
    try:
        return importlib.import_module(root)
    except BaseException:
        _restore_modules(before)
        if parent is not None:
            if parent_binding is _ABSENT:
                parent.__dict__.pop(child, None)
            else:
                parent.__dict__[child] = parent_binding
        raise


def _restore_modules(before: dict[str, ModuleType]) -> None:
    """Drop every ``sys.modules`` entry that differs from ``before`` (plus any parent attribute
    still bound to such a discarded object), then reinstate ``before``'s entries."""
    discarded = {
        name: module for name, module in list(sys.modules.items())
        if before.get(name, _ABSENT) is not module
    }
    for name in discarded:
        sys.modules.pop(name, None)
    for name, module in before.items():
        if sys.modules.get(name, _ABSENT) is not module:
            sys.modules[name] = module
    for name, module in discarded.items():
        parent_name, _, child = name.rpartition(".")
        holder = sys.modules.get(parent_name) if parent_name else None
        if isinstance(holder, ModuleType) and holder.__dict__.get(child, _ABSENT) is module:
            del holder.__dict__[child]


def _require_unlinked(path: str, name: str) -> None:
    """Fail closed if ``path``'s lexical form, or any existing ancestor of it, is a symlink."""
    lexical = Path(os.path.abspath(path))
    if any(candidate.is_symlink() for candidate in (lexical, *lexical.parents)):
        raise ModuleRefreshError(f"{name!r} source lies at or under a symlink", name=name)


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
