"""Static, NON-EXECUTING validation of a package's current source tree (stdlib only).

The warm refresh in ``module_refresh`` executes nothing until this preflight passes: every search
location and reachable source is free of raw ``..`` components and symlinks, the complete tree
holds no symlink or importable non-source module, and the current source's static family closure
is acyclic. Nothing here imports, purges or mutates module state.
"""
from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import os
from collections.abc import Iterator
from importlib.machinery import ModuleSpec
from pathlib import Path


class ModuleRefreshError(ImportError):
    """The package closure cannot be refreshed safely from its current source."""


# Loader suffixes that execute something other than validated source (sourceless or compiled).
_NON_SOURCE_SUFFIXES = (
    *importlib.machinery.BYTECODE_SUFFIXES, *importlib.machinery.EXTENSION_SUFFIXES)


def require_unlinked(path: str, name: str) -> None:
    """Fail closed if ``path`` has a raw ``..`` component (normalizing ``link/..`` would erase the
    link before it is inspected), or if its lexical form or any existing ancestor is a symlink."""
    if os.pardir in Path(path).parts:
        raise ModuleRefreshError(f"{name!r} source path contains a parent traversal", name=name)
    lexical = Path(os.path.abspath(path))
    if any(candidate.is_symlink() for candidate in (lexical, *lexical.parents)):
        raise ModuleRefreshError(f"{name!r} source lies at or under a symlink", name=name)


def require_source_tree(location: str, name: str) -> None:
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


def within(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


def static_closure(package: str, root: str) -> dict[str, set[str]]:
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
            require_unlinked(origin, name)
        if not isinstance(origin, str) or not origin.endswith(".py") or not Path(origin).is_file():
            raise ModuleRefreshError(f"{name!r} is not a Python source module", name=name)
        candidates = _source_imports(origin, spec.parent or "")
        if name != package:
            candidates.add(name.rpartition(".")[0])
        graph[name] = {
            target for target in candidates
            if within(target, package) and target != name and resolve(target) is not None
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


def require_acyclic(graph: dict[str, set[str]]) -> None:
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
