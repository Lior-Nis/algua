"""Re-execute an already-imported package closure from its CURRENT source (stdlib only).

``importlib.reload`` alone is not a faithful refresh: it trusts timestamp-validated cached
bytecode (a same-size, same-mtime edit keeps a stale ``.pyc`` valid), and reloading in
``sys.modules`` order is not dependency-first (that order records when each module last finished
executing, not what its current source imports), so a module can re-bind a stale value from a
dependency that has not been reloaded yet.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from pathlib import Path


def refresh_package_closure(package: str, last: str) -> None:
    """Reload every loaded module of ``package`` (the package and its submodules) from current
    source in the dependency-first order of its static imports, then ``last`` (already imported).

    A loaded module with no on-disk source (e.g. a deleted temp module) cannot be reloaded and is
    skipped. Each module's cached bytecode is purged before its reload."""
    members = {
        name: module for name, module in list(sys.modules.items())
        if (name == package or name.startswith(package + ".")) and name != last
        and module is not None and _source_exists(module)
    }
    importlib.invalidate_caches()
    graph = {name: _package_imports(module, set(members)) for name, module in members.items()}
    for name in [*dependency_first(graph), last]:
        _purge_cached_bytecode(sys.modules[name])
        importlib.reload(sys.modules[name])


def dependency_first(graph: dict[str, set[str]]) -> list[str]:
    """Deterministic post-order: every module after the modules it imports (cycles tolerated)."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name not in seen:
            seen.add(name)
            for dependency in sorted(graph[name]):
                visit(dependency)
            order.append(name)

    for name in sorted(graph):
        visit(name)
    return order


def _package_imports(module: object, members: set[str]) -> set[str]:
    """The ``members`` that ``module``'s current source statically imports (any nesting)."""
    spec = getattr(module, "__spec__", None)
    origin, parent = getattr(spec, "origin", ""), getattr(spec, "parent", "")
    targets: set[str] = set()
    for node in ast.walk(ast.parse(Path(origin).read_bytes(), filename=origin)):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = importlib.util.resolve_name("." * node.level + (node.module or ""), parent)
            targets.add(base)
            targets.update(f"{base}.{alias.name}" for alias in node.names)
    return (targets & members) - {getattr(spec, "name", "")}


def _purge_cached_bytecode(module: object) -> None:
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if isinstance(origin, str) and origin.endswith(".py") and sys.implementation.cache_tag:
        Path(importlib.util.cache_from_source(origin)).unlink(missing_ok=True)


def _source_exists(module: object) -> bool:
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if not isinstance(origin, str) or origin in ("built-in", "frozen", "namespace"):
        return False
    return Path(origin).exists()
