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
root only, so a fresh ``__init__`` cannot extend ``__path__`` into an unscanned tree. The family
location is found on the filesystem from the parent's search path and every family spec is built
from its exact in-root source path, so no path hook, cached path-entry finder or stale loaded
``__spec__`` can make preflight inspect one tree while execution loads another.

Concurrency: the complete transaction runs under a private module lock shared ONLY by the
supported Algua callers (``refresh_package_closure`` and ``serialized_import``, which the strategy
loader uses for every cold, warm and reload path). It deliberately does not take CPython's global
import lock, whose ordering against per-module import locks deadlocks with an import already in
progress. This is not a claim of arbitrary ``importlib`` concurrency safety: the process must be
import-quiescent for everything else. A direct import of a family module (or of anything the
refresh imports) from another thread, or a supported call made from inside a module body another
thread is initializing, while a refresh runs is unsupported and may observe a partially rebuilt
family or deadlock. In a forked child, a transaction the forking thread owns keeps its lock
until that thread commits or rolls back; one a vanished thread owned is undone and the lock
replaced, but CPython's per-module import locks that thread held are not reset, so re-importing
exactly the modules it was initializing there is unsupported.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import threading
from importlib.machinery import ModuleSpec, SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

from algua.primitives.module_source_scan import (
    ModuleRefreshError,
    package_location,
    require_acyclic,
    require_source_tree,
    source_spec,
    static_closure,
    within,
)

__all__ = ["ModuleRefreshError", "refresh_package_closure", "serialized_import"]

_ABSENT = object()
# Re-entrant: a supported caller that runs inside a refresh on the same thread must not self-block.
_REFRESH_LOCK = threading.RLock()
# Per-thread: set while THIS thread runs a refresh transaction, so a nested refresh is refused.
_TRANSACTION = threading.local()
# Process-wide (owner ident, pre-refresh ``sys.modules``, guard) of the running transaction (fork).
_ACTIVE: tuple[int, dict[str, ModuleType], _ImportGuard] | None = None


def _recover_after_fork() -> None:
    """Only the forking thread survives in a child. A transaction it owns keeps its lock, so that
    thread commits or rolls back as usual while any other child thread waits. Otherwise a lock
    held by another thread would never be released, so the child gets a fresh lock; and a
    transaction another thread was running will never finish, so the child restores its
    pre-refresh modules and direct parent bindings, drops the guard and resets the transaction."""
    global _ACTIVE, _REFRESH_LOCK, _TRANSACTION
    if _ACTIVE is not None and _ACTIVE[0] == threading.get_ident():
        return
    _REFRESH_LOCK = threading.RLock()
    if _ACTIVE is None:
        return
    _, before, guard = _ACTIVE
    _ACTIVE, _TRANSACTION = None, threading.local()
    sys.meta_path[:] = [finder for finder in sys.meta_path if not isinstance(finder, _ImportGuard)]
    _restore_modules(before, guard.bindings)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_recover_after_fork)


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
    waits for the final state; other imports must be quiescent (see the module docstring). A
    refresh started on the same thread from inside a running transaction fails closed, while a
    same-thread ``serialized_import`` stays re-entrant."""
    if not root.startswith(package + "."):
        raise ValueError(f"{root!r} is not inside package {package!r}")
    if getattr(_TRANSACTION, "active", False):
        raise ModuleRefreshError(
            f"nested refresh of {package!r} inside a running refresh transaction", name=root)
    with _REFRESH_LOCK:
        _TRANSACTION.active = True
        try:
            return _refresh_locked(package, root)
        finally:
            _TRANSACTION.active = False


def _refresh_locked(package: str, root: str) -> ModuleType:
    """Snapshot module state BEFORE package discovery (``find_spec`` imports cold parents), so any
    failure from preflight to the fresh import, including a ``BaseException``, restores it. The
    snapshot is ``sys.modules`` itself plus, recorded by the guard, the direct parent binding of
    each module the import system loads; no module namespace is copied wholesale."""
    global _ACTIVE
    before, guard = dict(sys.modules), _ImportGuard()
    _ACTIVE = (threading.get_ident(), before, guard)
    sys.meta_path.insert(0, guard)
    try:
        return _refresh_attempt(package, root, guard)
    except BaseException:
        _restore_modules(before, guard.bindings)
        raise
    finally:
        if guard in sys.meta_path:
            sys.meta_path.remove(guard)
        _ACTIVE = None


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
        self.specs: dict[str, _Facts] = {}  # every family spec handed out, with its facts

    def find_spec(
            self, fullname: str, path: object = None, target: object = None) -> ModuleSpec | None:
        parent_name, _, child = fullname.rpartition(".")
        holder = sys.modules.get(parent_name) if parent_name else None
        if fullname not in self.bindings and isinstance(holder, ModuleType):
            self.bindings[fullname] = (holder, holder.__dict__.get(child, _ABSENT))
        if self.family is None or not within(fullname, self.family[0]):
            return None
        package, location = self.family
        if fullname != package:
            search = _exact_search_path(package, location, parent_name)
            if path != search:
                raise ModuleRefreshError(
                    f"{parent_name!r} search path deviates from its prevalidated family root",
                    name=fullname)
        spec = source_spec(package, location, fullname)
        if spec is None:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        found = spec.submodule_search_locations
        self.specs[fullname] = _Facts(
            spec, fullname, spec.origin, spec.cached, None if found is None else tuple(found))
        return spec


def _exact_search_path(package: str, location: str, name: str) -> list[str]:
    """The only search path a family package ``name`` may have: its directory under the root."""
    return [os.path.join(location, *name.split(".")[package.count(".") + 1:])]


class _Facts(NamedTuple):
    """A source spec's facts, recorded when handed out: executed code can mutate it in place."""
    spec: ModuleSpec
    name: str
    origin: str | None
    cached: str | None
    search: tuple[str, ...] | None


def _require_committed_family(package: str, specs: dict[str, _Facts]) -> None:
    """Before commit, every family entry in ``sys.modules`` and every module the guard handed out
    is a ``ModuleType`` carrying that exact source spec, bound identically in ``sys.modules`` and
    on its direct parent, and the spec and module still carry its recorded facts (``_intact``)."""
    for name in sorted({*specs, *(name for name in sys.modules if within(name, package))}):
        module, facts = sys.modules.get(name), specs.get(name)
        parent_name, _, child = name.rpartition(".")
        holder = sys.modules.get(parent_name) if parent_name else None
        if not (
            facts is not None and isinstance(module, ModuleType)
            and vars(module).get("__spec__") is facts.spec
            and (isinstance(holder, ModuleType) or not parent_name)
            and (holder is None or vars(holder).get(child) is module)
            and _intact(module, facts)
        ):
            raise ModuleRefreshError(
                f"{name!r} is not bound as its fresh source module with its resolved spec, "
                "metadata and search path at commit", name=name)


def _intact(module: ModuleType, facts: _Facts) -> bool:
    """Whether ``facts.spec`` is still an exact ``SourceFileLoader`` spec with the recorded loader
    name and path, name, origin, cache and search locations, and ``module`` still has the matching
    ``__name__``, ``__package__``, ``__loader__``, ``__file__``, ``__cached__`` and ``__path__``."""
    spec, attrs, loader = facts.spec, vars(module), facts.spec.loader
    if type(loader) is not SourceFileLoader or attrs.get("__loader__") is not loader:
        return False  # exactly the stdlib source loader (never a subclass), shared by the module
    search = None if facts.search is None else list(facts.search)
    return all(_exact(value, expected) for value, expected in (
        (getattr(loader, "name", None), facts.name), (getattr(loader, "path", None), facts.origin),
        (spec.name, facts.name), (spec.origin, facts.origin), (spec.cached, facts.cached),
        (spec.submodule_search_locations, search), (attrs.get("__name__"), facts.name),
        (attrs.get("__package__"), facts.name if search else facts.name.rpartition(".")[0]),
        (attrs.get("__file__"), facts.origin), (attrs.get("__cached__"), facts.cached),
        (attrs.get("__path__", _ABSENT), _ABSENT if search is None else search)))


def _exact(value: object, expected: object) -> bool:
    """Equal with the exact expected type at every level, so a permissive ``__eq__`` cannot pass."""
    if type(value) is not type(expected):
        return False
    if isinstance(value, list) and isinstance(expected, list):
        return len(value) == len(expected) and all(map(_exact, value, expected))
    return value == expected


def _refresh_attempt(package: str, root: str, guard: _ImportGuard) -> ModuleType:
    importlib.invalidate_caches()
    parent = package.rpartition(".")[0]
    entries = vars(importlib.import_module(parent)).get("__path__", ()) if parent else sys.path
    location = package_location(package, entries)
    require_source_tree(location, package)
    require_acyclic(static_closure(package, location, root))
    _purge_package_bytecode(location, package)
    for name in [name for name in sys.modules if within(name, package)]:
        del sys.modules[name]
    guard.family = (package, location)
    fresh = importlib.import_module(root)
    _require_committed_family(package, guard.specs)
    return fresh


def _restore_modules(
        before: dict[str, ModuleType], bindings: dict[str, tuple[ModuleType, object]]) -> None:
    """Reinstate exactly ``before``'s ``sys.modules`` entries (dropping every entry the attempt
    introduced or replaced), then restore every direct parent binding an import changed to its
    recorded state: the previous value, or absence."""
    affected = {
        name for name in {*sys.modules, *before}
        if sys.modules.get(name, _ABSENT) is not before.get(name, _ABSENT)
    }
    for name in affected:
        sys.modules.pop(name, None)
        if name in before:
            sys.modules[name] = before[name]
    for name, (holder, previous) in bindings.items():
        child = name.rpartition(".")[2]
        if previous is _ABSENT:
            holder.__dict__.pop(child, None)
        else:
            holder.__dict__[child] = previous


def _purge_package_bytecode(location: str, package: str) -> None:
    """Unlink the cached bytecode of EVERY source file under the package, so no module the refresh
    imports (including one no process has loaded yet) can execute a stale timestamp-valid cache.
    It runs before any family entry is dropped, so a failure part-way has deleted only caches; it
    is the stable bounded refusal naming the error class and errno, never a host path."""
    if sys.implementation.cache_tag is None:
        return

    def fail(exc: OSError) -> None:
        raise exc

    try:
        for directory, _subdirs, files in os.walk(location, onerror=fail):
            for file in files:
                if file.endswith(".py"):
                    source = os.path.join(directory, file)
                    Path(importlib.util.cache_from_source(source)).unlink(missing_ok=True)
    except OSError as exc:
        raise ModuleRefreshError(
            f"{package!r} bytecode cache cannot be purged ({type(exc).__name__}, errno "
            f"{exc.errno})", name=package) from None
