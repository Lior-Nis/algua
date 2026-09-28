"""Static, NON-EXECUTING validation of a package's current source tree (stdlib only).

The warm refresh in ``module_refresh`` executes nothing until this preflight passes: every search
location and reachable source is free of raw ``..`` components and symlinks, the complete tree
holds no symlink or importable non-source module, and the current source's static family closure
is acyclic. Every family spec is CONSTRUCTED here from the exact expected in-root path
(``source_spec``), never obtained from a path hook, a cached path-entry finder or a loaded
module's ``__spec__``, so preflight and execution resolve the same source. Nothing here imports,
purges or mutates module state.
"""
from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import os
import stat
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
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
    with _inspecting(name):
        linked = any(candidate.is_symlink() for candidate in (lexical, *lexical.parents))
    if linked:
        raise ModuleRefreshError(f"{name!r} source lies at or under a symlink", name=name)


def require_source_tree(location: str, name: str) -> None:
    """Fail closed if ANY entry of the complete tree under ``location`` is a symlink (file,
    directory or dangling), a non-regular node (FIFO, socket or device) or an importable
    non-source module, scanning without following links: a dynamic import can reach any file,
    not only the static closure, the bytecode purge never descends into a link, reading a special
    node can block or yield unsupported content, and sourceless bytecode or an extension is never
    current source."""
    with _inspecting(name):
        _scan_tree(location, name)


def _scan_tree(location: str, name: str) -> None:
    pending = [location]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ModuleRefreshError(
                        f"{name!r} source tree contains a symlink: {entry.name!r}", name=name)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                elif not entry.is_file(follow_symlinks=False):
                    raise ModuleRefreshError(
                        f"{name!r} source tree contains a non-regular node: {entry.name!r}",
                        name=name)
                elif _importable_non_source(entry.name):
                    raise ModuleRefreshError(
                        f"{name!r} source tree contains an importable non-source module: "
                        f"{entry.name!r}", name=name)


@contextmanager
def _inspecting(name: str) -> Iterator[None]:
    """Translate an ``OSError`` raised while inspecting the family tree into the stable refusal,
    naming only the error class and errno, never a host path (the cause is not chained)."""
    try:
        yield
    except OSError as exc:
        raise ModuleRefreshError(
            f"{name!r} source tree cannot be inspected ({type(exc).__name__}, errno {exc.errno})",
            name=name) from None


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


def package_location(package: str, entries: Iterable[object]) -> str:
    """The directory of the regular source package ``package`` in its parent's search ``entries``
    (``sys.path`` for a top-level package): the first entry holding the package directory or a
    same-named module decides, as for the default finder, and only a regular source package with
    no raw ``..`` component or symlink is admissible. A relative entry is frozen against the
    working directory NOW, before anything executes, and the location is returned absolute, so
    refreshed code that changes directory cannot redirect preflight or any later resolution."""
    leaf = package.rpartition(".")[2]
    for entry in entries:
        if not isinstance(entry, str):
            continue
        base = os.path.join(_absolute(entry, package), leaf)
        if os.path.isdir(base) or os.path.lexists(base + ".py"):
            require_unlinked(base, package)
            if os.path.isdir(base) and _is_regular(os.path.join(base, "__init__.py"), package):
                return os.path.normpath(base)  # no raw ``..`` remains, so this is lexical only
            break
    raise ModuleRefreshError(f"{package!r} is not a regular package", name=package)


def _absolute(entry: str, name: str) -> str:
    """``entry`` joined to the current working directory unless already absolute, WITHOUT
    normalizing, so a raw ``..`` component is still refused afterwards."""
    if os.path.isabs(entry):
        return entry
    with _inspecting(name):
        return os.path.join(os.getcwd(), entry)


def source_spec(package: str, location: str, name: str) -> ModuleSpec | None:
    """The exact spec of family module ``name`` under the prevalidated root ``location``."""
    return _spec_at(os.path.join(location, *name.split(".")[package.count(".") + 1:]), name)


def _spec_at(base: str, name: str) -> ModuleSpec | None:
    """A source spec for ``name`` at ``base``, ordered as the default finder orders them: a regular
    package ``base/__init__.py`` (searching exactly ``base``) before a module ``base.py``. Absent
    is None; a namespace directory or a non-regular source node fails closed."""
    is_directory = stat.S_ISDIR(_mode(base, name) or 0)
    init = os.path.join(base, "__init__.py")
    if is_directory and _is_regular(init, name):
        return _source(name, init, [base])
    if _is_regular(base + ".py", name):
        return _source(name, base + ".py", None)
    if is_directory:
        raise ModuleRefreshError(f"{name!r} is not a Python source module", name=name)
    return None


def _source(name: str, origin: str, search: list[str] | None) -> ModuleSpec:
    loader = importlib.machinery.SourceFileLoader(name, origin)
    spec = importlib.util.spec_from_file_location(
        name, origin, loader=loader, submodule_search_locations=search)
    assert spec is not None  # a loader was given, so a spec is always built
    return spec


def _mode(path: str, name: str) -> int | None:
    """``path``'s own file type (a final symlink is not followed), or None when it is absent."""
    with _inspecting(name):
        try:
            return os.stat(path, follow_symlinks=False).st_mode
        except (FileNotFoundError, NotADirectoryError):
            return None


def _is_regular(path: str, name: str) -> bool:
    """Whether a source node exists at ``path``; one that is not a regular file fails closed."""
    mode = _mode(path, name)
    if mode is not None and not stat.S_ISREG(mode):
        raise ModuleRefreshError(f"{name!r} source is a non-regular node", name=name)
    return mode is not None


def static_closure(package: str, location: str, root: str) -> dict[str, set[str]]:
    """The ``package`` source modules under ``location`` reachable from ``root``'s current source,
    each mapped to the ``package`` modules it statically imports (at any nesting) plus its own
    parent package."""
    specs: dict[str, ModuleSpec | None] = {}

    def resolve(name: str) -> ModuleSpec | None:
        if name not in specs:
            parent = None if name == package else resolve(name.rpartition(".")[0])
            specs[name] = (
                source_spec(package, location, name)
                if name == package or (parent and parent.submodule_search_locations is not None)
                else None
            )
        return specs[name]

    graph: dict[str, set[str]] = {}
    pending = [root]
    while pending:
        name = pending.pop()
        if name in graph:
            continue
        spec = resolve(name)
        if spec is None or not isinstance(spec.origin, str):
            raise ModuleRefreshError(f"{name!r} has no importable source", name=name)
        require_unlinked(spec.origin, name)
        candidates = _source_imports(_parse(spec.origin, name), spec.parent or "")
        if name != package:
            candidates.add(name.rpartition(".")[0])
        graph[name] = {
            target for target in candidates
            if within(target, package) and target != name and resolve(target) is not None
        }
        pending.extend(sorted(graph[name]))
    return graph


def _parse(origin: str, name: str) -> ast.Module:
    """Parse ``name``'s current source. A read or parse failure is the stable bounded refusal
    naming only the error class and an errno or line, never a host path (the cause is not
    chained). The source is opened without blocking or following a link and must be a regular
    file, so a node swapped in after the scan cannot block the read."""
    with _inspecting(name):
        descriptor = os.open(origin, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
        with open(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ModuleRefreshError(f"{name!r} source is a non-regular node", name=name)
            source = handle.read()
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        line = getattr(exc, "lineno", None)
        raise ModuleRefreshError(
            f"{name!r} source cannot be parsed ({type(exc).__name__}, line {line})",
            name=name) from None


def _source_imports(tree: ast.Module, parent: str) -> set[str]:
    """Every module name ``tree`` statically imports, relative forms resolved."""
    targets: set[str] = set()
    for node in ast.walk(tree):
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
