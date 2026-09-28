"""Commit-time validation of a freshly imported package family (stdlib only).

The refresh in ``module_refresh`` records the facts of every source spec its import guard hands to
the import system. Executed family code can rebind, replace or mutate those specs and modules in
place, so before the transaction commits every family module must still be bound exactly as its
fresh source module, and every spec and module must still carry the recorded facts. Nothing here
imports, purges or mutates module state.
"""
from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec, SourceFileLoader
from types import ModuleType
from typing import NamedTuple

from algua.primitives.module_source_scan import ModuleRefreshError, within

_ABSENT = object()


class SourceSpecFacts(NamedTuple):
    """A source spec's facts, recorded when handed out: executed code can mutate it in place."""
    spec: ModuleSpec
    name: str
    origin: str | None
    cached: str | None
    search: tuple[str, ...] | None


def require_committed_family(package: str, specs: dict[str, SourceSpecFacts]) -> None:
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


def _intact(module: ModuleType, facts: SourceSpecFacts) -> bool:
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
