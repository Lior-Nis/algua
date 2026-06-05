from __future__ import annotations

import hashlib
import inspect
import sys
from types import ModuleType

from algua.provenance import lockfile
from algua.registry.repository import ArtifactIdentity, StrategyRepository
from algua.strategies.base import config_hash
from algua.strategies.loader import load_strategy

_FIRST_PARTY_ROOT = "algua"


def compute_artifact_hashes(name: str) -> ArtifactIdentity:
    """Recompute ``(code_hash, config_hash, dependency_hash)`` from the strategy's *actual*
    source, resolved config, and locked dependency set. This is the single function both the
    ``approve`` and the ``transition --to live`` paths call, so the live gate pins the real
    artifact: a constant or caller-supplied hash can no longer satisfy it, because both sides
    derive the identity from the loaded module and the lockfile themselves.

    ``code_hash`` covers the strategy's first-party *dependency closure*, not just its own
    module source: starting from the loaded strategy module we walk the imported ``algua.*``
    modules transitively (first-party only — stdlib/third-party are excluded) and hash their
    sorted source. So a behavior-changing edit to an imported ``algua`` helper the strategy
    relies on invalidates a prior approval, instead of silently slipping past the live gate.

    ``dependency_hash`` pins the locked third-party set (``uv.lock``) via the SAME shared
    function the backtest stamps use, so a lockfile bump that can change fill or numerical
    semantics invalidates a prior approval too — the binding is no longer blind to dependency
    drift.
    """
    loaded = load_strategy(name)
    root = inspect.getmodule(loaded.fn)
    closure = _first_party_closure(root)
    payload = "\n".join(
        f"# module: {mod_name}\n{source}" for mod_name, source in sorted(closure.items())
    )
    code_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return ArtifactIdentity(
        code_hash=code_hash,
        config_hash=config_hash(loaded),
        dependency_hash=lockfile.dependency_hash(),
    )


def _is_first_party(module_name: str | None) -> bool:
    return module_name == _FIRST_PARTY_ROOT or (
        module_name is not None and module_name.startswith(_FIRST_PARTY_ROOT + ".")
    )


def _first_party_closure(root: ModuleType | None) -> dict[str, str]:
    """Map ``module_name -> source`` for every first-party ``algua.*`` module transitively
    reachable from ``root`` via its imported names. Bounded to ``algua.*`` so we never recurse
    into stdlib/third-party trees; deterministic because callers sort by module name."""
    if root is None:
        return {}
    sources: dict[str, str] = {}
    seen: set[str] = set()
    queue: list[ModuleType] = [root]
    while queue:
        module = queue.pop()
        mod_name = getattr(module, "__name__", None)
        if mod_name is None or mod_name in seen or not _is_first_party(mod_name):
            continue
        seen.add(mod_name)
        try:
            sources[mod_name] = inspect.getsource(module)
        except (OSError, TypeError):
            sources[mod_name] = ""
        for dep in _imported_first_party_modules(module):
            if dep.__name__ not in seen:
                queue.append(dep)
    return sources


def _imported_first_party_modules(module: ModuleType) -> list[ModuleType]:
    """First-party module objects referenced by ``module``'s globals: directly imported
    ``algua.*`` modules, plus the defining modules of imported names (so
    ``from algua.x import helper`` pulls in ``algua.x``)."""
    deps: list[ModuleType] = []
    for value in vars(module).values():
        if isinstance(value, ModuleType):
            if _is_first_party(getattr(value, "__name__", None)):
                deps.append(value)
            continue
        owner = getattr(value, "__module__", None)
        if isinstance(owner, str) and _is_first_party(owner):
            resolved = sys.modules.get(owner)
            if resolved is not None:
                deps.append(resolved)
    return deps


def record_approval(repo: StrategyRepository, name: str, approved_by: str) -> int:
    """Record a human approval. The approved identity is computed from the live strategy source,
    config, and locked dependency set, never supplied by the caller, so the approval binds to the
    exact artifact it approves."""
    _require_non_empty("approved_by", approved_by)
    rec = repo.get(name)
    identity = compute_artifact_hashes(name)
    return repo.record_approval(
        rec.id,
        identity.code_hash,
        identity.config_hash,
        identity.dependency_hash,
        approved_by,
    )


def has_valid_approval(
    repo: StrategyRepository,
    strategy_id: int,
    code_hash: str,
    config_hash: str,
    dependency_hash: str | None,
) -> bool:
    return repo.has_valid_approval(strategy_id, code_hash, config_hash, dependency_hash)


def _require_non_empty(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
