from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import sys
from types import ModuleType

from algua.provenance import lockfile
from algua.registry.repository import ApprovalLedger, ApprovalRepository, ArtifactIdentity
from algua.strategies.base import LoadedStrategy, config_hash
from algua.strategies.loader import load_strategy

_FIRST_PARTY_ROOT = "algua"
_CONSTRUCTION_MODULE = "algua.portfolio.construction"
_OVERLAYS_MODULE = "algua.portfolio.overlays"


def _merged_closure_for(loaded: LoadedStrategy) -> dict[str, str]:
    """First-party source closure for a strategy's identity: the union of the closure reachable
    from its authored signal module AND the construction policy module AND the overlays module
    (resolved by NAME, not via the bound callable — getmodule on a partial returns functools). The
    construction module holds every policy + the dispatch table, so a policy-body edit, a helper
    edit, or an id retarget invalidates a prior approval."""
    signal_root = inspect.getmodule(loaded.authored_signal)
    construction_root = importlib.import_module(_CONSTRUCTION_MODULE)
    merged: dict[str, str] = {}
    merged.update(_first_party_closure(signal_root))
    merged.update(_first_party_closure(construction_root))
    merged.update(_first_party_closure(importlib.import_module(_OVERLAYS_MODULE)))
    return merged


def closure_module_names(loaded: LoadedStrategy) -> frozenset[str]:
    """The first-party module names in a strategy's identity closure (signal + construction +
    overlays). This is exactly the key set of the source closure ``compute_artifact_hashes`` hashes,
    so lineage (issue #140) and ``code_hash`` invalidation share ONE definition of a strategy's
    dependencies at the same module granularity. This consumer does NOT change the hash payload."""
    return frozenset(_merged_closure_for(loaded))


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

    The closure is rooted from BOTH the authored signal module AND the construction policy module
    (issue #141), so a portfolio-construction change — a policy-body edit or a CONFIG retarget to a
    different policy — invalidates a prior approval as well.
    """
    loaded = load_strategy(name)
    closure = _merged_closure_for(loaded)
    payload = "\n".join(
        f"# module: {mod_name}\n{source}" for mod_name, source in sorted(closure.items())
    )
    # 128-bit sha256 prefix (#341): widened from 64-bit for collision resistance on the live-gate
    # identity. Recomputed on both the approve and transition-to-live paths, so widening is
    # self-consistent — but any approval minted before the widen must be regenerated to match.
    code_hash = hashlib.sha256(payload.encode()).hexdigest()[:32]
    return ArtifactIdentity(
        code_hash=code_hash,
        config_hash=config_hash(loaded),
        dependency_hash=lockfile.dependency_hash(),
    )


def _is_first_party(module_name: str | None) -> bool:
    return module_name == _FIRST_PARTY_ROOT or (
        module_name is not None and module_name.startswith(_FIRST_PARTY_ROOT + ".")
    )


def _strip_cosmetics(source: str) -> str:
    """Source with comments, formatting and docstrings removed, via an AST round-trip.

    `code_hash` must track BEHAVIOUR, not typography. Hashing raw text meant a comment edit or a
    docstring rewrite reset every strategy's forward-evidence clock, against a gate that needs
    250-500 observations under ONE unchanged identity.

    A second consumer depends on the SAME normalized identity for the opposite reason: the
    forward-evidence epoch bound (`forward_evidence._epoch_start_id`) detects a revert-to-an-
    earlier-artifact attack only because the detour artifact hashes DIFFERENTLY from the one being
    reverted to. Narrowing what this function strips (making MORE edits cosmetic) weakens both
    consumers at once -- it lets more genuine code changes hide behind an unchanged `code_hash`,
    which both keeps a stale live-gate approval valid for changed code and lets a revert's detour
    escape epoch detection.

    Comments are absent from the AST, and `ast.unparse` emits canonical formatting, so both vanish.
    Docstrings survive as `Expr(Constant(str))` and are removed explicitly -- they are the most
    frequently edited text in this repo and cannot change a trading decision. The one runtime value
    this knowingly discards is `__doc__` itself: `algua/features/catalogue.py` reads it via
    `inspect.getdoc` to build `FactorSpec.summary`, and `algua.features.alphas` sits in this
    strategy's closure -- but that path feeds catalogue metadata, not a trading decision, so
    dropping it here is deliberate.

    Unparseable source is returned RAW rather than normalised to "": collapsing it would give every
    broken module one shared identity.

    HAZARD -- `ast.unparse` output is not stable across CPython minor versions: it depends on the
    interpreter's own unparser (e.g. f-string quote selection changed between 3.12 and 3.13 --
    `f'{c['name']}'` vs `f"{c['name']}"`), so the normalized identity is only stable while the
    interpreter is. Re-normalising all 290 first-party modules under 3.13 changes 28 of them versus
    3.12 -- a concrete, not hypothetical, count. The repo-root `.python-version` pin (3.12) is what
    holds `code_hash` stable; a deliberate interpreter bump is a deliberate identity move, exactly
    like a deliberate `uv.lock` bump already is for `dependency_hash`.
    """
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(
                    node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
        return ast.unparse(tree)
    except (SyntaxError, RecursionError, ValueError):
        # RecursionError depends on the remaining C-stack budget AT CALL TIME (frames already
        # consumed by the caller, thread-local stack size, platform), not purely on source shape --
        # so in principle the SAME deeply-nested module could recurse-fail in one checkout/process
        # and unparse cleanly in another, yielding two different `code_hash` values for identical
        # source. This is a MONITORED property, not a dismissed one: measured max AST depth across
        # all 290 first-party modules is 16, against Python's default recursion limit of 1000 --
        # nowhere near the boundary where remaining-stack variance could flip the outcome, so there
        # is no live trigger today. It would need to be revisited if a module's nesting grew by two
        # orders of magnitude.
        return source


def _normalized_source(module: ModuleType) -> str:
    """The module's source, normalised. "" when it is unavailable (a namespace package, a module
    built at runtime) -- the pre-existing contract at this call site."""
    try:
        return _strip_cosmetics(inspect.getsource(module))
    except (OSError, TypeError):
        return ""


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
        sources[mod_name] = _normalized_source(module)
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


def record_approval(repo: ApprovalRepository, name: str, approved_by: str) -> int:
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
    repo: ApprovalLedger,
    strategy_id: int,
    code_hash: str,
    config_hash: str,
    dependency_hash: str | None,
) -> bool:
    return repo.has_valid_approval(strategy_id, code_hash, config_hash, dependency_hash)


def _require_non_empty(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
