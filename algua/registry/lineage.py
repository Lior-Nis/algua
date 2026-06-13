# algua/registry/lineage.py
from __future__ import annotations

from dataclasses import dataclass

from algua.features.catalogue import FactorSpec, all_factors, get_factor, load_all_factors
from algua.registry.approvals import closure_module_names
from algua.registry.repository import StrategyRepository
from algua.strategies.loader import load_strategy


@dataclass
class Dependents:
    """Result of a blast-radius query. ``unloadable`` lists registered strategies whose module
    failed to load (reported, never silently dropped)."""

    factor: str
    dependents: list[str]
    unloadable: list[dict[str, str]]


def factors_used_by(strategy_name: str) -> list[FactorSpec]:
    """Catalogue factors whose defining module is in this strategy's identity closure. Module
    granular (matches code_hash); best-effort for top-level imports (lazy/dynamic imports escape
    the closure). Raises ``StrategyNotFound`` if the strategy module cannot be loaded."""
    load_all_factors()
    loaded = load_strategy(strategy_name)
    modules = closure_module_names(loaded)
    return [f for f in all_factors() if f.module in modules]


def dependents_of(repo: StrategyRepository, factor_name: str) -> Dependents:
    """Registered strategies whose identity closure reaches ``factor_name``'s module. Iterates the
    registry (not just filesystem-discoverable modules) so a registered strategy cannot silently
    vanish from blast radius; a strategy that fails to load lands in ``unloadable`` rather than
    being dropped. Raises ``FactorNotFound`` for an unknown factor."""
    spec = get_factor(factor_name)
    dependents: list[str] = []
    unloadable: list[dict[str, str]] = []
    for rec in repo.list_strategies():
        try:
            loaded = load_strategy(rec.name)
        except Exception as exc:  # noqa: BLE001 - any load failure is reported, not raised
            unloadable.append({"name": rec.name, "error": str(exc)})
            continue
        if spec.module in closure_module_names(loaded):
            dependents.append(rec.name)
    return Dependents(factor=factor_name, dependents=sorted(dependents), unloadable=unloadable)
