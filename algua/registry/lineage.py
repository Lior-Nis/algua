# algua/registry/lineage.py
from __future__ import annotations

from algua.features.catalogue import FactorSpec, all_factors, load_all_factors
from algua.registry.approvals import closure_module_names
from algua.strategies.loader import load_strategy


def factors_used_by(strategy_name: str) -> list[FactorSpec]:
    """Catalogue factors whose defining module is in this strategy's identity closure. Module
    granular (matches code_hash); best-effort for top-level imports (lazy/dynamic imports escape
    the closure). Raises ``StrategyNotFound`` if the strategy module cannot be loaded."""
    load_all_factors()
    loaded = load_strategy(strategy_name)
    modules = closure_module_names(loaded)
    return [f for f in all_factors() if f.module in modules]
