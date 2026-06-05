from __future__ import annotations

import importlib
import pkgutil

from algua.strategies import examples
from algua.strategies.base import LoadedStrategy


class StrategyNotFound(LookupError):
    pass


def load_strategy(name: str) -> LoadedStrategy:
    """Load a bundled strategy module by name; it must expose CONFIG + compute_weights."""
    try:
        module = importlib.import_module(f"algua.strategies.examples.{name}")
    except ModuleNotFoundError as exc:
        raise StrategyNotFound(name) from exc
    if not hasattr(module, "CONFIG") or not hasattr(module, "compute_weights"):
        raise StrategyNotFound(f"{name} is missing CONFIG or compute_weights")
    return LoadedStrategy(config=module.CONFIG, fn=module.compute_weights)


def list_strategies() -> list[str]:
    return [m.name for m in pkgutil.iter_modules(examples.__path__) if not m.name.startswith("_")]
