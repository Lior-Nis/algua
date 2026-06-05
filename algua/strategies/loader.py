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
    # OPTIONAL vectorized acceleration hook: a module MAY additionally define a module-level
    # `compute_weights_panel(bars, params) -> DataFrame`. It is bound as `panel_fn` (and the engine
    # uses it only behind a fail-closed parity guard). Validate it's callable; reject a non-callable
    # so a typo'd attribute fails loudly rather than silently disabling the fast path.
    panel_fn = getattr(module, "compute_weights_panel", None)
    if panel_fn is not None and not callable(panel_fn):
        raise StrategyNotFound(
            f"{name}.compute_weights_panel is not callable (got {type(panel_fn).__name__})"
        )
    return LoadedStrategy(config=module.CONFIG, fn=module.compute_weights, panel_fn=panel_fn)


def list_strategies() -> list[str]:
    return [m.name for m in pkgutil.iter_modules(examples.__path__) if not m.name.startswith("_")]
