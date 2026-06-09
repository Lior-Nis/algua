from __future__ import annotations

import importlib
import inspect
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

    panel_fn = getattr(module, "compute_weights_panel", None)
    if panel_fn is not None and not callable(panel_fn):
        raise StrategyNotFound(
            f"{name}.compute_weights_panel is not callable (got {type(panel_fn).__name__})"
        )

    needs_fundamentals = bool(getattr(module.CONFIG, "needs_fundamentals", False))
    n_params = len(inspect.signature(module.compute_weights).parameters)
    if needs_fundamentals:
        # The fundamentals lane forces the per-bar loop (no vectorized fast path yet) and needs a
        # 3-arg signature. Reject the panel hook + a wrong arity, loudly, at load time.
        if panel_fn is not None:
            raise StrategyNotFound(
                f"{name}: compute_weights_panel is not supported with needs_fundamentals "
                f"(no vectorized fundamentals fast path yet)"
            )
        if n_params != 3:
            raise StrategyNotFound(
                f"{name}: needs_fundamentals=True requires compute_weights(view, params, "
                f"fundamentals); got {n_params} params"
            )
        return LoadedStrategy(config=module.CONFIG, fundamentals_fn=module.compute_weights)

    if n_params != 2:
        raise StrategyNotFound(
            f"{name}: compute_weights must take (view, params); got {n_params} params"
        )
    return LoadedStrategy(config=module.CONFIG, fn=module.compute_weights, panel_fn=panel_fn)


def list_strategies() -> list[str]:
    return [m.name for m in pkgutil.iter_modules(examples.__path__) if not m.name.startswith("_")]
