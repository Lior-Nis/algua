from __future__ import annotations

import importlib
import inspect
import pkgutil

from algua.portfolio.construction import (
    ConstructionError,
    get_construction_policy,
    validate_construction_params,
)
from algua.strategies import examples
from algua.strategies.base import LoadedStrategy, StrategyConfig


class StrategyNotFound(LookupError):
    pass


def load_strategy(name: str) -> LoadedStrategy:
    """Load a bundled strategy module by name; it must expose CONFIG + signal, and CONFIG must name
    a known construction policy with valid params. Optional `signal_panel` is the vectorized twin."""
    try:
        module = importlib.import_module(f"algua.strategies.examples.{name}")
    except ModuleNotFoundError as exc:
        raise StrategyNotFound(name) from exc
    if not hasattr(module, "CONFIG") or not hasattr(module, "signal"):
        raise StrategyNotFound(f"{name} is missing CONFIG or signal")

    config = module.CONFIG
    try:
        construct_fn = get_construction_policy(config.construction)
        validate_construction_params(config.construction, config.construction_params)
    except ConstructionError as exc:
        raise StrategyNotFound(f"{name}: {exc}") from exc

    panel_fn = getattr(module, "signal_panel", None)
    if panel_fn is not None and not callable(panel_fn):
        raise StrategyNotFound(
            f"{name}.signal_panel is not callable (got {type(panel_fn).__name__})"
        )

    needs_fundamentals = bool(getattr(config, "needs_fundamentals", False))
    n_params = len(inspect.signature(module.signal).parameters)
    if needs_fundamentals:
        if panel_fn is not None:
            raise StrategyNotFound(
                f"{name}: signal_panel is not supported with needs_fundamentals "
                f"(no vectorized fundamentals fast path yet)"
            )
        if n_params != 3:
            raise StrategyNotFound(
                f"{name}: needs_fundamentals=True requires signal(view, params, fundamentals); "
                f"got {n_params} params"
            )
        return LoadedStrategy(
            config=config, fundamentals_signal_fn=module.signal, construct_fn=construct_fn
        )

    if n_params != 2:
        raise StrategyNotFound(f"{name}: signal must take (view, params); got {n_params} params")
    return LoadedStrategy(
        config=config, signal_fn=module.signal, signal_panel_fn=panel_fn, construct_fn=construct_fn
    )


def list_strategies() -> list[str]:
    return [m.name for m in pkgutil.iter_modules(examples.__path__) if not m.name.startswith("_")]


def _loaded_for_test(config: StrategyConfig) -> LoadedStrategy:
    """Test-only: build a LoadedStrategy from a config with a trivial signal + its resolved policy.
    Used by config_hash tests that should not depend on a real example module."""
    import pandas as pd
    fn = get_construction_policy(config.construction)
    return LoadedStrategy(
        config=config, signal_fn=lambda view, params: pd.Series(dtype="float64"), construct_fn=fn
    )
