from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import algua.strategies as _strategies_pkg
from algua.portfolio.construction import (
    ConstructionError,
    get_construction_policy,
    validate_construction_params,
)
from algua.strategies.base import LoadedStrategy, StrategyConfig


class StrategyNotFound(LookupError):
    pass


def _family_dirs() -> list[Path]:
    """Family subpackages directly under algua/strategies/: a dir with an __init__.py whose name is
    not `_`-prefixed (private/temp). Top-level infra modules (loader.py, base.py, __init__.py) are
    FILES not dirs, so excluded structurally; __pycache__ has no __init__.py so it is too."""
    root = Path(_strategies_pkg.__file__).parent
    return [
        p for p in sorted(root.iterdir())
        if p.is_dir() and not p.name.startswith("_") and (p / "__init__.py").exists()
    ]


def _index() -> dict[str, str]:
    """Map bare strategy name -> dotted module path by walking family dirs on the FILESYSTEM —
    imports nothing. Rebuilt per call (a directory listing is cheap, and tests write temp modules
    after import). `_`-prefixed modules (private/temp) are skipped HERE, not just at listing time —
    so a hidden helper or two temp modules sharing a stem can never make every load fail with a
    spurious duplicate. Fails closed (raises) on a duplicate bare name across families."""
    index: dict[str, str] = {}
    for fam in _family_dirs():
        for mod in pkgutil.iter_modules([str(fam)]):
            if mod.ispkg or mod.name.startswith("_"):
                continue  # sub-subpackages and private/temp modules are not strategies
            dotted = f"algua.strategies.{fam.name}.{mod.name}"
            if mod.name in index:
                raise StrategyNotFound(
                    f"duplicate strategy name {mod.name!r}: {index[mod.name]} and {dotted}"
                )
            index[mod.name] = dotted
    return index


def load_strategy(name: str) -> LoadedStrategy:
    """Load a bundled strategy by bare name; it must expose CONFIG + signal, and CONFIG must name a
    known construction policy with valid params. Optional `signal_panel` is the vectorized twin.
    Resolves the name via the filesystem family index, then imports EXACTLY ONE module."""
    dotted = _index().get(name)
    if dotted is None:
        raise StrategyNotFound(name)
    module = importlib.import_module(dotted)
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
    needs_news = bool(getattr(config, "needs_news", False))
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

    if needs_news:
        if panel_fn is not None:
            raise StrategyNotFound(
                f"{name}: signal_panel is not supported with needs_news "
                f"(no vectorized news fast path yet)"
            )
        if n_params != 3:
            raise StrategyNotFound(
                f"{name}: needs_news=True requires signal(view, params, news); "
                f"got {n_params} params"
            )
        return LoadedStrategy(
            config=config, news_signal_fn=module.signal, construct_fn=construct_fn
        )

    if n_params != 2:
        raise StrategyNotFound(f"{name}: signal must take (view, params); got {n_params} params")
    return LoadedStrategy(
        config=config, signal_fn=module.signal, signal_panel_fn=panel_fn, construct_fn=construct_fn
    )


def list_strategies() -> list[str]:
    """All discoverable strategy names (`_`-prefixed modules/dirs already excluded by `_index`)."""
    return sorted(_index())


def _loaded_for_test(config: StrategyConfig) -> LoadedStrategy:
    """Test-only: build a LoadedStrategy from a config with a trivial signal + its resolved policy.
    Used by config_hash tests that should not depend on a real example module."""
    import pandas as pd
    fn = get_construction_policy(config.construction)
    return LoadedStrategy(
        config=config, signal_fn=lambda view, params: pd.Series(dtype="float64"), construct_fn=fn
    )
