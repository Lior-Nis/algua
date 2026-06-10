from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import algua.strategies as _strategies_pkg
from algua.strategies.base import LoadedStrategy


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
    """Load a bundled strategy by bare name; it must expose CONFIG + compute_weights. Resolves the
    name via the filesystem index, then imports EXACTLY ONE module."""
    dotted = _index().get(name)
    if dotted is None:
        raise StrategyNotFound(name)
    module = importlib.import_module(dotted)
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
    """All discoverable strategy names (`_`-prefixed modules/dirs already excluded by `_index`)."""
    return sorted(_index())
