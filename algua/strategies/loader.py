from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from types import ModuleType

import algua.strategies as _strategies_pkg
from algua.contracts.model_types import ModelHandle
from algua.portfolio.construction import (
    ConstructionError,
    get_construction_policy,
    validate_construction_params,
)
from algua.portfolio.overlays import OverlayError, resolve_overlays
from algua.primitives.module_refresh import ModuleRefreshError, refresh_package_closure
from algua.strategies.base import (
    LoadedStrategy,
    StrategyConfig,
)
from algua.strategies.tradable import (
    assert_tradable_without_fundamentals,
    assert_tradable_without_model,
    assert_tradable_without_news,
)


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


def _reload_strategy_closure(dotted: str) -> ModuleType:
    """Re-import the strategy module ``dotted`` and its author-written first-party helper modules
    so a warm batch worker (#326) does not carry their module-level state across tasks. A
    strategy's helpers live as sibling modules in its family package ``algua.strategies.<family>``;
    every loaded module under that package is replaced by FRESH module objects imported from
    CURRENT source (the family ``__init__`` included), all-or-nothing, with stale bytecode purged
    first (see ``primitives.module_refresh``). The enforced-pure shared layers outside the family
    package stay warm. A cyclic or symlinked family closure fails closed as not found. Returns the
    fresh strategy module, so a caller never re-reads ``sys.modules`` after the import lock is
    released (another thread's refresh may then be mid-transaction)."""
    try:
        family = dotted.rsplit(".", 1)[0]  # algua.strategies.<family>
        return refresh_package_closure(family, root=dotted)
    except ModuleRefreshError as exc:
        raise StrategyNotFound(f"{dotted}: {exc}") from exc


def load_strategy(name: str, *, reload: bool = False) -> LoadedStrategy:
    """Load a bundled strategy by bare name; it must expose CONFIG + signal, and CONFIG must name a
    known construction policy with valid params. Optional `signal_panel` is the vectorized twin.
    Resolves the name via the filesystem family index, then imports EXACTLY ONE module.

    ``reload=True`` force-reloads the strategy AND its author-written first-party helper modules
    before extracting CONFIG/signal. A cold ``uv run algua`` process imports each strategy module
    exactly once, so its module-level globals start pristine. A long-lived batch worker (``research
    run-all``, #326) reuses ONE process across many strategies, so ``sys.modules`` would otherwise
    carry a strategy's OWN module-level state into the next task. A strategy's first-party helper
    modules are part of its artifact identity (they are hashed into ``code_hash`` — see
    ``registry.approvals``) and live as sibling modules in its family package, so every loaded
    ``algua.strategies.<family>.*`` module is replaced by a fresh import of the strategy's current
    closure (see ``_reload_strategy_closure``). The enforced-pure shared layers
    (``algua.features`` / ``portfolio`` / ``contracts``, import-linter-guarded to hold no mutable
    globals) need no reload; the heavy vectorbt/numba stack stays warm."""
    dotted = _index().get(name)
    if dotted is None:
        raise StrategyNotFound(name)
    if reload:
        module = _reload_strategy_closure(dotted)
    else:
        module = importlib.import_module(dotted)
    if not hasattr(module, "CONFIG") or not hasattr(module, "signal"):
        raise StrategyNotFound(f"{name} is missing CONFIG or signal")

    config = module.CONFIG
    # One name per strategy, enforced at the single load chokepoint: a hand-edited
    # CONFIG.name that diverges from the filesystem/registry name would silently fragment
    # the strategy's identity across MLflow, docs, and the registry (#275).
    if config.name != name:
        raise StrategyNotFound(
            f"{name}: CONFIG.name is {config.name!r} but the module was loaded as {name!r}; "
            f"they must match"
        )
    try:
        construct_fn = get_construction_policy(config.construction)
        validate_construction_params(config.construction, config.construction_params)
        overlay_fns = resolve_overlays(config.overlays, feature_lookback=config.feature_lookback)
    except (ConstructionError, OverlayError) as exc:
        raise StrategyNotFound(f"{name}: {exc}") from exc

    panel_fn = getattr(module, "signal_panel", None)
    if panel_fn is not None and not callable(panel_fn):
        raise StrategyNotFound(
            f"{name}.signal_panel is not callable (got {type(panel_fn).__name__})"
        )

    needs_fundamentals = bool(getattr(config, "needs_fundamentals", False))
    needs_news = bool(getattr(config, "needs_news", False))
    needs_model = bool(getattr(config, "needs_model", False))
    n_params = len(inspect.signature(module.signal).parameters)

    if needs_model:
        if panel_fn is not None:
            raise StrategyNotFound(
                f"{name}: signal_panel is not supported with needs_model "
                f"(no vectorized model fast path yet)"
            )
        if n_params != 3:
            raise StrategyNotFound(
                f"{name}: needs_model=True requires signal(view, params, model); "
                f"got {n_params} params"
            )
        handle = _resolve_model_handle(name, config)
        return LoadedStrategy(
            config=config,
            model_signal_fn=module.signal,
            model_handle=handle,
            construct_fn=construct_fn,
            overlay_fns=overlay_fns,
        )
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
            config=config, fundamentals_signal_fn=module.signal, construct_fn=construct_fn,
            overlay_fns=overlay_fns,
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
            config=config, news_signal_fn=module.signal, construct_fn=construct_fn,
            overlay_fns=overlay_fns,
        )

    if n_params != 2:
        raise StrategyNotFound(f"{name}: signal must take (view, params); got {n_params} params")
    return LoadedStrategy(
        config=config, signal_fn=module.signal, signal_panel_fn=panel_fn, construct_fn=construct_fn,
        overlay_fns=overlay_fns,
    )


def load_strategy_config(name: str) -> StrategyConfig:
    """Read a strategy's declared config without resolving any referenced model artifact.

    The module closure is force-refreshed first, so a warm process that already imported it
    cannot retain (and later freeze) a CONFIG that no longer matches the current source; later
    non-reloading loads in the same process then observe the refreshed module."""
    dotted = _index().get(name)
    if dotted is None:
        raise StrategyNotFound(name)
    module = _reload_strategy_closure(dotted)
    config = getattr(module, "CONFIG", None)
    if not isinstance(config, StrategyConfig) or config.name != name:
        raise StrategyNotFound(f"{name}: missing or mismatched CONFIG")
    return config


def _resolve_model_handle(name: str, config: StrategyConfig) -> ModelHandle:
    """Resolve a needs_model strategy's PINNED model_ref against the model registry and build the
    ModelHandle injected into signal(view, params, model). Fails closed (StrategyNotFound) unless
    the resolved version's artifact digest, training_as_of, AND provenance_digest all match the
    pinned ref — so a strategy can NEVER silently bind a different model (or a model whose training
    provenance was rewritten) than the one its config was validated with (issue #376)."""
    import hashlib

    from algua.models import ModelRegistryError, get_version_with_bytes

    ref = config.model_ref
    if ref is None:  # defensive — __post_init__ already enforces this
        raise StrategyNotFound(f"{name}: needs_model=True but model_ref is missing")
    try:
        # ONE atomic read of (metadata, bytes) — no window where the returned bytes could belong to
        # a different manifest state than the validated metadata.
        version, artifact_bytes = get_version_with_bytes(ref.name, ref.version)
    except ModelRegistryError as exc:
        raise StrategyNotFound(f"{name}: model {ref.name!r} v{ref.version}: {exc}") from exc
    bytes_digest = hashlib.sha256(artifact_bytes).hexdigest()[:16]
    if (
        version.digest != ref.digest
        or bytes_digest != ref.digest  # the ACTUAL bytes must match the pin, not only the metadata
        or version.training_as_of != ref.training_as_of
        or version.provenance_digest != ref.provenance_digest
    ):
        raise StrategyNotFound(
            f"{name}: pinned model_ref does not match registry model {ref.name!r} v{ref.version} "
            f"(digest/training_as_of/provenance mismatch — the config was validated against a "
            f"different model artifact)"
        )
    return ModelHandle(version=version, artifact_bytes=artifact_bytes)


def load_tradable_strategy(name: str) -> LoadedStrategy:
    """Load a strategy AND assert it can trade off bars alone.

    The shared paper/live preamble: a ``needs_fundamentals``/``needs_news``/``needs_model``
    strategy has no paper/live data lane yet, so it must be refused before any order work. Kept
    beside ``load_strategy`` because the tradability assertions are a strategies-layer concern.
    """
    strategy = load_strategy(name)
    assert_tradable_without_fundamentals(strategy)
    assert_tradable_without_news(strategy)
    assert_tradable_without_model(strategy)
    return strategy


def list_strategies() -> list[str]:
    """All discoverable strategy names (`_`-prefixed modules/dirs already excluded by `_index`)."""
    return sorted(_index())


def _loaded_for_test(config: StrategyConfig) -> LoadedStrategy:
    """Test-only: build a LoadedStrategy from a config with a trivial signal + its resolved policy.
    Used by config_hash tests that should not depend on a real example module."""
    import pandas as pd
    fn = get_construction_policy(config.construction)
    return LoadedStrategy(
        config=config, signal_fn=lambda view, params: pd.Series(dtype="float64"), construct_fn=fn,
        overlay_fns=resolve_overlays(config.overlays, feature_lookback=config.feature_lookback),
    )
