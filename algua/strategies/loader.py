from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

import algua.strategies as _strategies_pkg
from algua.contracts.model_types import ModelHandle
from algua.portfolio.construction import (
    ConstructionError,
    get_construction_policy,
    validate_construction_params,
)
from algua.strategies.base import (
    LoadedStrategy,
    StrategyConfig,
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


def _reload_strategy_closure(dotted: str) -> None:
    """Reload the strategy module ``dotted`` and its author-written first-party helper modules so a
    warm batch worker (#326) does not carry their module-level state across tasks. A strategy's
    helpers live as sibling modules in its family package ``algua.strategies.<family>``; reload
    every already-loaded module under that package, the strategy module LAST (so it re-binds helper
    references). The family ``__init__`` is reloaded too (a family may keep shared helpers there).
    The enforced-pure shared layers outside the family package are intentionally left warm."""
    family_pkg = dotted.rsplit(".", 1)[0]  # algua.strategies.<family>
    prefix = family_pkg + "."
    # sys.modules INSERTION order is dependency-first: Python finishes importing a module's
    # imported submodules (inserting them) before the importer finishes and is itself inserted. So
    # iterating in insertion order reloads a helper's dependencies BEFORE the helper, and the helper
    # re-executes its `from ._dep import x` against the freshly-reloaded dep — no stale object
    # rebind. The strategy module is reloaded LAST (below), after every helper it depends on.
    siblings = [
        m for m in list(sys.modules)
        if (m == family_pkg or m.startswith(prefix)) and m != dotted and sys.modules[m] is not None
    ]
    for mod_name in siblings:
        # Best-effort: a stale entry whose source file was deleted (e.g. a test's temp module left
        # in sys.modules after its file was unlinked) is not a live dependency of THIS strategy and
        # cannot be reloaded — skip it rather than fail the load.
        if not _module_source_exists(sys.modules[mod_name]):
            continue
        importlib.reload(sys.modules[mod_name])
    importlib.reload(sys.modules[dotted])  # the strategy module last


def _module_source_exists(module: object) -> bool:
    """True iff the module still has an on-disk source file (a reload target). A namespace package
    or a module whose file was deleted returns False."""
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if not isinstance(origin, str) or origin in ("built-in", "frozen", "namespace"):
        return False
    return Path(origin).exists()


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
    ``registry.approvals``) and live as sibling modules in its family package, so we reload every
    already-loaded ``algua.strategies.<family>.*`` module (helpers first, the strategy module last,
    so the root re-binds fresh helper references). The enforced-pure shared layers
    (``algua.features`` / ``portfolio`` / ``contracts``, import-linter-guarded to hold no mutable
    globals) need no reload; the heavy vectorbt/numba stack stays warm."""
    dotted = _index().get(name)
    if dotted is None:
        raise StrategyNotFound(name)
    module = importlib.import_module(dotted)
    if reload:
        _reload_strategy_closure(dotted)
        module = sys.modules[dotted]
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
    except ConstructionError as exc:
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
        config=config, signal_fn=lambda view, params: pd.Series(dtype="float64"), construct_fn=fn
    )
