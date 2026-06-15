"""Standalone factor evaluation (issue #140 slice B): wrap a single catalogued, signal-shaped
factor as an ephemeral synthetic strategy, run it through the existing backtest engine, and
compute construction-free rank IC/IR. Factors are NEVER registered, gate-tokened, or live-pathed:
the synthetic name uses the reserved `__factor__:` prefix and nothing here touches the registry."""
from __future__ import annotations

from typing import Any

from algua.contracts.types import ExecutionContract
from algua.features.catalogue import FactorSpec, load_factor_callable
from algua.portfolio.construction import get_construction_policy, validate_construction_params
from algua.strategies.base import LoadedStrategy, StrategyConfig

SYNTHETIC_PREFIX = "__factor__:"


def build_factor_strategy(
    spec: FactorSpec,
    *,
    symbols: list[str],
    params: dict[str, Any],
    construction: str,
    construction_params: dict[str, Any],
    execution: ExecutionContract | None = None,
) -> LoadedStrategy:
    """Wrap a standalone-evaluable factor as a synthetic LoadedStrategy.

    Construction is required (no default) so factor eval imposes no hidden weighting bias.
    Rejects a non-standalone factor.
    """
    if not spec.standalone:
        raise ValueError(
            f"factor {spec.name!r} is not standalone-evaluable (not signal-shaped); "
            f"only standalone factors can be evaluated on their own"
        )
    validate_construction_params(construction, construction_params)
    fn = load_factor_callable(spec)
    config = StrategyConfig(
        name=f"{SYNTHETIC_PREFIX}{spec.name}",
        universe=symbols,
        execution=execution or ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params=params,
        construction=construction,
        construction_params=construction_params,
    )
    return LoadedStrategy(
        config=config,
        construct_fn=get_construction_policy(construction),
        signal_fn=fn,
    )
