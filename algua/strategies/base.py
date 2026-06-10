from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel

from algua.contracts.types import ExecutionContract

# The AUTHORED signal: a pure module-level `compute_weights(view, params)`. The protocol-level
# `Strategy.target_weights(features)` (1-arg) is exposed only by the LoadedStrategy adapter below,
# which closes over `params`. Two layers, two names — no silent signature drift.
ComputeWeightsFn = Callable[[pd.DataFrame, dict[str, Any]], pd.Series]

# OPTIONAL vectorized acceleration hook: a pure module-level `compute_weights_panel(bars, params)`
# that returns the FULL decision-time weights matrix (index=timestamp, columns=symbol; PRE-lag) in
# one shot, instead of being called once per bar. It is NOT a second signal definition: the engine
# uses it only behind a fail-closed parity guard against the canonical per-bar `compute_weights`,
# and raises on any disagreement. `bars` is the long-format bar-schema frame (same as the per-bar
# view, but spanning the whole period). `None` for strategies that don't define it.
ComputeWeightsPanelFn = Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]

# OPT-IN fundamentals signal (issue #132): a strategy that declares `needs_fundamentals=True` in
# CONFIG authors `compute_weights(view, params, fundamentals)` — the 3rd arg is the PIT-correct tidy
# fundamentals frame the engine materialized for decision bar t (knowable_at <= t). Distinct type so
# the 2-arg and 3-arg forms never silently overload.
ComputeFundamentalsWeightsFn = Callable[[pd.DataFrame, dict[str, Any], pd.DataFrame], pd.Series]


class StrategyConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    name: str
    universe: list[str]
    execution: ExecutionContract
    params: dict[str, Any] = {}
    # Portfolio-construction policy (issue #141): the id is resolved by the loader against
    # algua.portfolio.construction; construction_params are validated per-policy at load.
    construction: str
    construction_params: dict[str, Any] = {}
    # Opt into the as-of fundamentals lane (issue #132). When True the loader binds the 3-arg
    # signal as `fundamentals_signal_fn` and the engine injects the PIT-correct frame per bar.
    needs_fundamentals: bool = False


@dataclass
class LoadedStrategy:
    """Binds a StrategyConfig + the authored signal function(s) into an object satisfying the
    Strategy protocol. Exactly one of (`fn`, `fundamentals_fn`) is active, selected by
    `config.needs_fundamentals`. The adapter is the ONLY place the protocol-level `target_weights`
    exists — it injects params (and, for the fundamentals lane, the masked frame)."""

    config: StrategyConfig
    fn: ComputeWeightsFn | None = None
    # OPTIONAL vectorized fast-path hook (loader-detected, see ComputeWeightsPanelFn). `None` when
    # the strategy module does not define `compute_weights_panel`. `fn` stays canonical regardless.
    panel_fn: ComputeWeightsPanelFn | None = None
    fundamentals_fn: ComputeFundamentalsWeightsFn | None = None

    def __post_init__(self) -> None:
        if self.config.needs_fundamentals:
            if self.fundamentals_fn is None:
                raise ValueError(
                    "needs_fundamentals=True requires a 3-arg compute_weights (fundamentals_fn)"
                )
        elif self.fn is None:
            raise ValueError("needs_fundamentals=False requires a 2-arg compute_weights (fn)")

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def universe(self) -> list[str]:
        return self.config.universe

    @property
    def execution(self) -> ExecutionContract:
        return self.config.execution

    @property
    def params(self) -> dict[str, Any]:
        return self.config.params

    @property
    def signal_fn(self) -> ComputeWeightsFn | ComputeFundamentalsWeightsFn:
        """The active authored function — `fundamentals_fn` for the fundamentals lane, else `fn`.
        Used wherever code needs the strategy's source module (e.g. code_hash), since `fn` is None
        for a needs_fundamentals strategy."""
        fn = self.fundamentals_fn if self.config.needs_fundamentals else self.fn
        assert fn is not None  # __post_init__ guarantees the active fn is set
        return fn

    def target_weights(
        self, features: pd.DataFrame, fundamentals: pd.DataFrame | None = None
    ) -> pd.Series:
        if self.config.needs_fundamentals:
            if fundamentals is None:
                raise ValueError(
                    f"strategy {self.name!r} needs fundamentals but target_weights was called "
                    f"without a fundamentals frame (fail closed)"
                )
            assert self.fundamentals_fn is not None
            return self.fundamentals_fn(features, self.config.params, fundamentals)
        assert self.fn is not None
        return self.fn(features, self.config.params)


def assert_tradable_without_fundamentals(strategy: LoadedStrategy) -> None:
    """Fail closed: a needs_fundamentals strategy must NOT run paper/live yet — the as-of
    fundamentals lane is wired only into the backtest engine (issue #132). Called at every trading
    load point so no actor (agent promote OR human raw transition) can run it blind."""
    if strategy.config.needs_fundamentals:
        raise ValueError(
            f"strategy {strategy.name!r} declares needs_fundamentals; paper/live fundamentals "
            f"wiring is not built yet (#132 follow-up) — refusing to trade it blind"
        )


def config_hash(strategy: LoadedStrategy) -> str:
    """Stable digest of a strategy's resolved configuration (name + universe + params +
    execution contract + construction policy id and params). The single source of truth for the
    config side of the artifact identity, shared by the backtest engine and the registry's
    live-approval gate.

    Serializes the *full* ExecutionContract via asdict, so every behavior-affecting field
    (warmup_bars, allow_fractional, max_gross_exposure, decision_lag_bars, rebalance_frequency)
    is part of the identity — and any field added later is included automatically. The construction
    policy id + its params (issue #141) are folded in too, so retuning the construction policy
    (e.g. top_k) or swapping the policy invalidates a prior live approval. Serialized with
    allow_nan=False so a non-finite construction param cannot produce a non-canonical hash. Two
    configs that produce different trades can therefore never collide on config_hash."""
    payload = json.dumps(
        {
            "name": strategy.name,
            "universe": strategy.universe,
            "params": strategy.params,
            "execution": asdict(strategy.execution),
            "construction": strategy.config.construction,
            "construction_params": strategy.config.construction_params,
            "needs_fundamentals": strategy.config.needs_fundamentals,
        },
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
