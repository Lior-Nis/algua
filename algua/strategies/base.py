from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel

from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import ConstructFn

# The AUTHORED signal: a pure module-level `signal(view, params) -> pd.Series` of cross-sectional
# scores (NOT weights). The protocol-level `Strategy.target_weights(features)` is exposed only by
# the LoadedStrategy adapter, which composes signal -> construction.
SignalFn = Callable[[pd.DataFrame, dict[str, Any]], pd.Series]

# OPTIONAL vectorized acceleration: a pure `signal_panel(bars, params)` returning the FULL decision-
# time SCORES matrix (index=timestamp, columns=symbol; PRE-lag) in one shot. NOT a second signal
# definition: the engine uses it only behind a fail-closed WEIGHT-level parity guard.
SignalPanelFn = Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]

# OPT-IN fundamentals signal (issue #132): `signal(view, params, fundamentals)`. Distinct type so
# the 2-arg and 3-arg forms never silently overload.
FundamentalsSignalFn = Callable[[pd.DataFrame, dict[str, Any], pd.DataFrame], pd.Series]


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
    """Binds a StrategyConfig + the authored signal fn(s) + the RESOLVED construction policy into an
    object satisfying the Strategy protocol. Exactly one of (`signal_fn`, `fundamentals_signal_fn`)
    is active, selected by `config.needs_fundamentals`. The adapter is the ONLY place the
    protocol-level `target_weights` exists; it composes construct(signal(view), view).

    `construct_fn` is the RAW policy callable (never a params-bound partial): `construct` reads
    `config.construction_params` at call time, so a sweep that rebuilds the config takes effect and
    `inspect.getmodule(construct_fn)` resolves to the policy module for the identity hash.
    """

    config: StrategyConfig
    construct_fn: ConstructFn
    signal_fn: SignalFn | None = None
    signal_panel_fn: SignalPanelFn | None = None
    fundamentals_signal_fn: FundamentalsSignalFn | None = None

    def __post_init__(self) -> None:
        if self.config.needs_fundamentals:
            if self.fundamentals_signal_fn is None:
                raise ValueError(
                    "needs_fundamentals=True requires a 3-arg signal (fundamentals_signal_fn)"
                )
        elif self.signal_fn is None:
            raise ValueError("needs_fundamentals=False requires a 2-arg signal (signal_fn)")

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
    def authored_signal(self) -> SignalFn | FundamentalsSignalFn:
        """The active authored signal fn — used wherever code needs the strategy's source module
        (e.g. code_hash), since `signal_fn` is None for a needs_fundamentals strategy."""
        fn = self.fundamentals_signal_fn if self.config.needs_fundamentals else self.signal_fn
        assert fn is not None  # __post_init__ guarantees the active fn is set
        return fn

    def signal(self, view: pd.DataFrame, fundamentals: pd.DataFrame | None = None) -> pd.Series:
        if self.config.needs_fundamentals:
            if fundamentals is None:
                raise ValueError(
                    f"strategy {self.name!r} needs fundamentals but signal was called without a "
                    f"fundamentals frame (fail closed)"
                )
            assert self.fundamentals_signal_fn is not None
            return self.fundamentals_signal_fn(view, self.config.params, fundamentals)
        assert self.signal_fn is not None
        return self.signal_fn(view, self.config.params)

    def signal_panel(self, bars: pd.DataFrame) -> pd.DataFrame | None:
        if self.signal_panel_fn is None:
            return None
        return self.signal_panel_fn(bars, self.config.params)

    def construct(self, scores: pd.Series, view: pd.DataFrame) -> pd.Series:
        return self.construct_fn(scores, view, self.config.construction_params)

    def target_weights(
        self, features: pd.DataFrame, fundamentals: pd.DataFrame | None = None
    ) -> pd.Series:
        return self.construct(self.signal(features, fundamentals), features)


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
