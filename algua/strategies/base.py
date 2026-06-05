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


class StrategyConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    name: str
    universe: list[str]
    execution: ExecutionContract
    params: dict[str, Any] = {}


@dataclass
class LoadedStrategy:
    """Binds a StrategyConfig + a pure authored compute_weights(view, params) function into an
    object that satisfies the Strategy protocol (.name, .execution, .target_weights). The adapter
    is the ONLY place the protocol-level 1-arg `target_weights` exists — it injects params."""

    config: StrategyConfig
    fn: ComputeWeightsFn
    # OPTIONAL vectorized fast-path hook (loader-detected, see ComputeWeightsPanelFn). `None` when
    # the strategy module does not define `compute_weights_panel`. `fn` stays canonical regardless.
    panel_fn: ComputeWeightsPanelFn | None = None

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

    def target_weights(self, features: pd.DataFrame) -> pd.Series:
        return self.fn(features, self.config.params)


def config_hash(strategy: LoadedStrategy) -> str:
    """Stable digest of a strategy's resolved configuration (name + universe + params +
    execution contract). The single source of truth for the config side of the artifact identity,
    shared by the backtest engine and the registry's live-approval gate.

    Serializes the *full* ExecutionContract via asdict, so every behavior-affecting field
    (warmup_bars, allow_fractional, max_gross_exposure, decision_lag_bars, rebalance_frequency)
    is part of the identity — and any field added later is included automatically. Two configs
    that produce different trades can therefore never collide on config_hash."""
    payload = json.dumps(
        {
            "name": strategy.name,
            "universe": strategy.universe,
            "params": strategy.params,
            "execution": asdict(strategy.execution),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
