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
