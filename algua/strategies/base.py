from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel

from algua.contracts.types import ExecutionContract

TargetWeightsFn = Callable[[pd.DataFrame, dict[str, Any]], pd.Series]


class StrategyConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    name: str
    universe: list[str]
    execution: ExecutionContract
    params: dict[str, Any] = {}


@dataclass
class LoadedStrategy:
    """Binds a StrategyConfig + a pure target_weights function into an object that
    satisfies the Strategy protocol (.name, .execution, .target_weights)."""

    config: StrategyConfig
    fn: TargetWeightsFn

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
    shared by the backtest engine and the registry's live-approval gate."""
    payload = json.dumps(
        {
            "name": strategy.name,
            "universe": strategy.universe,
            "params": strategy.params,
            "execution": {
                "rebalance_frequency": strategy.execution.rebalance_frequency,
                "decision_lag_bars": strategy.execution.decision_lag_bars,
                "max_gross_exposure": strategy.execution.max_gross_exposure,
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
