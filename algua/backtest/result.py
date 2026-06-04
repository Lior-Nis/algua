from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from algua.contracts.types import DataProvider
from algua.strategies.base import LoadedStrategy


def config_hash(strategy: LoadedStrategy) -> str:
    """Stable short hash of the config that defines a backtest's identity.

    Lives beside the result/provenance code because it is provenance, not simulation.
    Used by both run() and walk_forward() (#38).
    """
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


def provenance(provider: DataProvider, seed: int | None) -> dict[str, Any]:
    """Provenance fields shared by every backtest-family result (#43).

    A provider that pins its own `seed`/`snapshot_id` wins (real snapshots and the
    synthetic provider both do this); otherwise the caller's explicit `seed` is recorded.
    Used identically by run() and walk_forward() so their seed/source/snapshot provenance
    can never drift.
    """
    return {
        "data_source": type(provider).__name__,
        "seed": getattr(provider, "seed", seed),
        "snapshot_id": getattr(provider, "snapshot_id", None),
    }


@dataclass
class BacktestResult:
    strategy: str
    metrics: dict[str, float]
    config_hash: str
    data_source: str
    timeframe: str
    period: dict[str, str]
    seed: int | None = None
    snapshot_id: str | None = None
    code_hash: str | None = None
    dependency_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
