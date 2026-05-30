from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "metrics": self.metrics,
            "config_hash": self.config_hash,
            "data_source": self.data_source,
            "timeframe": self.timeframe,
            "period": self.period,
            "seed": self.seed,
            "snapshot_id": self.snapshot_id,
        }
