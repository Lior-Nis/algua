from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from algua.contracts.types import DataProvider

# config_hash is canonical in strategies.base (shared by the backtest engine and the registry's
# live-approval gate). Re-exported here because it is also part of backtest provenance (#38).
from algua.strategies.base import config_hash as config_hash


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
    # Point-in-time universe provenance — a SEPARATE dimension from the bars `snapshot_id`
    # (which still names only the bars provider snapshot). `None` in static-universe runs.
    universe_name: str | None = None
    universe_snapshots: list[dict[str, str]] | None = None
    # Fundamentals snapshot used by a needs_fundamentals strategy (issue #132); None otherwise.
    fundamentals_snapshot: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
