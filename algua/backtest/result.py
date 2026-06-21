from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

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
    # News snapshot used by a needs_news strategy (issue #132); None otherwise.
    news_snapshot: str | None = None
    # Delisting provenance (issue #212): snapshot name + forced exits applied during simulate().
    delisting_snapshot: str | None = None
    forced_exits: list[dict] = field(default_factory=list)
    # Daily return series from the backtest portfolio; None if non-finite returns detected.
    returns: pd.Series | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name != "returns"
        }


def series_frame(result: BacktestResult) -> tuple[pd.DataFrame, dict[str, str]]:
    """Pure projection of a backtest's daily return series to a `[date, ret]` frame + metadata
    (#181). NO serialization, NO I/O — keeps `algua.backtest` off the data lane.
    Caller guards `returns is not None and len(returns) > 0`.

    `ret` is canonicalized `-0.0 -> +0.0` (a flat day must not perturb the parquet bytes — mirrors
    `logical_bars_hash`). Metadata embeds the WHOLE run identity as one sorted-key JSON blob so the
    standalone file is self-describing (carries config/code/dependency hashes, snapshot, seed,
    timeframe, period, universe/fundamentals/news/delisting provenance, metrics — everything in
    `to_dict()` except the series itself)."""
    if result.returns is None:
        raise ValueError("series_frame requires non-None returns; caller must guard")
    r = result.returns
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp(ts).isoformat() for ts in r.index],
            "ret": r.to_numpy(dtype=float) + 0.0,  # -0.0 -> +0.0
        }
    )
    metadata = {"algua.result_json": json.dumps(result.to_dict(), sort_keys=True, default=str)}
    return frame, metadata
