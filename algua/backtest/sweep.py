from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from algua.backtest.engine import BacktestError
from algua.backtest.walkforward import walk_forward
from algua.contracts.types import DataProvider
from algua.strategies.base import LoadedStrategy

_MAX_COMBOS = 200


def _override(strategy: LoadedStrategy, combo: dict[str, Any]) -> LoadedStrategy:
    """Return a LoadedStrategy whose params are the base params with `combo` merged over them.
    Does not mutate the base strategy/config."""
    new_params = {**strategy.config.params, **combo}
    new_config = strategy.config.model_copy(update={"params": new_params})
    return LoadedStrategy(config=new_config, fn=strategy.fn)


def _combos(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of the grid into a list of param dicts. Guarded by _MAX_COMBOS."""
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    total = 1
    for values in value_lists:
        total *= len(values)
    if total > _MAX_COMBOS:
        raise BacktestError(
            f"grid too large: {total} combos > {_MAX_COMBOS}; narrow the grid"
        )
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*value_lists)]


def _coerce(value: str) -> Any:
    """Coerce a grid value string to int, then float, else leave as str."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_grid(params: list[str]) -> dict[str, list[Any]]:
    """Parse repeatable `KEY=v1,v2,...` flags into a grid dict. Values coerced int->float->str."""
    if not params:
        raise ValueError("provide at least one --param KEY=v1,v2,...")
    grid: dict[str, list[Any]] = {}
    for item in params:
        if "=" not in item:
            raise ValueError(f"malformed --param {item!r}: expected KEY=v1,v2,...")
        key, _, raw = item.partition("=")
        key = key.strip()
        values = [v.strip() for v in raw.split(",") if v.strip() != ""]
        if not key or not values:
            raise ValueError(f"malformed --param {item!r}: empty key or value list")
        if key in grid:
            raise ValueError(f"duplicate --param key {key!r}: specify each key only once")
        grid[key] = [_coerce(v) for v in values]
    return grid


_RANK_KEYS = {"mean_sharpe", "min_sharpe"}


@dataclass
class SweepResult:
    strategy: str
    data_source: str
    snapshot_id: str | None
    timeframe: str
    seed: int | None
    period: dict[str, str]
    windows: int
    holdout_frac: float
    grid: dict[str, list[Any]]
    n_combos: int
    rank_by: str
    ranked: list[dict[str, Any]]
    best: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "data_source": self.data_source,
            "snapshot_id": self.snapshot_id,
            "timeframe": self.timeframe,
            "seed": self.seed,
            "period": self.period,
            "windows": self.windows,
            "holdout_frac": self.holdout_frac,
            "grid": self.grid,
            "n_combos": self.n_combos,
            "rank_by": self.rank_by,
            "ranked": self.ranked,
            "best": self.best,
        }


def sweep(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    grid: dict[str, list[Any]],
    windows: int = 4,
    holdout_frac: float = 0.2,
    rank_by: str = "mean_sharpe",
) -> SweepResult:
    """Evaluate every grid combo with walk_forward and rank by an out-of-sample window metric.

    The holdout is carried per combo but never used for ranking (reserved for promotion gates).
    """
    if rank_by not in _RANK_KEYS:
        raise ValueError(f"rank_by must be one of {sorted(_RANK_KEYS)}, got {rank_by!r}")
    combos = _combos(grid)

    records: list[dict[str, Any]] = []
    meta = None
    for combo in combos:
        wf = walk_forward(
            _override(strategy, combo), provider, start, end,
            windows=windows, holdout_frac=holdout_frac,
        )
        if meta is None:
            meta = wf
        h = wf.holdout_metrics
        records.append({
            "params": combo,
            "config_hash": wf.config_hash,
            "n_windows": wf.windows,
            "stability": wf.stability,
            "holdout": {
                "n_bars": h["n_bars"], "sharpe": h["sharpe"],
                "total_return": h["total_return"], "max_drawdown": h["max_drawdown"],
            },
            "score": wf.stability[rank_by],
        })

    records.sort(key=lambda r: r["score"], reverse=True)
    best = {"params": records[0]["params"], "score": records[0]["score"]}
    assert meta is not None  # combos is always non-empty (grid has >=1 key with >=1 value)
    return SweepResult(
        strategy=strategy.name,
        data_source=meta.data_source,
        snapshot_id=meta.snapshot_id,
        timeframe=meta.timeframe,
        seed=meta.seed,
        period=meta.period,
        windows=windows,
        holdout_frac=holdout_frac,
        grid=grid,
        n_combos=len(combos),
        rank_by=rank_by,
        ranked=records,
        best=best,
    )
