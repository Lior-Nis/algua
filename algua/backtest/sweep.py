from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from algua.backtest.engine import BacktestError
from algua.backtest.walkforward import walk_forward
from algua.contracts.types import DataProvider
from algua.portfolio.construction import ConstructionError, validate_construction_params
from algua.strategies.base import LoadedStrategy

_MAX_COMBOS = 200

_CONSTRUCTION_PREFIX = "construction."


def _override(strategy: LoadedStrategy, combo: dict[str, Any]) -> LoadedStrategy:
    """Return a LoadedStrategy whose params/construction_params are the base merged with `combo`.

    A grid key prefixed `construction.` tunes `construction_params` (re-validated by the policy);
    any other key tunes signal `params` and MUST already exist in the base params (so a typo'd key
    is rejected, never a silent no-op). Preserves the resolved construction policy + signal_panel.
    Does not mutate the base strategy/config.
    """
    new_params = dict(strategy.config.params)
    new_cparams = dict(strategy.config.construction_params)
    for key, value in combo.items():
        if key.startswith(_CONSTRUCTION_PREFIX):
            new_cparams[key[len(_CONSTRUCTION_PREFIX):]] = value
        else:
            if key not in strategy.config.params:
                raise ValueError(
                    f"sweep key {key!r} is not a base signal param "
                    f"{sorted(strategy.config.params)}; prefix with 'construction.' to tune the "
                    f"construction policy"
                )
            new_params[key] = value
    try:
        validate_construction_params(strategy.config.construction, new_cparams)
    except ConstructionError as exc:
        raise ValueError(f"swept construction_params invalid: {exc}") from exc
    new_config = strategy.config.model_copy(
        update={"params": new_params, "construction_params": new_cparams}
    )
    return LoadedStrategy(
        config=new_config,
        construct_fn=strategy.construct_fn,
        signal_fn=strategy.signal_fn,
        signal_panel_fn=strategy.signal_panel_fn,
        fundamentals_signal_fn=strategy.fundamentals_signal_fn,
    )


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


def _coerce_values(values: list[Any]) -> list[Any]:
    """Widen a homogeneous-numeric value list to float if any element is float.

    Prevents silent type mixing when a grid mixes e.g. "10,10.5" → [int(10), float(10.5)].
    Any list that already contains a non-numeric value is returned unchanged.
    """
    has_float = any(type(v) is float for v in values)
    if has_float and all(isinstance(v, (int, float)) for v in values):
        return [float(v) for v in values]
    return list(values)


def parse_grid(params: list[str]) -> dict[str, list[Any]]:
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
        grid[key] = _coerce_values([_coerce(v) for v in values])
    return grid


def _rank_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort records descending by score; ties broken by ascending std_sharpe (more stable wins).

    Non-finite score or std_sharpe (NaN, ±inf) always sorts last so a degenerate combo
    never beats a finite one.  The sort is stable: equal score + equal std_sharpe
    preserves the original order.
    """
    def _key(r: dict[str, Any]) -> tuple[float, float]:
        score = r["score"]
        std = r["stability"]["std_sharpe"]
        # Map non-finite to sentinels that sink to the bottom under reverse=True.
        # score: -inf → always last;  std_sharpe: +inf → -(-inf) = -inf → also last.
        finite_score = score if math.isfinite(score) else -math.inf
        finite_std = std if math.isfinite(std) else math.inf
        return (finite_score, -finite_std)

    return sorted(records, key=_key, reverse=True)


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
    code_hash: str | None = None
    dependency_hash: str | None = None
    # Point-in-time universe provenance — separate from the bars `snapshot_id` (see BacktestResult).
    universe_name: str | None = None
    universe_snapshots: list[dict[str, str]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


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
    universe_by_date: Mapping[date, Collection[str]] | None = None,
    universe_name: str | None = None,
    universe_snapshots: list[dict[str, str]] | None = None,
) -> SweepResult:
    """Evaluate every grid combo with walk_forward and rank by an out-of-sample window metric.

    The holdout is COMPUTED by each combo's walk_forward but is DELIBERATELY NOT recorded here:
    it is never used for ranking (ranking is on window/stability), and exposing a per-combo holdout
    would let a caller SELECT the best combo on the untouched holdout across the whole grid — the
    exact multiple-testing leak the promotion breadth gate fights. The holdout is revealed (and
    burned) in exactly one place: `research promote`.
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
            universe_by_date=universe_by_date,
            universe_name=universe_name, universe_snapshots=universe_snapshots,
        )
        if meta is None:
            meta = wf
        # Note: wf.holdout_metrics is intentionally NOT copied into the record (see docstring).
        records.append({
            "params": combo,
            "config_hash": wf.config_hash,
            "n_windows": wf.windows,
            "stability": wf.stability,
            "score": wf.stability[rank_by],
        })

    ranked = _rank_records(records)
    best = {"params": ranked[0]["params"], "score": ranked[0]["score"]}
    assert meta is not None  # combos is always non-empty (grid has >=1 key with >=1 value)
    return SweepResult(
        strategy=strategy.name,
        data_source=meta.data_source,
        snapshot_id=meta.snapshot_id,
        timeframe=meta.timeframe,
        seed=meta.seed,
        code_hash=meta.code_hash,
        dependency_hash=meta.dependency_hash,
        period=meta.period,
        windows=windows,
        holdout_frac=holdout_frac,
        grid=grid,
        n_combos=len(combos),
        rank_by=rank_by,
        ranked=ranked,
        best=best,
        universe_name=meta.universe_name,
        universe_snapshots=meta.universe_snapshots,
    )
