from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import multiprocessing
import pickle
from collections.abc import Collection, Mapping
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from algua.backtest.core_budget import admit, close_lease_fd_in_worker, cpu_budget
from algua.backtest.delisting import DelistingRecord
from algua.backtest.engine import BacktestError
from algua.backtest.walkforward import walk_forward
from algua.contracts.types import DataProvider, FundamentalsProvider, NewsProvider
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
        news_signal_fn=strategy.news_signal_fn,
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
    """Coerce a grid value string to int, then float, else leave as str.

    Non-finite floats ('inf'/'nan'/'-inf'/'1e400') are rejected here with a clear message
    rather than coerced and carried downstream, where they only surface as an opaque
    JSON-serialization failure in config_hash (json.dumps(allow_nan=False)) (#258).
    """
    try:
        return int(value)
    except ValueError:
        pass
    try:
        f = float(value)
    except ValueError:
        return value
    if not math.isfinite(f):
        raise ValueError(f"non-finite grid value: {value!r}")
    return f


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


def _trial_sharpe_stats(
    records: list[dict[str, Any]],
) -> tuple[int, float | None, float | None]:
    """Per-combo trial-Sharpe ``(count, mean, sample-variance)`` for the DSR evidence layer (#211).

    Uses each combo's CANONICAL per-combo Sharpe — the mean-window Sharpe
    (``stability["mean_sharpe"]``) — NOT the ranking ``score``: ``score`` is ``rank_by``-dependent
    (it is ``min_sharpe`` when ranking by worst window), a different statistic from the holdout
    Sharpe the DSR compares against, which would mis-calibrate the benchmark.

    FAILS CLOSED — returns ``(0, None, None)`` — when the sweep is degenerate (no combos, or ANY
    non-finite per-combo Sharpe). Silently dropping non-finite combos would make the recorded count
    undercount the trials the DSR's N still counts, shrinking the dispersion and WEAKENING the gate;
    None stats instead make the pooled accessor return None so the binding DSR check fails closed.
    Variance is sample variance (``ddof=1``) for count ≥ 2, and ``0.0`` for a single combo.
    """
    combo_sharpes = [r["stability"]["mean_sharpe"] for r in records]
    if not combo_sharpes or not all(math.isfinite(s) for s in combo_sharpes):
        return 0, None, None
    count = len(combo_sharpes)
    mean = float(np.mean(combo_sharpes))
    var = float(np.var(combo_sharpes, ddof=1)) if count >= 2 else 0.0
    return count, mean, var


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
    trial_sharpe_count: int = 0
    trial_sharpe_mean: float | None = None
    trial_sharpe_var_ann: float | None = None
    code_hash: str | None = None
    dependency_hash: str | None = None
    # Point-in-time universe provenance — separate from the bars `snapshot_id` (see BacktestResult).
    universe_name: str | None = None
    universe_snapshots: list[dict[str, str]] | None = None
    fundamentals_snapshot: str | None = None
    news_snapshot: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _evaluate_combo(
    overridden: LoadedStrategy,
    *,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    windows: int,
    holdout_frac: float,
    universe_by_date: Mapping[date, Collection[str]] | None,
    universe_name: str | None,
    universe_snapshots: list[dict[str, str]] | None,
    rank_by: str,
    delisting_records: Mapping[str, list[DelistingRecord]] | None,
    assume_terminal_last_close: bool,
    fundamentals_provider: FundamentalsProvider | None = None,
    news_provider: NewsProvider | None = None,
) -> dict[str, Any]:
    """Evaluate one already-overridden combo via walk_forward; return its rankable record + the
    combo-independent meta. Module-level so it is picklable into a ProcessPoolExecutor worker.

    Deliberately does NOT return wf.holdout_metrics: the holdout never leaves the worker process,
    preserving sweep's single-use-holdout discipline (the holdout is revealed only in
    `research promote`).
    """
    wf = walk_forward(
        overridden, provider, start, end,
        windows=windows, holdout_frac=holdout_frac,
        universe_by_date=universe_by_date,
        universe_name=universe_name, universe_snapshots=universe_snapshots,
        fundamentals_provider=fundamentals_provider,
        news_provider=news_provider,
        delisting_records=delisting_records,
        assume_terminal_last_close=assume_terminal_last_close,
    )
    return {
        "config_hash": wf.config_hash,
        "n_windows": wf.windows,
        "stability": wf.stability,
        "score": wf.stability[rank_by],
        "meta": {
            "data_source": wf.data_source,
            "snapshot_id": wf.snapshot_id,
            "timeframe": wf.timeframe,
            "seed": wf.seed,
            "code_hash": wf.code_hash,
            "dependency_hash": wf.dependency_hash,
            "period": wf.period,
            "universe_name": wf.universe_name,
            "universe_snapshots": wf.universe_snapshots,
            "fundamentals_snapshot": wf.fundamentals_snapshot,
            "news_snapshot": wf.news_snapshot,
        },
    }


def _evaluate_combo_pooled(overridden: LoadedStrategy, **kwargs: Any) -> dict[str, Any]:
    """Pool-worker wrapper: pin BLAS/OpenMP to ONE thread for this combo. numpy here is OpenBLAS
    (many threads by default), so N worker processes each spawning a full BLAS pool would
    oversubscribe the cores. The runtime `threadpool_limits` works under the default `fork` start
    method (no env-before-import needed). The inline path deliberately does NOT call this — a lone
    combo should use every core.
    """
    with threadpool_limits(limits=1):
        return _evaluate_combo(overridden, **kwargs)


def _run_combos(
    overridden: list[LoadedStrategy], eval_kwargs: dict[str, Any]
) -> list[dict[str, Any]]:
    """Evaluate every pre-built combo strategy via walk_forward, returning records in COMBO ORDER.

    A single combo (or a single-core host) runs inline — no pool overhead, full BLAS threads.
    Otherwise a ProcessPoolExecutor fans the combos out; `executor.map` preserves input order so the
    stable rank downstream stays reproducible.

    Errors:
      * A non-picklable strategy/provider is caught by a PARENT-side pickle preflight and raised as
        BacktestError — a non-picklable nested fn raises AttributeError, a lambda PicklingError,
        other cases TypeError, none of which @json_errors wraps, so the preflight keeps the JSON
        contract intact and the message actionable.
      * A combo's own failure (e.g. walk_forward raising BacktestError) is delivered back through
        `map` and propagates with its OWN type — `except BacktestError: raise` keeps it unwrapped.
      * A worker crash/OOM/kill (BrokenExecutor) is re-raised as BacktestError so the CLI's
        @json_errors(ValueError, LookupError, BacktestError) still emits a JSON envelope. The catch
        stays narrow (BrokenExecutor only) so a real worker bug surfacing as its own type is not
        masked.
      * Ordering matters: the domain re-raise MUST precede the infrastructure catch, or a worker
        BacktestError would be double-wrapped into "parallel sweep failed".
    """
    # Inline (no pool, full BLAS threads) when there is at most one combo or the core budget is a
    # single core. The budget is the GLOBAL injected core allowance (#327), not os.cpu_count():
    # `ALGUA_SWEEP_CPU_BUDGET=1` collapses every sweep to the inline path.
    if len(overridden) <= 1 or cpu_budget() <= 1:
        return [_evaluate_combo(ov, **eval_kwargs) for ov in overridden]

    worker = functools.partial(_evaluate_combo_pooled, **eval_kwargs)
    # Preflight the pickling in the PARENT so a non-picklable input becomes a JSON-safe
    # BacktestError rather than a raw AttributeError/PicklingError/TypeError escaping mid-dispatch.
    # Picklability is identical across combos (they share the strategy's fn refs + the provider
    # bound in `worker`), so the first override is representative.
    try:
        pickle.dumps(worker)
        pickle.dumps(overridden[0])
    except (pickle.PicklingError, AttributeError, TypeError) as exc:
        raise BacktestError(
            "parallel sweep requires picklable strategy/provider inputs (a strategy signal must be "
            f"a module-level function, not a closure/lambda): {exc}"
        ) from exc
    # Admit this sweep into the GLOBAL core budget shared across ALL concurrent sweeps (#327):
    # `admit` grants a worker count carved from the budget minus what live sibling sweeps already
    # hold, so K concurrent sweeps total <= budget + (K-1) workers instead of K x cpu_count. The
    # lease flock is held for the pool's lifetime; a fork context + initializer close the inherited
    # lease fd in every worker so a parent crash still frees the lease.
    fork_ctx = multiprocessing.get_context("fork")
    try:
        with admit(len(overridden)) as lease:
            with ProcessPoolExecutor(
                max_workers=lease.grant,
                mp_context=fork_ctx,
                initializer=close_lease_fd_in_worker,
                initargs=(lease.lease_fd,),
            ) as executor:
                return list(executor.map(worker, overridden))
    except BacktestError:
        raise
    except BrokenExecutor as exc:
        raise BacktestError(f"parallel sweep failed: {exc}") from exc


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
    fundamentals_provider: FundamentalsProvider | None = None,
    news_provider: NewsProvider | None = None,
    delisting_records: Mapping[str, list[DelistingRecord]] | None = None,
    assume_terminal_last_close: bool = False,
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
    # Model lane (#376) is not supported in sweep (it runs walk_forward per combo, which is not
    # PIT-safe for a fixed-per-run model across OOS windows). Fail fast at the top with a clear
    # message rather than surfacing the per-combo walk_forward failure. See walk_forward.
    if strategy.config.needs_model:
        raise BacktestError(
            f"strategy {strategy.name!r} declares needs_model; sweep/walk-forward do not support "
            f"the model lane yet (#376 follow-up) — refusing to run (fail closed)"
        )
    combos = _combos(grid)
    # Parent pre-pass: build + validate EVERY override here so a bad signal key or invalid
    # construction param fails fast (ValueError) BEFORE any worker process spawns — exactly the
    # parent-side behavior the sequential loop had.
    overridden = [_override(strategy, combo) for combo in combos]

    eval_kwargs: dict[str, Any] = dict(
        provider=provider, start=start, end=end,
        windows=windows, holdout_frac=holdout_frac,
        universe_by_date=universe_by_date,
        universe_name=universe_name, universe_snapshots=universe_snapshots,
        rank_by=rank_by,
        fundamentals_provider=fundamentals_provider, news_provider=news_provider,
        delisting_records=delisting_records,
        assume_terminal_last_close=assume_terminal_last_close,
    )
    results = _run_combos(overridden, eval_kwargs)

    # Build records in COMBO ORDER (zip with the original combos) so _rank_records' stable
    # tie-break on equal score+std_sharpe stays reproducible regardless of worker completion order.
    records = [
        {
            "params": combo,
            "config_hash": res["config_hash"],
            "n_windows": res["n_windows"],
            "stability": res["stability"],
            "score": res["score"],
        }
        for combo, res in zip(combos, results, strict=True)
    ]
    # meta fields are combo-independent (same data + code identity for every combo); take the
    # first for parity with the prior `meta = first wf` behavior.
    meta = results[0]["meta"]

    ranked = _rank_records(records)
    best = {"params": ranked[0]["params"], "score": ranked[0]["score"]}

    t_count, t_mean, t_var = _trial_sharpe_stats(records)

    return SweepResult(
        strategy=strategy.name,
        data_source=meta["data_source"],
        snapshot_id=meta["snapshot_id"],
        timeframe=meta["timeframe"],
        seed=meta["seed"],
        code_hash=meta["code_hash"],
        dependency_hash=meta["dependency_hash"],
        period=meta["period"],
        windows=windows,
        holdout_frac=holdout_frac,
        grid=grid,
        n_combos=len(combos),
        rank_by=rank_by,
        ranked=ranked,
        best=best,
        trial_sharpe_count=t_count,
        trial_sharpe_mean=t_mean,
        trial_sharpe_var_ann=t_var,
        universe_name=meta["universe_name"],
        universe_snapshots=meta["universe_snapshots"],
        fundamentals_snapshot=meta["fundamentals_snapshot"],
        news_snapshot=meta["news_snapshot"],
    )
