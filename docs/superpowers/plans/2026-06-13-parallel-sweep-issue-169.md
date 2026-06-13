# Parallelize backtest sweep (#169) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize `sweep()`'s per-combo `walk_forward` evaluation across CPU cores with a `ProcessPoolExecutor`, preserving bit-for-bit ranking and the `SweepResult` contract.

**Architecture:** Each grid combo is a pure, independent `walk_forward` (no holdout burn, no DB writes). A parent pre-pass builds + validates every override (fail-fast), then a `ProcessPoolExecutor` fans the combos out — each worker pinned to 1 BLAS thread via `threadpoolctl` to avoid oversubscription. `executor.map` preserves combo order so the stable rank stays reproducible. Pool/pickle failures are converted to `BacktestError` so the CLI JSON envelope holds. A single combo (or single core) runs inline.

**Tech Stack:** Python `concurrent.futures.ProcessPoolExecutor`, `functools.partial`, `threadpoolctl` (new direct dep), pytest. Spec: `docs/superpowers/specs/2026-06-13-parallel-sweep-issue-169-design.md`.

---

## File Structure

- `algua/backtest/sweep.py` — MODIFY. Add module-level worker fns (`_evaluate_combo`, `_evaluate_combo_pooled`) and a dispatcher (`_run_combos`); replace the sequential loop + assembly in `sweep()` with a combo-order parent pre-pass → dispatch → assemble. New stdlib imports + `threadpoolctl`.
- `pyproject.toml` — MODIFY. Add `threadpoolctl` to `dependencies` (now imported directly).
- `tests/test_sweep.py` — MODIFY. Hoist `_momentum`'s nested `signal` to a module-level `_momentum_signal` (so the fixture pickles across the process boundary); add inline-path + combo-error-type tests.

**Quality gate (run after every commit):**
`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

### Task 1: Add `threadpoolctl` as a direct dependency

**Files:**
- Modify: `pyproject.toml:6-16`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add `threadpoolctl` to the `dependencies` list (it is already installed transitively via numpy/scipy; `sweep.py` will import it directly, so it must be declared):

```toml
dependencies = [
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "typer>=0.12,<0.28",  # _click private fork; bump only after verifying test_cli_main smoke tests pass
    "exchange-calendars>=4.5",
    "pandas>=2.2,<3",
    "yfinance>=1.4.1",
    "pyarrow>=15,<24",
    "vectorbt>=0.26",
    "mlflow>=3.1",
    "threadpoolctl>=3,<4",
]
```

- [ ] **Step 2: Sync + verify it imports**

Run: `uv sync --quiet && uv run python -c "import threadpoolctl; print(threadpoolctl.__version__)"`
Expected: prints a 3.x version (e.g. `3.6.0`), no error.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(169): declare threadpoolctl as a direct dependency"
```

---

### Task 2: Refactor the test fixture so its strategy pickles

The `_momentum()` fixture currently defines `signal` as a nested closure. A closure cannot be pickled to a worker process, so it must be hoisted to module level (matching production, where the loader only ever binds a module-level `signal`). This is a pure refactor — no behavior change — and Task 4's pool path depends on it.

**Files:**
- Modify: `tests/test_sweep.py:1-34`

- [ ] **Step 1: Hoist the signal to module level**

Replace lines 1-34 of `tests/test_sweep.py` with:

```python
from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.backtest._sample import SyntheticProvider
from algua.backtest.sweep import SweepResult, sweep
from algua.contracts.types import ExecutionContract
from algua.features.indicators import momentum
from algua.portfolio.construction import get_construction_policy
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2023, 12, 31, tzinfo=UTC)


def _momentum_signal(view, params):
    # Module-level (not a closure) so the strategy pickles to a ProcessPoolExecutor worker,
    # exactly like a real loaded strategy (the loader binds a module-level `signal`).
    wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    if len(wide) <= int(params["lookback"]):
        return pd.Series(dtype="float64")
    return momentum(wide, lookback=int(params["lookback"])).iloc[-1].dropna()


def _momentum():
    cfg = StrategyConfig(
        name="m", universe=["AAA", "BBB", "CCC"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={"lookback": 40},
        construction="top_k_equal_weight", construction_params={"top_k": 1},
    )
    return LoadedStrategy(
        config=cfg, signal_fn=_momentum_signal,
        construct_fn=get_construction_policy(cfg.construction),
    )
```

- [ ] **Step 2: Run the existing sweep tests to confirm the refactor is behavior-preserving**

Run: `uv run pytest tests/test_sweep.py -q`
Expected: PASS — all 4 existing tests (`test_sweep_ranks_and_counts`, `test_sweep_is_deterministic`, `test_sweep_rejects_bad_rank_by`, `test_sweep_records_windows_and_holdout_frac`). These 4-combo / 2-combo grids will already exercise the *current* sequential `sweep()`; they must stay green before any sweep.py change.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sweep.py
git commit -m "test(169): hoist sweep fixture signal to module level (picklable)"
```

---

### Task 3: Add the module-level worker functions in sweep.py

These are the picklable units a pool worker runs. `_evaluate_combo` is the pure per-combo computation (returns a small dict, deliberately NOT including `holdout_metrics`). `_evaluate_combo_pooled` wraps it in a 1-thread BLAS limit for the pool path only.

**Files:**
- Modify: `algua/backtest/sweep.py` (imports near top; new fns before `def sweep`)
- Test: `tests/test_sweep.py`

- [ ] **Step 1: Write a failing test for the worker fn**

Add to `tests/test_sweep.py`:

```python
def test_evaluate_combo_returns_record_without_holdout():
    from algua.backtest.sweep import _evaluate_combo, _override

    ov = _override(_momentum(), {"lookback": 20})
    rec = _evaluate_combo(
        ov, provider=SyntheticProvider(seed=3), start=START, end=END,
        windows=4, holdout_frac=0.2,
        universe_by_date=None, universe_name=None, universe_snapshots=None,
        rank_by="mean_sharpe",
    )
    # The rankable fields are present; the holdout never leaves the worker.
    assert set(rec) == {"config_hash", "n_windows", "stability", "score", "meta"}
    assert "holdout_metrics" not in rec and "holdout" not in rec
    assert rec["score"] == rec["stability"]["mean_sharpe"]
    assert set(rec["meta"]) == {
        "data_source", "snapshot_id", "timeframe", "seed", "code_hash",
        "dependency_hash", "period", "universe_name", "universe_snapshots",
    }
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_sweep.py::test_evaluate_combo_returns_record_without_holdout -v`
Expected: FAIL with `ImportError: cannot import name '_evaluate_combo'`.

- [ ] **Step 3: Add the imports**

In `algua/backtest/sweep.py`, the top imports become (add `functools`, `os`, `pickle`, the two `concurrent.futures` names, and `threadpool_limits`):

```python
from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import os
import pickle
from collections.abc import Collection, Mapping
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from threadpoolctl import threadpool_limits

from algua.backtest.engine import BacktestError
from algua.backtest.walkforward import walk_forward
from algua.contracts.types import DataProvider
from algua.portfolio.construction import ConstructionError, validate_construction_params
from algua.strategies.base import LoadedStrategy
```

- [ ] **Step 4: Add the worker functions**

Insert these two module-level functions in `algua/backtest/sweep.py` immediately **before** `def sweep(`:

```python
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_sweep.py::test_evaluate_combo_returns_record_without_holdout -v`
Expected: PASS.

- [ ] **Step 6: Run the quality gate**

Run: `uv run pytest tests/test_sweep.py -q && uv run ruff check algua/backtest/sweep.py && uv run mypy algua/backtest/sweep.py && uv run lint-imports`
Expected: all pass (existing sweep tests still green; `sweep()` itself unchanged so far).

- [ ] **Step 7: Commit**

```bash
git add algua/backtest/sweep.py tests/test_sweep.py
git commit -m "feat(169): add picklable per-combo sweep worker fns (BLAS-pinned pool wrapper)"
```

---

### Task 4: Add the `_run_combos` dispatcher (inline ≤1 / pool, with error wrapping)

**Files:**
- Modify: `algua/backtest/sweep.py` (new `_run_combos` before `def sweep`)
- Test: `tests/test_sweep.py`

- [ ] **Step 1: Write a failing test for the dispatcher (both branches + error type)**

Add to `tests/test_sweep.py`:

```python
def _run_kwargs():
    return dict(
        provider=SyntheticProvider(seed=3), start=START, end=END,
        windows=4, holdout_frac=0.2,
        universe_by_date=None, universe_name=None, universe_snapshots=None,
        rank_by="mean_sharpe",
    )


def test_run_combos_inline_single_combo():
    from algua.backtest.sweep import _override, _run_combos

    overridden = [_override(_momentum(), {"lookback": 20})]  # len==1 -> inline path
    results = _run_combos(overridden, _run_kwargs())
    assert len(results) == 1
    assert results[0]["score"] == results[0]["stability"]["mean_sharpe"]


def test_run_combos_pool_preserves_order():
    from algua.backtest.sweep import _override, _run_combos

    combos = [{"lookback": 20}, {"lookback": 30}, {"lookback": 40}]  # >1 -> pool path
    overridden = [_override(_momentum(), c) for c in combos]
    a = _run_combos(overridden, _run_kwargs())
    b = _run_combos(overridden, _run_kwargs())
    # Order preserved (map) and reproducible across runs (determinism under the pool).
    assert [r["config_hash"] for r in a] == [r["config_hash"] for r in b]
    assert len(a) == 3
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_sweep.py::test_run_combos_inline_single_combo tests/test_sweep.py::test_run_combos_pool_preserves_order -v`
Expected: FAIL with `ImportError: cannot import name '_run_combos'`.

- [ ] **Step 3: Add the dispatcher**

Insert in `algua/backtest/sweep.py` immediately **before** `def sweep(`:

```python
def _run_combos(
    overridden: list[LoadedStrategy], eval_kwargs: dict[str, Any]
) -> list[dict[str, Any]]:
    """Evaluate every pre-built combo strategy via walk_forward, returning records in COMBO ORDER.

    A single combo (or a single-core host) runs inline — no pool overhead, full BLAS threads.
    Otherwise a ProcessPoolExecutor fans the combos out; `executor.map` preserves input order so the
    stable rank downstream stays reproducible.

    Errors:
      * A combo's own failure (e.g. walk_forward raising BacktestError) is delivered back through
        `map` and propagates with its OWN type — `except BacktestError: raise` keeps it unwrapped.
      * Pool/pickle INFRASTRUCTURE failures (a worker segfault/OOM -> BrokenExecutor; a non-picklable
        arg -> pickle.PicklingError) are re-raised as BacktestError so the CLI's
        @json_errors(ValueError, LookupError, BacktestError) still emits a JSON envelope.
      * Ordering matters: the domain re-raise MUST precede the infrastructure catch, or a worker
        BacktestError would be double-wrapped into "parallel sweep failed".
    """
    n_workers = min(os.cpu_count() or 1, len(overridden))
    if n_workers <= 1:
        return [_evaluate_combo(ov, **eval_kwargs) for ov in overridden]

    worker = functools.partial(_evaluate_combo_pooled, **eval_kwargs)
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            return list(executor.map(worker, overridden))
    except BacktestError:
        raise
    except (BrokenExecutor, pickle.PicklingError) as exc:
        raise BacktestError(f"parallel sweep failed: {exc}") from exc
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_sweep.py::test_run_combos_inline_single_combo tests/test_sweep.py::test_run_combos_pool_preserves_order -v`
Expected: PASS (the pool test runs real worker processes on a multicore box).

- [ ] **Step 5: Run the quality gate**

Run: `uv run pytest tests/test_sweep.py -q && uv run ruff check algua/backtest/sweep.py && uv run mypy algua/backtest/sweep.py && uv run lint-imports`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add algua/backtest/sweep.py tests/test_sweep.py
git commit -m "feat(169): add _run_combos dispatcher (inline/pool + BacktestError wrapping)"
```

---

### Task 5: Wire `sweep()` to the parallel dispatcher

Replace the sequential loop + WalkForwardResult-based assembly in `sweep()` with: parent pre-pass (`_override` all combos → fail-fast) → `_run_combos` → assemble `records` in combo order, `meta` from `results[0]`.

**Files:**
- Modify: `algua/backtest/sweep.py:185-228` (the body of `sweep()` from `combos = _combos(grid)` to the `return SweepResult(...)`)
- Test: `tests/test_sweep.py` (existing tests are the regression guard)

- [ ] **Step 1: Replace the sweep() body**

In `algua/backtest/sweep.py`, replace everything from `    combos = _combos(grid)` through the end of the `return SweepResult(...)` statement with:

```python
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
        universe_name=meta["universe_name"],
        universe_snapshots=meta["universe_snapshots"],
    )
```

Note: the `records: list[dict[str, Any]] = []` declaration, the `meta = None` line, the `for combo in combos:` loop, and the `assert meta is not None` line from the old body are all removed (replaced by the block above).

- [ ] **Step 2: Run the full sweep test file**

Run: `uv run pytest tests/test_sweep.py -q`
Expected: PASS — all existing tests (`test_sweep_ranks_and_counts` [4 combos → pool], `test_sweep_is_deterministic` [4 combos → pool, a==b], `test_sweep_rejects_bad_rank_by`, `test_sweep_records_windows_and_holdout_frac` [2 combos → pool]) plus the Task 3/4 tests.

- [ ] **Step 3: Verify the docstring invariant still holds (holdout withheld)**

Confirm the `sweep()` docstring's "holdout DELIBERATELY NOT recorded" paragraph is unchanged and still accurate — `_evaluate_combo` never returns `holdout_metrics`, so the property is now enforced at the worker boundary too. No edit needed; just read it.

- [ ] **Step 4: Commit**

```bash
git add algua/backtest/sweep.py
git commit -m "feat(169): parallelize sweep across combos via ProcessPoolExecutor"
```

---

### Task 6: Add the combo-error-surfaces-as-BacktestError test

A worker-side `walk_forward` failure must surface as `BacktestError` (the CLI-wrappable type), not a raw `BrokenProcessPool` or a partial result. Worker code can't be monkeypatched across the process boundary, so trigger a *real* failing config: too many windows for the period makes `walk_forward` raise `BacktestError` ("not enough bars").

**Files:**
- Test: `tests/test_sweep.py`

- [ ] **Step 1: Write the test**

Add to `tests/test_sweep.py`:

```python
def test_sweep_combo_error_surfaces_as_backtest_error():
    from algua.backtest.engine import BacktestError

    # >1 combo so this runs through the pool; `windows` far too large for the period forces
    # walk_forward to raise BacktestError ("not enough bars"). It must come back as BacktestError
    # (CLI-wrappable), not a BrokenProcessPool and not a partial result.
    with pytest.raises(BacktestError):
        sweep(_momentum(), SyntheticProvider(seed=3), START, END,
              grid={"lookback": [20, 40]}, windows=500, holdout_frac=0.2)
```

- [ ] **Step 2: Run it to verify it passes**

Run: `uv run pytest tests/test_sweep.py::test_sweep_combo_error_surfaces_as_backtest_error -v`
Expected: PASS (raises `BacktestError`). If it instead raises `BrokenProcessPool`, the `except` ordering in `_run_combos` is wrong — fix Task 4 Step 3.

- [ ] **Step 3: Sanity-check the error path is real**

Run: `uv run python -c "from algua.backtest.walkforward import _segment_bounds; _segment_bounds(100, 500, 0.2)"` (or inspect `walkforward.py`) to confirm `windows=500` raises a `ValueError`/`BacktestError` for the synthetic period length. If `walk_forward` raises `ValueError` rather than `BacktestError` for this input, change the test to `pytest.raises((ValueError, BacktestError))` and note it — both are CLI-wrappable (`json_errors(ValueError, LookupError, BacktestError)`), so the JSON contract holds either way.

- [ ] **Step 4: Commit**

```bash
git add tests/test_sweep.py
git commit -m "test(169): combo error surfaces as BacktestError through the pool"
```

---

### Task 7: Full quality gate + CLI smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the complete gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass. (`lint-imports` "0 broken" — `sweep.py`'s new imports are stdlib + `threadpoolctl`, no algua cross-module boundary added.)

- [ ] **Step 2: CLI smoke — a real multi-combo sweep emits JSON**

Run: `uv run algua backtest sweep cross_sectional_momentum --demo --param lookback=10,20,30,40 --windows 4 | head -c 400`
Expected: a JSON object with `"ok": true` and a `ranked` array — proving the pooled path runs end-to-end through the CLI and the JSON envelope is intact.

- [ ] **Step 3: CLI smoke — error still emits JSON (not a raw traceback)**

Run: `uv run algua backtest sweep cross_sectional_momentum --demo --param lookback=10,20 --windows 500; echo "exit=$?"`
Expected: a JSON error envelope (`"ok": false` with a message), non-zero exit — NOT a Python traceback. Confirms pool/combo failures stay inside the JSON contract.

- [ ] **Step 4: Commit any final touch-ups (if the gate required none, skip)**

```bash
git add -A && git commit -m "chore(169): quality gate green for parallel sweep" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:**
- ProcessPoolExecutor over combos, standard pickle → Tasks 3-5. ✓
- Parent pre-pass fail-fast (`_override` all combos) → Task 5 Step 1. ✓
- `n_workers = min(os.cpu_count() or 1, len(combos))`, inline when ≤1 → Task 4 Step 3. ✓
- BLAS thread-pinning via `threadpoolctl` in the pool worker only → Task 3 (`_evaluate_combo_pooled`) + Task 1 (dep). ✓
- Error wrapping with strict ordering (domain re-raise before infra catch) → Task 4 Step 3 + Task 6. ✓
- Worker returns no `holdout_metrics` → Task 3 (`_evaluate_combo`) + test. ✓
- Combo-order records + `meta` from `results[0]` (reproducible stable rank) → Task 5. ✓
- `SweepResult` / `sweep()` signature unchanged → Task 5 reuses the same constructor + signature. ✓
- Test fixture hoisted to module level (picklable) → Task 2. ✓
- Both dispatch branches + combo-error-type tested → Tasks 4, 6. ✓
- `threadpoolctl` declared as a direct dep → Task 1. ✓

**Placeholder scan:** none — every code/command step is concrete.

**Type consistency:** `_evaluate_combo` returns `{config_hash, n_windows, stability, score, meta}`; `_run_combos` returns `list[dict]`; `sweep()` reads those keys + `meta[...]` consistently across Tasks 3-5. `_evaluate_combo_pooled(**kwargs)` forwards the same kwarg set `_evaluate_combo` accepts. Names match (`_run_combos`, `_evaluate_combo`, `_evaluate_combo_pooled`) everywhere referenced.
