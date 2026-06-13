# Parallelize backtest sweep (#169)

**Status:** GATE-1 CLOSED — round 1 (Codex deep + Gemini broad) accepted findings folded in;
round 2 (Codex) returned "no accepted findings remain". Pending user sign-off, then writing-plans.
Accepted: CLI JSON-contract wrapping (with strict exception ordering), BLAS thread-pinning via
threadpoolctl, precise concurrency wording, documented worker precondition. Declines recorded below.
**Issue:** #169 — *Parallelize backtest sweep: 200 sequential walk-forwards sit on the research
loop's critical path* (P3, `enhancement`, `severity:low`, `area:backtest`)
**Date:** 2026-06-13

## Problem

`algua/backtest/sweep.py:189-205` runs each grid combo's `walk_forward()` sequentially on one core,
with the grid capped at `_MAX_COMBOS = 200`. A full sweep is hours of wall-clock and sits directly
on the autonomous research loop's critical path, so the parallelization pays for itself in loop
throughput. Expected win: ~4–8× on a typical multi-core host.

## Ordering constraint (#161) — resolved as non-blocking

The issue says "land after #161 (holdout reservation)". Verified this is a *soft* ordering
preference, not a hard dependency:

- The holdout burn lives **only** in `algua/cli/research_cmd.py` (the `research promote` path).
  `walk_forward` and `sweep` have **zero** references to the holdout DB — they are pure compute
  (confirmed by grep over `algua/backtest/`).
- `sweep` parallelizes those pure-compute walk-forwards. It never writes `holdout_evaluations`.
- #161 modifies a **disjoint** code path (`research_cmd.py` + `store.py` reservation methods) that
  this change does not touch.

The literal failure mode the constraint guards — "concurrent walk-forwards double-burning holdouts"
— cannot arise from `sweep`, because `sweep` doesn't burn at all. The issue author's own words:
"Sweep itself doesn't burn holdouts directly." User signed off on proceeding with #169 now
(2026-06-13); #161 lands separately on its own timeline.

## Picklability — verified empirically

`ProcessPoolExecutor` ships the callable + args to each worker via `pickle`. Round-trip tested:

| Object | Result |
|---|---|
| `LoadedStrategy` (real, via `load_strategy`) | OK — 825 bytes (fn refs resolve by qualified module name) |
| `SyntheticProvider` (demo) | OK — 75 bytes |
| `StoreBackedProvider` + `DataStore` (snapshot) | OK — `DataStore` holds a `Path`, opens sqlite lazily (no handle to pickle) |
| `walk_forward` fn | OK |

Real strategies are **always** module-level: `load_strategy` only ever binds a module's top-level
`signal`/`signal_panel`. So the worker can re-import everything it needs. No "re-load by name in the
worker" machinery is required (the issue floated it as a fallback; unneeded).

## Approach (chosen)

`concurrent.futures.ProcessPoolExecutor` over combos, standard `pickle`. Inside `sweep()`:

1. **Parent pre-pass:** `overridden = [_override(strategy, c) for c in combos]`.
   Keeps `_override`'s fail-fast validation (bad signal key, or invalid construction param →
   `ValueError`) in the parent, **before** any process spawns — preserving today's error behavior
   exactly. (`_combos`' grid-too-large `BacktestError` already fires earlier, also in the parent.)

2. **Worker count:** `n_workers = min(os.cpu_count() or 1, len(combos))`. Auto; no CLI flag, no env
   override (per user decision — keep the surface clean).

3. **One shared worker fn** `_evaluate_combo(overridden, *, provider, start, end, windows,
   holdout_frac, universe_by_date, universe_name, universe_snapshots, rank_by) -> dict`
   (module-level → picklable). It runs `walk_forward(overridden, ...)` and returns a small dict:
   `{"config_hash", "n_windows", "stability", "score", "meta": {...}}`.
   - **It does NOT return `holdout_metrics`** — those stay in the worker process and are GC'd.
     Defense-in-depth on the single-use-holdout discipline + a smaller cross-process payload.
   - `meta` carries the combo-independent fields (`data_source`, `snapshot_id`, `timeframe`,
     `seed`, `code_hash`, `dependency_hash`, `period`, `universe_name`, `universe_snapshots`).

4. **Dispatch:**
   - `n_workers <= 1` (single combo, or single-core host) → call `_evaluate_combo` **inline** in a
     list comprehension. No pool overhead, clean tracebacks, full BLAS threads for the lone combo.
   - else → run under a `ProcessPoolExecutor(max_workers=n_workers)` with
     `executor.map(worker, overridden)`, where `worker` is a module-level
     `_evaluate_combo_pooled` that wraps `_evaluate_combo` in `threadpool_limits(limits=1)` (see
     thread-pinning below). `map` **preserves input order**; on the first worker exception it
     re-raises that exception when iteration reaches it.
   - **Error wrapping (GATE-1, HIGH):** the whole pool block is wrapped so that pool-/pickle-level
     failures — `BrokenProcessPool` (worker segfault/OOM/kill), `pickle.PicklingError`, and other
     `concurrent.futures` errors — are re-raised as
     `BacktestError("parallel sweep failed: ...") from exc`. The CLI's
     `@json_errors(ValueError, LookupError, BacktestError)` only wraps those three, so without this
     a pool crash would leak a raw traceback and break the "every data command emits JSON"
     contract. When `__cause__` is a `PicklingError` the message names the offending input (so a
     non-picklable strategy/provider is actionable — no separate pre-flight pickle gate needed).
     Combo-level errors (`_override`'s `ValueError`, `walk_forward`'s `BacktestError`) keep their
     own type and propagate unwrapped, exactly as today.
     - **Exception ordering (GATE-1 r2, load-bearing):** the handler must re-raise domain errors
       *before* catching infrastructure errors — `except BacktestError: raise` (and `ValueError`)
       **first**, then `except (BrokenProcessPool, BrokenExecutor, pickle.PicklingError, ...) as
       exc: raise BacktestError(...) from exc`. A broad `except Exception` would double-wrap a
       worker-raised `BacktestError` (delivered back through `map`) into "parallel sweep failed".
       Note a non-picklable arg can surface as `AttributeError`/`TypeError` from the pickling layer,
       not only `PicklingError` — but the documented picklability precondition + module-level loader
       make this a non-issue in production (caught generically by the infra clause, still wrapped).
   - **BLAS thread-pinning (GATE-1, HIGH):** numpy here is OpenBLAS with 24 threads. Without
     pinning, `n_workers` processes × 24 BLAS threads massively oversubscribes the cores. Each pool
     worker runs `walk_forward` inside `threadpoolctl.threadpool_limits(limits=1)` (already an
     installed dependency; runtime limit, so it works under the default `fork` start method and
     needs no env-before-import dance). With 1 compute thread per worker, `n_workers = cpu_count`
     is then correctly sized rather than oversubscribed. The inline path does NOT limit threads —
     a single combo should use all cores.
   - The inline and pool branches call the **same** `_evaluate_combo` computation; they differ only
     in **object isolation** (inline passes live references; the pool passes pickled copies) and in
     thread-pinning. Because sweep inputs are required to be side-effect-free and deterministic
     (worker precondition below), the two produce identical results.

5. **Assemble (logic unchanged):** build `records` in **combo order** — `zip(combos, results)`,
   attaching `"params": combo` to each — take `meta` from `results[0]`, run `_rank_records`, and
   return the same `SweepResult`.

### Worker precondition (GATE-1)

Sweep's parallel path requires that the `strategy`, `provider`, and `universe_by_date` inputs are
**picklable, deterministic, and read-only / side-effect-free**. Algua's real inputs satisfy this:
strategy signal/construct fns are pure module-level functions; `SyntheticProvider` carries a fixed
seed (pickled with it); `StoreBackedProvider` reads an immutable parquet snapshot; `universe_by_date`
is plain dates → symbol lists. A provider with caches/cursors/RNG/lazy network state would violate
the precondition — but none exists in the codebase, and adding one would be the thing to revisit,
not this design. Documented as the sweep contract rather than enforced with machinery (the only
non-picklable strategies are *test* closures, fixed by the test refactor below).

### Fail semantics (precise)

The pool returns **no partial result**: it either returns a full ranked sweep or raises. On a combo
error it raises the **earliest failing combo in `map` order**. Unlike the sequential loop, other
already-dispatched combos may have *executed* before the raise — that is observationally identical
because the combos are pure compute (no holdout burn, no writes, no external calls). "Fail-on-first
*execution*" is therefore relaxed to "no partial result; earliest failing combo by map order."

### Why combo order matters

`_rank_records` is a **stable** sort whose documented tie-break (equal `score` + equal
`std_sharpe`) "preserves the original order" = combo/grid order. If parallel completion order leaked
into `records`, tie-broken ranking would become non-deterministic across runs. Building `records`
in combo order (which `executor.map` guarantees by preserving input order) keeps ranking
bit-for-bit reproducible. `meta` is taken from `results[0]` for parity with today
(`meta = first wf`), though every combo yields identical meta fields anyway.

### Rejected alternatives

- **`cloudpickle` / `loky`.** Would transparently ship closures/lambdas to workers. But production
  strategies are *always* module-level (the loader guarantees it); the only things needing closure
  support are test fixtures. Adding a dependency to support a capability production never uses is
  overengineering. Chosen: standard `pickle` + module-level test signal (below).
- **`ThreadPoolExecutor`.** `walk_forward` is CPU-bound Python under the GIL; threads would not
  deliver the 4–8× win. The issue explicitly calls for processes.
- **CLI `--jobs` flag / `ALGUA_SWEEP_WORKERS` env hatch.** Declined by user — auto worker count is
  deterministic and the surface stays clean. Add later only if an operator need appears.

## Testing

- **Refactor `tests/test_sweep.py::_momentum`:** hoist its nested `def signal` to a **module-level**
  `_momentum_signal(view, params)` so the fixture pickles. This only makes the test strategy
  faithful to production (real signals are module-level). The `_momentum()` builder then binds
  `signal_fn=_momentum_signal`.
- **Determinism (existing, now stronger):** `test_sweep_determinism` (`a.to_dict() == b.to_dict()`
  over a `>1` combo grid) now also guards that worker completion order does not leak into the
  ranking. On a multicore box this exercises the real pool path.
- **Errors (existing, still valid):** bad-`rank_by` and bad-key/grid-too-large all raise in the
  parent (`rank_by` check and `_override`/`_combos` pre-pass), unaffected by the pool.
- **New — both dispatch branches:** a single-combo grid (`len==1` → inline path) and a multi-combo
  grid (`>1` → pool path on a multicore box) each produce a correct ranked `SweepResult`, and the
  multi-combo result is identical across two invocations (reproducibility under the pool).
- **New — combo error surfaces as the right type:** a grid whose combos make `walk_forward` raise a
  `BacktestError` (a real failing config, e.g. windows too large for the period) surfaces a
  `BacktestError` through the pool — not a `BrokenProcessPool` and not a partial result. (A genuine
  failing config, because worker-side code cannot be monkeypatched across the process boundary.)

Worker-side exceptions cannot be monkeypatched across the process boundary; surfacing of a genuine
worker failure rests on the `executor.map` re-raise contract, the `BacktestError`-wrapping of
pool/pickle failures, and the parent-side error tests above.

## Invariants & boundaries

- `sweep()` signature and the `SweepResult` contract are **unchanged** → `backtest_cmd.py`,
  `--track` / `log_sweep`, and `research promote`'s breadth measurement are untouched.
- The holdout is still **never** recorded or revealed by `sweep` (the docstring invariant holds;
  the worker doesn't even return `holdout_metrics`).
- `algua/contracts` / `algua/features` purity untouched. New imports in `sweep.py` are stdlib
  (`os`, `concurrent.futures`, `functools`) plus `threadpoolctl` (already installed; add it as an
  explicit project dependency in `pyproject.toml` since `sweep.py` now imports it directly rather
  than relying on it transitively) — no cross-module boundary change (`lint-imports` stays
  "0 broken").
- Gate green after every commit:
  `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

## Risks / notes

- **BLAS oversubscription (FIXED):** OpenBLAS = 24 threads; each pool worker is pinned to 1 thread
  via `threadpoolctl.threadpool_limits(1)`, so `cpu_count` workers = `cpu_count` compute threads on
  `cpu_count` cores. (Was the panel's strongest perf finding; resolved, not deferred.)
- **Start method:** default `fork` on Linux is safe here — `DataStore` opens sqlite lazily (no
  handle inherited at `sweep()` time), and BLAS is pinned at *runtime* in the worker (not via
  env-before-import), so fork is correct. Not forcing `spawn` (would add per-worker startup cost
  for no correctness gain).
- **Closure strategies can't be swept in parallel** — but production never has them (loader binds
  module-level `signal`). Documented as a worker constraint; the test refactor aligns fixtures.
- **Fundamentals sweep already unsupported (pre-existing):** `walk_forward` has no fundamentals
  provider parameter, so a `needs_fundamentals` strategy cannot be swept today regardless of
  parallelism. This change adds no new picklability surface there (e.g. for
  `StoreBackedFundamentalsProvider`).
- **Memory:** each worker re-reads the immutable parquet snapshot (windowed bars, bounded). A
  shared-memory/mmap data server (panel suggestion) is a large architecture change unjustified at
  P3 for the current single-host operator; deferred to the existing read-path-scale follow-ups if a
  real OOM ever appears.

## GATE-1 review outcome (multi-model, round 1)

Panel: Codex (GPT-5, deep lens) + Gemini 2.5 Flash (broad lens). Scaled to a P3 perf refactor;
strong cross-model agreement on the load-bearing items.

**Accepted → folded in above:**
- CLI JSON-contract: wrap `BrokenProcessPool`/`PicklingError`/other pool errors as `BacktestError`
  (Codex HIGH, Gemini MEDIUM) — else a worker crash leaks a raw traceback past `json_errors`.
- BLAS thread-pinning via `threadpoolctl` per worker (Codex HIGH, Gemini CRITICAL) — confirmed real
  (OpenBLAS 24 threads).
- Precise concurrency wording: "no partial result; earliest failing combo by map order" not
  "fail-on-first execution"; inline-vs-pool is "same computation, different object isolation" not
  "identical by construction" (Codex MEDIUM/LOW). Test both branches.
- Worker precondition documented: picklable + deterministic + read-only inputs (Codex HIGH).
- Note fundamentals sweep already unsupported (Codex/Gemini MEDIUM) — no new picklability surface.

**Declined (rationale):**
- CLI `--jobs` / `ALGUA_SWEEP_WORKERS` env / admin worker cap (Codex HIGH, Gemini HIGH/CRITICAL):
  user explicitly declined the control surface; with BLAS pinned to 1 thread/worker there is no
  oversubscription left to cap, so `cpu_count` is correctly sized. Shared-host fairness is a
  deployment-policy decision the user owns (single operator now; VPS scale-out later).
- Shared-memory/mmap data server (Gemini CRITICAL): out-of-proportion architecture change at P3;
  deferred (read-path-scale follow-ups) pending a real OOM.
- Per-worker RNG reseed (Gemini HIGH): no global RNG in the compute path (verified); determinism
  comes from the pickled provider seed + pure strategy fns.
- meta-consistency guard / compute stamps once in parent (Codex MEDIUM): `code_hash`/
  `dependency_hash` derive from immutable strategy source + installed deps — process-invariant
  within a run, so `results[0]` is safe.
- `spawn`/`forkserver` over `fork` (Codex MEDIUM): unneeded once BLAS is pinned at runtime; fork is
  safe (no open handles at `sweep()` time).
- Separate pre-flight pickle round-trip gate (Codex MEDIUM): redundant — the `PicklingError` path
  already produces an actionable `BacktestError`.

## Out of scope

- #161 atomic holdout reservation (disjoint; lands separately).
- Parallelizing `walk_forward`'s internal windows (a different, smaller axis).
- Configurable worker count / env override (declined above).
- Shared-memory data provider (deferred; read-path-scale follow-ups).
