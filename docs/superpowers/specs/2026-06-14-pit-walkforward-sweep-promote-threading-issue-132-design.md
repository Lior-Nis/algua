# PIT-sidecar threading through walk_forward / sweep / promote — issue #132 (slice 4)

**Status:** design
**Date:** 2026-06-14
**Issue:** #132 — non-tabular data (news / fundamentals): point-in-time consumption lanes.

## Context

The fundamentals (#132 / PR#154) and news (#132 / PR#203) **as-of signal lanes** are merged: a
strategy declares `needs_fundamentals` or `needs_news` and the **backtest engine** (`simulate` /
`run`) injects the PIT-correct sidecar per bar. But the providers are threaded **only** into the
single-shot backtest. `walk_forward`, `sweep`, and `research promote` do **not** accept a PIT
provider — PR#203 added a `_reject_pit_sidecar` guard that **fails closed** for any
`needs_fundamentals` / `needs_news` strategy in those paths.

**The consequence is a funnel blocker.** An agent reaches `candidate` only via `research promote`,
which runs `walk_forward` (holdout) and requires **measured search breadth** from `backtest sweep`.
Both refuse a PIT-sidecar strategy today. So a `needs_fundamentals` / `needs_news` strategy is
**stuck at `backtested`** — it can never be promoted, paper-traded, or forward-tested. (#180's
GATE-2 flagged exactly this.)

This slice threads the providers through `walk_forward`, `sweep`, and the three CLI commands that
drive them, removing the guard and **relying on the engine's existing fail-closed** ("declares
`needs_X` but no `X_provider` supplied"). It unblocks the research funnel for **both** PIT lanes at
once (they share every seam).

## Goals

1. `walk_forward` and `sweep` accept `fundamentals_provider` / `news_provider` and thread them to
   `build_portfolio` (= `simulate`, which already consumes them).
2. **Remove the `needs_fundamentals` / `needs_news` blockers in `promotion_preflight`** so
   `backtested → candidate` is allowed for a PIT strategy (the actual funnel enabler — see §0).
3. `backtest walk-forward`, `backtest sweep`, and `research promote` CLIs accept
   `--fundamentals-snapshot` / `--news-snapshot`, build the `StoreBacked*Provider`, and pass it
   through — with the same misuse guard the `backtest run` CLI already has (snapshot given but the
   strategy doesn't declare the matching `needs_*` → error), AND the reverse fail-closed (a
   `needs_X` strategy promoted without the matching snapshot → error, raised *before* any holdout
   reservation).
4. A `needs_news` / `needs_fundamentals` strategy can be swept (breadth measured), walk-forwarded,
   and **`research promote`d to `candidate`** end-to-end, with the PIT snapshot recorded in the
   result + gate-evaluation provenance.
5. The parallel sweep (#169) keeps working: the PIT provider must be picklable and flow to workers.

## Non-goals (deferred, documented)

- **Vectorized PIT fast path** — the engine still forces the per-bar loop for a PIT strategy
  (`signal_panel` is rejected at load for `needs_*`). Sweep/walk-forward of a PIT strategy are
  therefore loop-speed, and each parallel-sweep worker re-reads the same PIT snapshot parquet
  (the provider re-reads on each `get_*`); correct but IO/CPU-heavier than a bars-only sweep. No
  worker-local caching or `--max-workers` cap this slice (the #169 worker-cap was already
  declined); a `signal_panel`-style PIT fast path + worker caching is a separate follow-up if
  fixture/perf pressure shows up.
- **Paper/live PIT lane** — still fail-closed; lifting it is the next slice (now reachable, since a
  PIT strategy can finally become a `candidate`).
- **Unified PIT-context arg** (a strategy using both news AND fundamentals) — still rejected at
  load; out of scope.
- Cross-snapshot retraction reconciliation, fuzzy dedup, vendor ingest, cloud storage — untouched.

## Design

### 0. Remove the PIT promotion blockers — `algua/registry/promotion.py` (THE enabler)

`promotion_preflight` contains a **second, independent** guard (separate from `_reject_pit_sidecar`)
that runs **pre-peek**, before `walk_forward`:

```python
if _loaded is not None and _loaded.config.needs_fundamentals:
    raise ValueError(f"strategy {name!r} declares needs_fundamentals; it cannot be promoted past "
                     f"backtested until the paper/live fundamentals lane is built (#132)")
if _loaded is not None and _loaded.config.needs_news:
    raise ValueError(f"strategy {name!r} declares needs_news; it cannot be promoted past "
                     f"backtested until the paper/live news lane is built (#132)")
```

These block `backtested → candidate` for any PIT strategy. **Threading providers through
`walk_forward`/`sweep` is useless unless these are removed** — the strategy would hit this
`ValueError` before `walk_forward` ever runs. **Remove both blocks.** `backtested → candidate` is a
*research* edge (a candidate isn't trading), so blocking it was over-conservative; the
*trading*-lane guards (`assert_tradable_without_fundamentals` / `assert_tradable_without_news` in
`strategies/base.py`, called at paper/live load) **remain** — paper/live PIT trading is still
deferred. The corresponding `tests/test_promotion_needs_news.py` (and any fundamentals equivalent)
that assert the block must be rewritten to assert a PIT strategy now reaches the gate (and is
refused only for a missing PIT snapshot / failing holdout, not for being PIT).

### 1. `walk_forward` — `algua/backtest/walkforward.py`

- Add keyword params `fundamentals_provider: FundamentalsProvider | None = None` and
  `news_provider: NewsProvider | None = None` (import the protocols from
  `algua.contracts.types`).
- **Update the `build_portfolio(strategy, provider, start, end, universe_by_date=...)` call** to
  also pass `fundamentals_provider=fundamentals_provider, news_provider=news_provider`
  (`build_portfolio` is the public alias of `simulate`, which already accepts them). Without this
  the providers never reach the engine.
- Add `fundamentals_snapshot` / `news_snapshot` fields to `WalkForwardResult` (default `None`),
  populated `getattr(fundamentals_provider, "snapshot_id", None)` /
  `getattr(news_provider, "snapshot_id", None)` — mirroring `BacktestResult` so a PIT
  walk-forward records *which* snapshot produced it (reproducibility).
- **Remove `_reject_pit_sidecar` and its call.** The engine's `simulate` already raises
  `BacktestError("strategy {name!r} declares needs_X but no X_provider was supplied (fail
  closed)")` when `needs_X` and the provider is `None` — that is the single source of truth for
  the missing-provider failure. No WF/sweep-level duplicate is needed.

### 2. `sweep` — `algua/backtest/sweep.py`

- Add the same two provider kwargs to `sweep`.
- Add them to the `eval_kwargs` dict that is fanned out to workers, AND add explicit
  `fundamentals_provider` / `news_provider` params to **`_evaluate_combo`** (it unpacks
  `eval_kwargs` by name — a kwarg with no matching param is a `TypeError`); forward them in its
  `walk_forward(...)` call. `_evaluate_combo_pooled` already passes `**kwargs` through.
- Add `fundamentals_snapshot` / `news_snapshot` fields to `SweepResult` (default `None`),
  populated from the providers — mirroring `WalkForwardResult` / `BacktestResult`.
- Drop the `_reject_pit_sidecar(strategy, "sweep")` call and the now-unused import.
- **Picklability:** `StoreBacked{Fundamentals,News}Provider` hold a `DataStore` (paths only) +
  `snapshot_id` — picklable, exactly like the bars `StoreBackedProvider` already passed to workers.
  The existing parent-side pickle preflight (`pickle.dumps(worker)`) binds the providers via the
  `functools.partial`, so it automatically covers them — a non-picklable provider already surfaces
  as the JSON-safe `BacktestError`. No new preflight code.

### 3. CLI — `algua/cli/backtest_cmd.py` (`walk_forward_cmd`, `sweep_cmd`)

Both currently call `walk_forward(...)` / `sweep(...)` with no PIT provider. The `run` command in
the **same file** already has the full `--fundamentals-snapshot` / `--news-snapshot` wiring +
misuse guards + `StoreBacked*Provider` construction (`StoreBackedFundamentalsProvider`,
`StoreBackedNewsProvider` are already imported). Mirror that block in both commands:

- Add `--fundamentals-snapshot` / `--news-snapshot` options.
- After `resolve_eval_inputs`: the two misuse guards (`if fundamentals_snapshot and not
  strategy.config.needs_fundamentals: raise ValueError(...)`, same for news).
- Build the providers (or `None`) and pass `fundamentals_provider=` / `news_provider=` into the
  `walk_forward(...)` / `sweep(...)` call.

### 4. CLI — `algua/cli/research_cmd.py` (`promote`)

- Add `--fundamentals-snapshot` / `--news-snapshot` options.
- **Two guards, raised up front (before `reserve_holdout` / `holdout_window`)** so a deterministic
  operator error never creates a pending reservation:
  - misuse: snapshot given but the strategy doesn't declare the matching `needs_*` → error;
  - fail-closed: `needs_fundamentals` (resp. `needs_news`) but no `--fundamentals-snapshot`
    (resp. `--news-snapshot`) → error. (The engine would also fail closed inside `walk_forward`,
    but that is *after* the reservation; reject earlier.)
- Build the providers and pass them **only** into the `walk_forward(...)` call. `promotion_preflight`
  needs no PIT provider (its `verify_signal_panel_parity` is a no-op for a PIT strategy — `needs_*`
  strategies have no `signal_panel`), and `holdout_window` reads only the bar date-index. So the
  PIT provider is threaded at exactly one point in `promote`.
- Pass the PIT snapshot ids through to `run_gate` → `record_gate_evaluation` (see §6) so the
  durable promotion audit row records which snapshot produced the passing holdout.

### 5b. `research dormant-sweep` — honest skip, not threaded

`dormant-sweep` (the #125 advisory stability screen) runs `walk_forward` over **every** dormant
strategy in one invocation and currently skips PIT strategies with reason
`"{sidecar}: walk-forward lane not wired"`. After this slice that reason is false, but
dormant-sweep takes a single `--snapshot` and cannot carry a *per-strategy* fundamentals/news
snapshot across a heterogeneous pool — full threading there needs its own design. So: **keep
skipping** PIT strategies in dormant-sweep, but update the skip reason to be accurate, e.g.
`"{sidecar}: dormant-sweep takes no per-strategy PIT snapshot — re-audition individually via
backtest walk-forward/research promote --{news,fundamentals}-snapshot"`. Update its test
accordingly. (Per-strategy PIT snapshots in dormant-sweep = explicit follow-up.)

### 6. Gate-evaluation audit provenance — `algua/registry/store.py`, `algua/registry/promotion.py`

`gate_evaluations` already records `data_source` + bars `snapshot_id` for each promotion decision.
A PIT promotion's decision also depends on the fundamentals/news snapshot, which is currently
unrecorded — so a candidate promoted off a later-corrupted news snapshot can't be traced. Add two
**additive nullable** columns `fundamentals_snapshot` / `news_snapshot` to `gate_evaluations`
(SCHEMA_VERSION bump, additive — no backfill needed; old rows are `NULL`), thread them through
`run_gate(...)` → `record_gate_evaluation(...)`, and populate them in `promote` from the providers'
`snapshot_id`. (Verify in planning that `gate_evaluations` indeed stores the bars `snapshot_id`
today; mirror that column's plumbing exactly.) Add the new params to the
`record_gate_evaluation` Protocol + store impl as **optional keyword-only**
(`fundamentals_snapshot: str | None = None`, `news_snapshot: str | None = None`) so existing
callers/tests that don't pass them keep working.

### 5. Holdout single-use identity — UNCHANGED

Per the design decision: the burn identity stays `(data_source, bars-snapshot, period,
holdout-window)` and continues to **exclude** the PIT snapshot (as it already excludes the
universe). Promoting on the same bars OOS window with a *different* news/fundamentals snapshot is
the **same** burn → refused. This is the conservative, strongest multiple-testing posture and needs
**no schema change**. (Documented so a future reader knows it's deliberate, not an oversight.)

## Correctness / safety

- **Fail-closed preserved:** removing `_reject_pit_sidecar` does not open a hole — a PIT strategy
  run through WF/sweep/promote *without* the matching provider still fails closed, now via the
  engine's `simulate` check (one source of truth). A test pins this for each path.
- **No look-ahead change:** the per-bar `knowable_at <= t` mask, the `t→t+1` shift, and the holdout
  segmentation are all unchanged — this slice only *delivers the provider* to the same engine code
  that already masks correctly. Each `walk_forward` window and the holdout get the as-of-correct
  sidecar because masking happens per bar inside `simulate`.
- **Parallel determinism:** the providers are combo-independent (same snapshot for every combo),
  bound once into the worker partial; combo ordering / holdout-non-crossing (#169) are unaffected.

## Testing

- **`walk_forward`:** a `needs_news` (and a `needs_fundamentals`) strategy walk-forwards
  successfully **with** the provider and the result carries the right `news_snapshot` /
  `fundamentals_snapshot`; **without** the provider it raises `BacktestError` mentioning `needs_X` /
  "no X_provider" (fail closed via the engine).
- **`sweep`:** same with-/without-provider pair; the `SweepResult` carries the PIT snapshot; plus a
  **parallel** sweep (>1 combo) of a PIT strategy completes (exercises provider picklability through
  the worker pool). An explicit `pickle.dumps(StoreBackedNewsProvider(...))` /
  `StoreBackedFundamentalsProvider(...)` unit test makes the picklability contract visible.
- **Promotion blocker removal:** rewrite `tests/test_promotion_needs_news.py` (+ add a
  fundamentals equivalent) — a PIT strategy at `backtested` now passes `promotion_preflight`
  (reaches the gate) rather than being refused for being PIT.
- **Guard test rewrite:** `tests/test_wf_sweep_pit_guard.py` currently asserts PIT strategies are
  *rejected* in WF/sweep — that behavior is intentionally replaced. Rewrite to assert the new
  contract (runs with provider, fails closed without). Keep `_override` copies both
  `news_signal_fn` AND `fundamentals_signal_fn` (a test for each).
- **CLI:** `backtest sweep` and `backtest walk-forward` with `--news-snapshot` /
  `--fundamentals-snapshot` exit 0 + produce results for the bundled `news_coverage_tilt` /
  `fundamentals_earnings_tilt`; misuse (snapshot on a non-matching strategy) errors; in `promote`,
  `needs_X` without the matching snapshot errors **before** a holdout reservation is created.
- **Gate-audit provenance:** after a passing PIT promote, the `gate_evaluations` row records the
  `news_snapshot` / `fundamentals_snapshot`.
- **End-to-end funnel unblock (BOTH lanes):** a `needs_news` strategy AND a `needs_fundamentals`
  strategy — `backtest sweep` records breadth, then `research promote --news-snapshot ...` /
  `--fundamentals-snapshot ...` reaches the gate and (on a passing fixture) mints the token /
  transitions to `candidate`. This is the headline behavior the slice delivers; both lanes are
  tested because the promotion blocker removal (§0) covers both.

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.
