# Design — holdout single-use guard matches on the actual OOS bar-interval

**Issue:** #192 — *Holdout matching keys on holdout_frac: a different --holdout-frac re-burns
overlapping OOS bars.*
**Date:** 2026-06-13
**Surfaced by:** GATE-2 multi-model review of PR #191 (#161 atomic holdout reservation), Codex HIGH.
Pre-existing — the predicate was ported verbatim from the removed
`overlapping_holdout_evaluations`; #161's scope was the TOCTOU race only.

## Problem

`reserve_holdout` (`algua/registry/store.py`) decides whether a walk-forward holdout has already
been consumed by matching on:

```
WHERE strategy_id=? AND holdout_frac=? AND <data identity> AND period_start<=? AND ?<=period_end
```

i.e. **exact `holdout_frac`** AND **full-period overlap**. But the quantity that must be single-use
is not the full period — it is the **out-of-sample tail**, the last `int(n·holdout_frac)` bars that
`walk_forward` carves and gates on. Keying on `holdout_frac` and the full period lets the *same OOS
bars* be re-evaluated two ways:

1. **Vary `--holdout-frac`** on the same period: a different frac → different key → the overlapping
   tail is re-peeked without tripping the single-use guard.
2. **Re-frame the overall period** so the final holdout sub-interval overlaps a previously-burned
   one but the `(period, holdout_frac)` tuple differs.

Both are a partial bypass of the multiple-testing defense (#137). `promote` exposes an arbitrary
`--holdout-frac` (default 0.2) to an agent, so this is reachable autonomously.

## Approach (chosen)

Match the single-use guard on the **actual OOS bar-interval** `[holdout_start, holdout_end]` — the
exact bars `walk_forward` would carve as the holdout — instead of on `(full period, holdout_frac)`.
Two runs whose holdout bars overlap then collide regardless of how the period/frac is framed,
closing both vectors above.

The exact interval is computed at reserve time from the **bar date-index** (no strategy execution):
`build_portfolio` fetches `bars = provider.get_bars(_fetch_symbols(…), start, end, "1d")`, pivots to
`adj` (index = unique sorted bar-timestamps), and `pf.returns()` carries that index. So
`n = len(adj.index)` and the burned tail is `adj.index[train_n:]` with `train_n = n − int(n·frac)`.
Re-deriving `adj.index` is a pure data load + pivot — the same grid `walk_forward` uses — giving the
**identical `n`** and therefore the **exact** tail dates. This is done in the **CLI seam**
(`research_cmd`, which already builds the provider): compute the window there, pass
`holdout_start/holdout_end` into `reserve_holdout` as date strings, so the registry layer stays pure
(no data import, no engine-internals duplication).

### Why not the alternatives (GATE-1, two rounds)

- **Drop `holdout_frac`, match full-period overlap (coarsest, sound).** Provably fail-closed (the
  true tail is always ⊆ the full period) but over-blocks the routine "re-test the same strategy on
  an overlapping *newer* period" move — refused with a human-only `--allow-holdout-reuse`. Rejected:
  needless false-block of legitimate research.
- **Approximate the interval at reserve (calendar-day or trading-session), no bar load.** Cheap, but
  Codex proved it is **fail-open**: any reserve-time interval that does not see the actual bars can
  diverge from `walk_forward`'s real bar-tail in *either* direction. Concrete counterexample —
  provider data ends before `period_end` (sessions 1–100, bars only 1–80, frac 0.2): a session/
  calendar estimate stores `[81,100]` while the true tail is `[65,80]`, so a prior burn on `[1,80]`
  (true tail `[65,80]`) does not collide → the guard the feature exists to enforce silently passes.
  Rejected: an approximation cannot be sound for a guard whose whole job is to stop an agent gaming
  OOS reuse.

The chosen actual-bar-interval is the only option that is both **fail-closed** (never misses a true
overlap) and **precise** (never false-blocks a genuinely-fresh tail).

## The interval function

A new public helper in `algua/backtest/engine.py` (it needs `_fetch_symbols` and the same
`bars → adj` pivot as `build_portfolio`; both share one extracted `_grid_index(bars)` so they can
never drift):

```python
def holdout_window(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    holdout_frac: float,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
) -> tuple[str, str]:
    """The exact OOS holdout interval [start, end] (ISO dates) that `walk_forward` would carve as
    the last `holdout_frac` of the simulation grid — computed from the bar date-index WITHOUT
    running the strategy. Reproduces `build_portfolio`'s grid (identical `n`), so the boundary is
    bit-identical to walk_forward's `holdout_metrics`. Degenerate inputs (no bars, holdout rounds to
    <1 bar) return the conservative full grid/period: the subsequent `walk_forward` raises and the
    reservation is released, so the value is immaterial but stays fail-closed."""
    bars = provider.get_bars(_fetch_symbols(strategy, universe_by_date), start, end, "1d")
    if bars.empty:
        return start.date().isoformat(), end.date().isoformat()   # walk_forward will raise+release
    idx = _grid_index(bars)                                        # == build_portfolio's adj.index
    n = len(idx)
    holdout_n = int(n * holdout_frac)                              # floor — mirrors _segment_bounds
    if holdout_n < 1:
        return idx[0].date().isoformat(), idx[-1].date().isoformat()  # degenerate; raises+releases
    train_n = n - holdout_n                                        # 1 ≤ train_n ≤ n-1 for frac∈(0,1)
    return idx[train_n].date().isoformat(), idx[-1].date().isoformat()
```

- **Floor (`int`)**, the same `int(n·holdout_frac)` `_segment_bounds` uses.
- `holdout_end` is the **last actual bar date** (`idx[-1]`), which may be earlier than `period_end`
  when data ends early — exactly the case the approximations got wrong.
- Lives in `backtest` (uses the injected `DataProvider` contract — no `algua.data` import, within
  the import-linter rules). The CLI passes its output to `reserve_holdout`.

### Exactness guarantee + the one residual

A **cross-check test** pins the guarantee: for a real `(strategy, provider, period, frac)`,
`holdout_window(...)` must equal `(wf.holdout_metrics["start"], wf.holdout_metrics["end"])`. That
test is what prevents drift from `walk_forward`.

The only residual is a **double bar-load** (once in `holdout_window` at reserve, once inside
`walk_forward`): if the provider were non-deterministic between the two loads, the reserved interval
could skew from the burned one. Agent promotes **require** a PIT `--universe` (snapshot-backed,
immutable — CLAUDE.md), so the gameable path is always deterministic and the two loads are identical.
The human/live-provider path is trusted and can `--allow-holdout-reuse` anyway, so a late-arriving-bar
skew there is immaterial. Threading the already-loaded bars into `walk_forward` to remove the second
load is a deferred optimization (non-goal), not a correctness requirement.

## Schema change (22 → 23)

Add two columns to `holdout_evaluations`:

```
holdout_start TEXT   -- ISO date; OOS tail start (matched)
holdout_end   TEXT   -- ISO date; OOS tail end   (matched)
```

Mirroring the `committed_at` (v22) pattern, the columns are added in **three** places:

1. **`CREATE TABLE` in `_SCHEMA`** — fresh DBs get them at bootstrap.
2. **`_add_missing_columns(conn, "holdout_evaluations", {...})`** in `migrate()` — populated DBs get
   them via idempotent, cross-process-safe `ALTER TABLE`.
3. **Backfill** (new `_backfill_holdout_intervals(conn)`, run in `migrate()` right after the ALTER):
   for every row with `holdout_start IS NULL OR holdout_end IS NULL`, set the **conservative full
   period** `holdout_start = period_start, holdout_end = period_end` in one `UPDATE`. The exact tail
   cannot be recomputed at migration time (no provider in `migrate()`), and the full period is a
   guaranteed superset of whatever the legacy true tail was → **fail-closed** (may over-block new
   runs overlapping the legacy period, the acceptable direction for already-burned data). After the
   loop, assert zero `holdout_start IS NULL OR holdout_end IS NULL` rows remain before stamping the
   schema (a partial-NULL row would otherwise fail the matcher open — see below).

`holdout_frac` stays as an **evidence-only** column (like `config_hash`) — recorded, no longer part
of identity. `period_start/period_end` likewise remain as evidence.

### Why this backfill is safe (unlike v22 `committed_at`)

v22 skipped a backfill because writing `committed_at` could race a genuine concurrent reservation.
This one writes only the *new* `holdout_start/end` columns, deterministically from *immutable*
columns, scoped to `WHERE holdout_start IS NULL OR holdout_end IS NULL`. A concurrent reserve under
the new code inserts a row with the interval already set, so the backfill's `IS NULL` filter never
touches it. Result: no NULL-interval dual-path — after `migrate()` every row has an interval.

**Invariant + fail-closed belt (GATE-1 Codex MEDIUM ×2):** `migrate()` runs on every `connect()`
before any `reserve_holdout`, and every new insert sets the interval, so no NULL-interval row
*should* exist. But a NULL would make `NULL <= ?` evaluate NULL/false → the matcher would fail
**open**. Rather than rely on backfill completeness alone, the matcher treats a NULL-interval row as
an unconditional **match** (fail-closed) — see below. Belt-and-suspenders: the backfill removes
legacy NULLs, and the matcher fails closed if one ever slips through (e.g. an old-code writer
inserting after the columns are added).

## Matching change (`reserve_holdout`)

`reserve_holdout` gains two parameters, `holdout_start` / `holdout_end` (the exact interval the CLI
computed via `holdout_window`). The data-identity branch (`snapshot_id` vs `data_source`) is
unchanged. The SELECT and INSERT change:

- **SELECT** drops `holdout_frac=?` and full-period overlap, swaps in interval overlap, and treats a
  NULL-interval row as an unconditional match (fail-closed):
  ```sql
  SELECT 1 FROM holdout_evaluations
   WHERE strategy_id=? AND <data identity>
     AND (holdout_start IS NULL OR holdout_end IS NULL
          OR (holdout_start <= ?holdout_end AND ?holdout_start <= holdout_end))
   LIMIT 1
  ```
  (still matches BOTH pending reservations and committed burns — no `committed_at` filter — so a
  pending row blocks fail-closed, exactly as today.)
- **INSERT** writes `holdout_start`/`holdout_end` alongside the existing columns.

`finalize_holdout_reservation` and `release_holdout_reservation` are unchanged (the interval is set
at insert; finalize only flips `committed_at`/`config_hash`).

## CLI change (`research_cmd.py`)

In `promote`, after the provider/universe are built and **before** `reserve_holdout`, compute the
window and pass it in:

```python
holdout_start, holdout_end = holdout_window(
    strategy, provider, start_dt, end_dt,
    holdout_frac=holdout_frac, universe_by_date=universe_by_date)
reservation_id, reused = repo.reserve_holdout(
    repo.get(name).id, data_source=data_source, snapshot_id=snapshot_id,
    period_start=period_start, period_end=period_end, holdout_frac=holdout_frac,
    holdout_start=holdout_start, holdout_end=holdout_end,
    allow_reuse=allow_holdout_reuse)
```

The reserve → `walk_forward` → finalize/release flow is otherwise unchanged.

## Components touched

| File | Change |
|---|---|
| `algua/backtest/engine.py` | **New** `holdout_window()` + extracted `_grid_index(bars)` shared with `build_portfolio` (prevents drift). |
| `algua/registry/store.py` | `reserve_holdout` gains `holdout_start`/`holdout_end` params; matches interval-overlap (drops `holdout_frac` + full-period from WHERE; NULL-interval ⇒ fail-closed match); inserts the interval columns. |
| `algua/registry/repository.py` | Update the `reserve_holdout` Protocol signature (adds the two params). |
| `algua/registry/db.py` | `SCHEMA_VERSION` 22→23; add `holdout_start/holdout_end` to `_SCHEMA` CREATE TABLE; add to `_add_missing_columns`; new `_backfill_holdout_intervals` (conservative full-period; set both cols; assert zero NULLs); update the table-doc comment to describe interval matching. |
| `algua/cli/research_cmd.py` | Compute `holdout_window(...)` before reserve; pass the interval to `reserve_holdout`. |

**Base branch:** work off `origin/main` (it has #161's `reserve_holdout`/`finalize`/`release` flow
and the Protocol that exposes only `reserve_holdout` — verified). Local `main` is behind and still
has the pre-#161 `overlapping_holdout_evaluations` shape; do **not** base on it.

## Testing

**Cross-check (the linchpin)** (`tests/`)
- For a real `(strategy, fake-provider, period, frac)`, `holdout_window(...)` equals
  `(wf.holdout_metrics["start"], wf.holdout_metrics["end"])`. Proves the reserved interval is exactly
  what `walk_forward` burns. Run for a couple of `(period, frac)` pairs.

**Unit — `holdout_window()`**
- Known bar-date index → expected tail dates; floor boundary; `holdout_end == last bar date` (incl.
  the data-ends-early case where it is < `period_end`); degenerate `holdout_n < 1` and `bars.empty`
  → conservative full grid/period (no `IndexError`).

**Reserve matching**
- **Regression (the exploit):** same period, burn at `frac=0.2`, then reserve at `frac=0.4` whose
  tail overlaps → now **blocks** (previously passed). With `allow_reuse=True` → succeeds, `reused=1`.
- **Re-framed period, overlapping tail:** different `(period, frac)` whose actual holdout intervals
  overlap → blocks.
- **Fresh non-overlapping tail:** burn an earlier overlapping *period* whose tail does NOT overlap
  the new run's tail → **allowed** (proves we did not regress to coarse full-period matching).
- **Data identity:** a snapshot-backed burn does not block a non-snapshot probe (and vice-versa) —
  unchanged, re-asserted against the new predicate.

**Migration / backfill**
- A DB seeded with a legacy `holdout_evaluations` row lacking the interval → after `migrate()`, the
  columns are the conservative `[period_start, period_end]`, and that legacy burn now blocks an
  overlapping reserve.
- Idempotent: second `migrate()` no-ops; backfill `WHERE … IS NULL` touches nothing; the zero-NULL
  assertion holds.
- **Fail-closed on a NULL-interval row:** insert a NULL-interval row directly (simulating an
  old-code writer) and assert `reserve_holdout` blocks against it.

**Concurrency** — the BEGIN IMMEDIATE atomic section is unchanged; confirm the existing #164/#161
harness still passes with the new predicate (two concurrent same-period reserves still serialize to
one winner).

## Non-goals

- Threading the bars loaded in `holdout_window` into `walk_forward` to avoid the second load — a
  deferred optimization; correctness already holds for the deterministic PIT path.
- Recording the actual interval at finalize from `wf.holdout_metrics` — the reserve-time value is
  already exact for the PIT path, so there is nothing to reconcile.
- Touching the concurrency mechanics (BEGIN IMMEDIATE, pending-row semantics, finalize/release).
- Changing relaxation policy: `--allow-holdout-reuse` stays the human-only audited override.

## Deferred follow-up (GATE-1 Codex HIGH, out of scope)

The data-identity rule (snapshot-backed burns do not block non-snapshot probes, and vice-versa) is
pre-existing — ported from #161 and unchanged here. Codex notes the *same physical bars* reachable
once via an ingested snapshot and again via a provider class could be consumed twice without
`--allow-holdout-reuse`. That is a separate identity-definition question orthogonal to #192's
frac/period framing bug. **File a follow-up issue** to define a stricter physical data identity (or
require snapshot-backed promotion for agent promotes); do not expand #192's scope to cover it.

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
