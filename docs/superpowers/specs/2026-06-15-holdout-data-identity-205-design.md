# Holdout data identity (#205): physical-bars single-use, provenance-agnostic

**Issue:** #205 — *Holdout data identity: snapshot-backed vs provider-backed burns don't collide
(same physical bars consumable twice).*
**Surfaced by:** GATE-1 of #192 (Codex HIGH), deferred as out-of-scope.
**Builds on:** #161 (atomic reservation), #192 (exact-OOS-interval match), #137 (multiple-testing
defense / shortlist gate).

## Problem

The holdout single-use guard's data-identity rule in `reserve_holdout` (`algua/registry/store.py`)
buckets by provenance:

```python
if snapshot_id is not None:
    data_match = "snapshot_id = ?"            # snapshot-backed rows
else:
    data_match = "snapshot_id IS NULL AND data_source = ?"   # provider-backed rows
```

A snapshot-backed burn and a non-snapshot/provider-backed probe are **distinct identities** — they
never collide. So the *same physical out-of-sample bars*, reached once via an ingested snapshot S
and again via a provider P (or via two different snapshots of the same data, with different
`snapshot_id`s), can be evaluated **twice** without tripping the guard or requiring the human
`--allow-holdout-reuse` override. That is a partial bypass of the multiple-testing defense (#137).

This is the **provenance axis**, orthogonal to the **framing axis** #192 already closed (#192 made
the match key the exact OOS *interval* `[holdout_start, holdout_end]` rather than
`(period, holdout_frac)`, so a re-framed `--holdout-frac` landing on overlapping OOS bars is caught
— but only *within one provenance bucket*).

### What is reachable today

The promote CLI's `select_provider` exposes only two providers:
`SyntheticProvider` (`--demo`, `snapshot_id=None`) and `StoreBackedProvider` (`--snapshot`,
`snapshot_id` set). A live/mutable provider (e.g. yfinance) is **not** reachable from
`research promote` today. So the realistic, currently-reachable collisions are:

- **two snapshots of identical bars** (different `snapshot_id` ⇒ different bucket), and
- **synthetic vs snapshot** of the same bars.

The fix is nonetheless written provenance-agnostically (a future live-provider path must not
reopen the hole).

## Design

Two reinforcing, orthogonal components, plus a hardened match rule.

### Component 1 — content-hash the exact OOS bars (the core fix)

`holdout_window` (`algua/backtest/engine.py`) already loads the bars and slices the exact OOS grid
(`_adj_grid(bars).iloc[train_n:]`) at reserve time, *without running the strategy* — it is the
single source of truth for the bars `walk_forward` will burn. It will additionally compute and
return two deterministic content hashes:

- `holdout_data_hash` — a hash of the **OOS-window** grid slice (the exact bars burned).
- `period_data_hash` — a hash of the **full requested-period** grid (frame-independent lineage).

Hashing reuses the serialization discipline of the existing `logical_bars_hash`
(`algua/data/files.py`): canonical column order (sorted symbols, length-prefixed UTF-8), index as
int64 ns-since-epoch UTC, values as IEEE-754 float64 little-endian, signed-zero canonicalized. The
grid is wide (timestamp index × symbol columns of `adj_close`) rather than the long canonical bars
frame `logical_bars_hash` consumes, so a small dedicated serializer (`_grid_content_hash`) lives
next to `_adj_grid`/`holdout_window` in `engine.py`. Identical bars ⇒ identical digest regardless of
how they were reached.

`holdout_window` return type changes from `tuple[str, str]` to
`tuple[str, str, str | None, str | None]` (`holdout_start, holdout_end, holdout_data_hash,
period_data_hash`). The hashes are `None` only in the degenerate empty-bars branch (where
`walk_forward` subsequently raises and the reservation is released, so the value is immaterial);
whenever real OOS bars exist, both hashes are populated.

### Component 2 — agents must use a *reproducible* data source

A strict "snapshot required" would refuse every agent `--demo` promote (the synthetic provider has
no `snapshot_id`) and break the demo-based test/dev path. But `SyntheticProvider(seed=0)` is
deterministic — reproducible, just not snapshot-backed. So the rule is **reproducible source**, not
snapshot-only:

> An **agent** promote requires `snapshot_id is not None OR getattr(provider, "reproducible", False)`.

- `SyntheticProvider` gains a class attribute `reproducible = True`.
- `StoreBackedProvider` qualifies via its non-null `snapshot_id` (snapshots are immutable,
  content-addressed).
- A future mutable/live provider (no `snapshot_id`, no `reproducible` marker) is **refused for
  agents** — fail-closed — so an agent can never burn a holdout over bars that may silently revise
  between runs. Humans are exempt (they accept the cost, mirroring `--allow-non-pit`); no new flag.

Enforced in `promotion_preflight` (`algua/registry/promotion.py`), pre-peek, alongside the existing
PIT / fundamentals / breadth refusals (it already receives `provider`). Duck-typed `getattr` marker
avoids a `registry → data` import-boundary violation.

### The hardened match rule

`reserve_holdout` gains a `data_hash: str | None` and a `period_hash: str | None` parameter and
matches a prior row for the same `strategy_id` if **any** branch holds:

| # | Branch | Closes |
|---|--------|--------|
| 1 | `holdout_data_hash IS NULL` → match unconditionally | legacy/pre-column rows, fail-closed |
| 2 | `holdout_data_hash = :data_hash` | **#205** — identical OOS bars, any provenance/period |
| 3 | *(same provenance bucket)* **AND** OOS-interval overlap **OR NULL-interval** | **#192** — frac/period reframe within one provenance |
| 4 | `period_data_hash = :period_hash` **AND** OOS-interval overlap **OR NULL-interval** | same-period **cross-provenance** partial-overlap (e.g. two snapshots of identical bars at different `--holdout-frac`) |

- Provenance bucket (branch 3) is the existing `snapshot_id`/`data_source` logic, retained
  verbatim so #192's blast radius is unchanged.
- OOS-interval overlap is the existing `(holdout_start IS NULL OR holdout_end IS NULL OR
  (holdout_start <= ? AND ? <= holdout_end))` test (NULL interval ⇒ unconditional, fail-closed),
  reused by branches 3 and 4.
- A row matches whether it is a **pending reservation or a committed burn** (no `committed_at`
  filter) — a pending row blocks too, preserving #161's TOCTOU guarantee.
- The whole SELECT + INSERT stays inside the existing `BEGIN IMMEDIATE … commit / rollback`
  critical section.

When the incoming probe's `data_hash`/`period_hash` is `None` (degenerate empty-bars case), branch
2/4 simply don't fire; the run fails closed via `walk_forward` raising and the reservation is
released.

### Schema

`holdout_evaluations` gains two nullable columns: `holdout_data_hash TEXT`, `period_data_hash
TEXT`. `SCHEMA_VERSION` 23 → 24. Migration adds both via the existing `_add_missing_columns`
mechanism. Legacy rows (and #192's conservative full-period backfilled rows) keep them `NULL` ⇒
branch 1 fail-closed (cannot recompute historical bars at migration time). `data_source` /
`snapshot_id` columns are retained (branch 3 + audit/evidence).

## Data flow (research promote)

1. `research_cmd.promote` resolves provider; `data_source`, `snapshot_id` as today.
2. `promotion_preflight` — **new**: reproducible-source refusal for agents (pre-peek).
3. `holdout_window(...)` → `(holdout_start, holdout_end, holdout_data_hash, period_data_hash)`.
4. `reserve_holdout(..., holdout_start, holdout_end, data_hash=holdout_data_hash,
   period_hash=period_data_hash, allow_reuse=...)` — atomic reserve under the new four-branch match;
   raises (fail-closed) on collision without `--allow-holdout-reuse`.
5. `walk_forward(..., on_peek=finalize)` burns on peek (unchanged, #193).
6. `run_gate(...)` records the gate row (unchanged).

## Out of scope / deferred (follow-up issue)

- **Cross-provenance + period-reframe diagonal:** a *different period* reached via *different
  provenance* whose OOS windows overlap but whose bars are not identical is **not** caught by
  branches 1–4 (different `period_data_hash`, different bucket, non-identical OOS hash). Extremely
  narrow; closing it would require provenance-independent OOS-interval matching, which over-blocks
  "same dates, genuinely different data" and changes #192's blast radius. Deferred — mirrors how
  #192 was scoped to one axis and deferred #205 (this issue) as the other.
- Live/mutable-provider promote *path* (the reproducible-source guard fail-closes it; wiring such a
  provider into `select_provider` is separate work).
- Per-bar revision tolerance for live providers (a provider that revises a single tick yields a
  different `holdout_data_hash`; matched only if dates overlap within the same provenance bucket).

## Testing

- **Issue's stated test:** a burn over snapshot S (whose OOS bars equal another reproducible source
  P's over the window) blocks a subsequent overlapping-interval probe via P **without**
  `--allow-holdout-reuse` (branch 2).
- **Two snapshots, identical bars, same frac:** second burn blocked (branch 2).
- **Two snapshots, identical bars, different `--holdout-frac` (same period):** blocked (branch 4).
- **#192 regression:** same provenance, re-framed frac/period landing on overlapping OOS bars still
  blocked (branch 3).
- **Legacy NULL-hash row:** fail-closed (branch 1).
- **Genuinely different data, non-overlapping OOS / different period:** NOT blocked (no false
  positive).
- **Component 2:** agent + non-reproducible provider (no `snapshot_id`, no `reproducible`) refused
  pre-peek; agent + `--demo` (synthetic) and agent + `--snapshot` allowed; human + non-reproducible
  allowed.
- **Concurrency:** existing `test_concurrent_reserve_holdout_single_burn` harness still passes
  (single burn under contention) with the new signature.
- Determinism of `_grid_content_hash` (same grid ⇒ same digest; column-order independence).

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
