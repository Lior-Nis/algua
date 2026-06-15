# Holdout data identity (#205): provenance-independent single-use OOS window

**Issue:** #205 — *Holdout data identity: snapshot-backed vs provider-backed burns don't collide
(same physical bars consumable twice).*
**Surfaced by:** GATE-1 of #192 (Codex HIGH), deferred as out-of-scope.
**Builds on:** #161 (atomic reservation), #192 (exact-OOS-interval match), #137 (multiple-testing
defense / shortlist gate).
**GATE-1 (this design):** Codex + Gemini Flash + OpenCode/GLM unanimously rejected an earlier
whole-window-content-hash draft (it could not detect *partial-overlap* reuse — the issue's own
acceptance test). This design is the chosen replacement: provenance-independent OOS-interval
matching.

## Problem

The holdout single-use guard's data-identity rule in `reserve_holdout` (`algua/registry/store.py`)
buckets by provenance:

```python
if snapshot_id is not None:
    data_match = "snapshot_id = ?"            # snapshot-backed rows
else:
    data_match = "snapshot_id IS NULL AND data_source = ?"   # provider-backed rows
```

The two buckets never collide. So the *same out-of-sample window* for a strategy, reached once via
an ingested snapshot S and again via a provider P (or via a different snapshot S2 of the same
data), can be evaluated **twice** without tripping the guard or requiring the human
`--allow-holdout-reuse` override — a partial bypass of the multiple-testing defense (#137).

This is the **provenance axis**, orthogonal to the **framing axis** #192 already closed (#192 made
the match key the exact OOS *interval* `[holdout_start, holdout_end]` — but only *within one
provenance bucket*).

### Why not content-hash the bars (rejected at GATE-1)

The intuitive fix — hash the physical OOS bars and match on the hash — was drafted and rejected by
all three reviewers. A whole-window hash is an *exact-match* key: it cannot detect **partial
overlap**. Burn A over `[Jan–Dec]` and probe B over `[Jun–Dec]` (reached via a different snapshot)
re-burn the byte-identical `Jun–Dec` bars, yet B's whole-window hash differs (smaller slice), so the
guard misses it — which is precisely the issue's stated test ("an overlapping-interval probe via
provider P"). Closing partial overlap with hashes would require per-bar content identity (a child
table of `(date, bar_hash)` + overlap join) and correctly solving bar-hash determinism (full OHLCV
vs `adj_close`, NaN canonicalization in the pivoted grid, symbol/column ordering, warmup-induced
period shifts). That is a large, pitfall-rich build for a *weaker* guarantee than the alternative
below: per-bar hashing would still let a strategy re-examine the same calendar era for free whenever
the data differs by a tick.

## Design — the OOS calendar window is the single-use unit, provenance-independent

The natural unit of the multiple-testing defense is **the out-of-sample calendar window for a
strategy**: once you have seen how strategy X performed over a given OOS interval, you have spent
that look — regardless of which data source revealed it. The fix is therefore to **drop provenance
from the match key**.

### Component 1 — provenance-independent OOS-interval match (the #205 fix)

`reserve_holdout` already matches on the exact OOS interval `[holdout_start, holdout_end]` via the
standard overlap test, scoped by `strategy_id`. The change is to **remove the `data_source` /
`snapshot_id` bucket from the SELECT** so the overlap test applies across all provenance:

```sql
SELECT 1 FROM holdout_evaluations
 WHERE strategy_id = ?
   AND (holdout_start IS NULL OR holdout_end IS NULL
        OR (holdout_start <= ? AND ? <= holdout_end))
 LIMIT 1
```

- **Closes #205:** snapshot S, snapshot S2, and provider P over an overlapping OOS interval for the
  same strategy now collide — no `--allow-holdout-reuse`, no provenance escape.
- **Preserves #192:** the exact-interval / NULL-interval (fail-closed) overlap test is unchanged;
  the frac/period-reframe exploit stays closed, now across provenance too.
- **Strictly more conservative than today** (a superset of what currently matches) — it can only
  block *more*, never newly fail-open. Fail-closed is the safe direction.
- `data_source` / `snapshot_id` remain **columns** (persisted as audit/evidence, still recorded by
  `record_gate_evaluation`) but are no longer part of the match.
- Unchanged: the SELECT + INSERT stay inside the single `BEGIN IMMEDIATE … commit/rollback`
  critical section (#161 atomicity); a **pending** reservation blocks exactly like a committed burn
  (no `committed_at` filter); `release`/`finalize` semantics (#193) are untouched.
- **No schema change, no new columns, no hashing.** `SCHEMA_VERSION` stays 23.

**Intended behavior change (documented, not a bug):** re-evaluating a strategy over an *overlapping*
OOS calendar window now requires the human-only `--allow-holdout-reuse` **even when the data source
differs** (e.g. a re-ingested/corrected snapshot, or the same strategy on a different universe over
the same dates). Re-testing one strategy against the same market era *is* multiple testing; gating
it on a human override is the intended discipline, and the override already exists and is audited.

**Consistency with existing identity decisions:** the holdout identity already *excludes* the
universe by design (`research_cmd.py`: "the same OOS data window is burned regardless of universe").
A date-interval key is naturally universe-independent, so this design is consistent with — and
strengthens — that existing stance, rather than reintroducing universe sensitivity (which a
fetched-grid content hash would have).

### Component 2 — agents must use a *reproducible* data source (independent hygiene)

Orthogonal to the match change: ensure an agent's holdout burn is over **reproducible** bars, so the
"OOS truth" it spent is deterministic on a re-run. A strict "snapshot required" would refuse every
agent `--demo` promote (the synthetic provider has no `snapshot_id`) and break the demo/test path,
but `SyntheticProvider(seed=0)` is deterministic — reproducible, just not snapshot-backed. So the
rule is **reproducible source**, not snapshot-only:

> An **agent** promote requires `snapshot_id is not None OR getattr(provider, "reproducible",
> False)`.

- `SyntheticProvider` gains a class attribute `reproducible = True`.
- `StoreBackedProvider` qualifies via its non-null `snapshot_id` (snapshots are immutable,
  content-addressed).
- A future mutable/live provider (no `snapshot_id`, no `reproducible` marker) is **refused for
  agents** — fail-closed. Humans are exempt (they accept the cost, mirroring `--allow-non-pit`); no
  new flag.
- Enforced in `promotion_preflight` (`algua/registry/promotion.py`), **before any provider read**
  (i.e. before `verify_signal_panel_parity`, which fetches bars) so an agent never even reads from a
  forbidden provider. It is a pre-peek refusal alongside the existing PIT / fundamentals / breadth
  checks; `promotion_preflight` already receives `provider`.
- The duck-typed `getattr` marker avoids a `registry → data` import-boundary violation. It is an
  opt-in capability declaration (honor-system); today only `SyntheticProvider` sets it and it always
  carries a fixed integer `seed`. A stronger contract-level capability (a `contracts` Protocol with
  a structural replayability check) is noted as a deferred hardening — not load-bearing while
  `select_provider` exposes only demo/snapshot.

`reserve_holdout` is reached **only** from `research promote` (verified: the other walk-forward
callers — `dormant-sweep`, `sweep`, the forward/paper gates — never reserve/burn the holdout), so
enforcing in `promotion_preflight` covers every burn path.

## Data flow (research promote) — what changes

1. `research_cmd.promote` resolves provider; `data_source`, `snapshot_id` as today.
2. `promotion_preflight` — **new**: reproducible-source refusal for agents, placed *before*
   `verify_signal_panel_parity`.
3. `holdout_window(...)` → `(holdout_start, holdout_end)` — **unchanged** (no hashing).
4. `reserve_holdout(...)` — **changed match** (provenance-independent interval); same signature
   (`data_source`/`snapshot_id` still passed and stored as evidence).
5. `walk_forward(..., on_peek=finalize)` and `run_gate(...)` — unchanged.

## Out of scope / deferred

- **Distinguishing genuinely-different data on the same OOS dates.** This design intentionally
  treats overlapping OOS calendar windows as the single-use unit and does *not* try to tell apart
  "same dates, different data" (it blocks both, human-overridable). A future per-bar physical
  identity could relax this, but it is a *weaker* multiple-testing guard and was rejected here.
- **Stronger reproducibility capability** (a `contracts` Protocol replacing the duck-typed
  `reproducible` marker) — deferred until a mutable/live provider is actually wired into
  `select_provider`.
- **Live/mutable-provider promote path** — the reproducible-source guard fail-closes it; wiring such
  a provider is separate work.

## Testing

- **Issue's stated test:** a burn over snapshot S blocks a subsequent overlapping-interval probe via
  a different reproducible source P (different `data_source`/`snapshot_id`) **without**
  `--allow-holdout-reuse`.
- **Partial overlap, cross-provenance:** burn `[Jan–Dec]` via S blocks a probe `[Jun–Dec]` via P
  (the GATE-1 CRITICAL the rejected draft missed).
- **Two snapshots of the same window:** second burn blocked.
- **#192 regression:** same provenance, re-framed frac/period landing on overlapping OOS bars still
  blocked.
- **Non-overlapping OOS windows:** NOT blocked (no false positive) — different *dates* are always
  allowed, regardless of provenance.
- **`--allow-holdout-reuse` (human):** overrides the cross-provenance block; sets `reused=True`.
- **Legacy NULL-interval row:** fail-closed (matches unconditionally) — pre-existing behavior,
  now provenance-independent.
- **Component 2:** agent + non-reproducible provider (no `snapshot_id`, no `reproducible`) refused
  pre-peek, *before* any provider read; agent + `--demo` (synthetic) and agent + `--snapshot`
  allowed; human + non-reproducible allowed.
- **Concurrency:** `test_concurrent_reserve_holdout_single_burn` still passes (single burn under
  contention) with the provenance-independent match.

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
