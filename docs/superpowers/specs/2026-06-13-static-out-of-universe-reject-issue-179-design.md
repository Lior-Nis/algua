# Static backtest: reject out-of-universe target weights (parity with PIT mode) — Issue #179

## Problem

A target weight returned for a symbol **outside the operating universe** is handled three
different ways across the decision paths:

- **Static backtest** (`algua/backtest/engine.py::_decision_weights` and `_decision_weights_fast`):
  `w.reindex(columns).fillna(0.0)` silently **drops** the out-of-universe weight.
- **PIT backtest** (`_decision_weights` with `universe_by_date`): a bespoke inline block
  (engine.py:150-156) **rejects** any nonzero weight for a non-member → `BacktestError`.
- **Paper/live** (`algua/live/paper_loop.py::decide` → `build_intents`): `build_intents` builds
  intents from `set(weights.index) | set(current_weights)`, so the out-of-universe weight would be
  **traded**.

Result: a buggy/injected `signal`/construction policy that emits an out-of-universe symbol behaves
differently in static backtest (dropped) vs PIT (rejected) vs live (traded) — a backtest↔live
parity gap and a capital-risk asymmetry. Well-formed strategies only weight their own universe, so
this bites a buggy strategy; it is a real latent gap nonetheless.

Surfaced by the GATE-2 review of #141 (PR #174). **Pre-existing** (predates #141).

## Goal

Make every decision path enforce universe membership identically, through the **one** shared
validation rail, so research and live can never drift. A nonzero target weight for a symbol outside
the path's operating universe must hard-fail (backtest) / be rejected (live) — never silently
dropped or traded.

## Design

### New check — `check_universe_membership`

In `algua/risk/limits.py`:

```python
def check_universe_membership(
    weights: pd.Series, allowed_symbols: Collection[str], strategy_name: str
) -> None:
    """Reject any NONZERO target weight for a symbol outside the operating universe. Mirrors the
    PIT loop's `w != 0.0` 'nonzero' semantics exactly (a strict superset of the old inline PIT
    check). Raises RiskBreach('out_of_universe', ...) naming the offenders and the allowed set."""
```

- "Nonzero" is exact `weights != 0.0`, matching the PIT block this subsumes (NOT `abs > WEIGHT_TOL`).
  A sub-tolerance weight is below `build_intents`' trade threshold anyway; mirroring `!= 0.0` keeps
  the check a strict superset of prior PIT behavior and avoids a semantic change. A code comment
  states this strictness rationale (any nonzero weight for a non-member is a strategy bug; the
  tolerance can be raised to `WEIGHT_TOL` later without changing the architecture if numeric noise
  ever becomes a practical problem).
- `allowed_symbols` is converted to a `set` for O(1) membership.
- The breach message names `strategy_name` (consistent with the sibling `check_*` functions) and
  renders the offending symbols with `sorted(offenders, key=str)` — a mixed/non-string symbol index
  must not raise a bare `TypeError` that escapes the `RiskBreach → BacktestError` / live-kill-switch
  contract.
- Raises `RiskBreach("out_of_universe", ...)`. The new kind joins the existing breach kinds; in
  live it trips/flattens like any breach, in backtest it is wrapped to `BacktestError` (below).
- Empty `weights` → no-op (consistent with the other checks). Empty `allowed_symbols` with any
  nonzero weight → ALL nonzero weights breach (there is no allowed universe); a caller that means
  "flat" must SKIP the call rather than pass an empty allowed set — exactly what the PIT loop does
  today via `if not members: continue` before it ever reaches the rail.

### Fold into the one rail — `validate_decision_weights`

`validate_decision_weights` gains a **required** `allowed_symbols: Collection[str]` parameter (no
`None` default → no optional/dual path). New check order:

```
finite (fail-closed) → universe → short policy → per-symbol cap → gross exposure
```

Membership goes immediately after the fail-closed finite check (structural: are these even valid
symbols for this path?) and before the value-based policy checks. Because this is the ONE rail every
path calls, membership can no longer drift between research and live.

### Callers pass their operating universe

The `validate_decision_weights` call sites (current `engine.py` after #178 — `_fast_weights` is the
fast-path validator, `_decision_weights_fast` just adds the bounded parity guard around it):

| Validate call site | `allowed_symbols` |
|---|---|
| `paper_loop.decide()` | `strategy.universe` (declared static universe) |
| `_decision_weights` — static (`universe_by_date is None`) | `set(strategy.universe) & set(adj.columns)` |
| `_decision_weights` — PIT | `members` (as-of members at `t`) |
| `_fast_weights` (fast path) | `set(strategy.universe) & set(adj.columns)` |
| `_canonical_row` (bounded parity proxy — see below) | `set(strategy.universe) & set(adj.columns)` |

**Static operating universe = `strategy.universe ∩ adj.columns` (declared AND available)** — not raw
`adj.columns`. `simulate` computes it ONCE in the static branch (right after the existing
`if bars.empty` guard) and **fails closed with `BacktestError` if it is empty while the declared
universe is non-empty** — i.e. the provider returned bars but NONE for a declared symbol. Without
this guard the empty-allowed set would no-op validation and run a silently-meaningless flat backtest
over an all-undeclared price panel (an edge the intersection itself introduces). The static decision
paths (`_decision_weights` static branch, `_fast_weights`, `_canonical_row`) recompute the same
`set(strategy.universe) & set(columns)` locally (each already holds `strategy` + `columns`); the
intersection is trivial and self-contained per path.

This single intersection closes BOTH edges and makes the invariant `allowed ⊆ strategy.universe`
HOLD rather than be assumed:
- A symbol DECLARED but with no fetched price data (in `strategy.universe`, absent from
  `adj.columns`) → not in the intersection → fails-closed in backtest (it cannot be simulated
  without a price). Acceptable, stricter-in-backtest divergence vs live (which uses
  `strategy.universe` and would attempt the order, then the broker handles no-fill).
- A symbol a misbehaving provider returned UNDECLARED (in `adj.columns`, absent from
  `strategy.universe`) → not in the intersection → fails-closed, matching live's
  `strategy.universe` rejection. For well-formed providers (`adj.columns ⊆ strategy.universe`) the
  intersection equals `adj.columns`, so there is no behavior change.

The truly-out-of-universe bug (symbol not in `strategy.universe` at all) fails under every path.

`verify_signal_panel_parity` (the exhaustive promotion gate) calls `_fast_weights` AND
`_decision_weights` directly and compares every bar; both now enforce membership, so the promotion
gate inherits the rejection for free — no separate change there.

### Delete the bespoke PIT block

The inline non-member rejection in `_decision_weights` (the `non_members` block, currently
engine.py ~150-156) is **removed** — now subsumed by passing `allowed_symbols=members` to the rail.
This is the unification: one check, no second code path to drift.

### Keep `_canonical_row` a faithful loop-proxy

`_canonical_row` (engine.py ~170-181) recomputes the per-bar canonical `construct(signal(view),
view)` used by the BOUNDED runtime parity guard (`_assert_parity`) — it is the loop's stand-in.
Today it reindex-drops without validating. Once `_decision_weights` (the loop) rejects out-of-universe
weights, the proxy must reject them too, or the bounded guard silently diverges from the loop for a
strategy whose per-bar `signal` emits an out-of-universe weight while its `signal_panel` does not
(the fast path's `construct` only ever sees in-`columns` scores, so it cannot surface this itself).
Fix: call the FULL `validate_decision_weights(w, strategy.execution, strategy.name,
allowed_symbols=set(strategy.universe) & set(columns))` in `_canonical_row` BEFORE the reindex,
wrapping `RiskBreach → BacktestError` like the loop. Using the full rail (not just the membership
check) makes `_canonical_row` a TRULY faithful loop-proxy with identical check ordering
(finite → universe → short → cap → gross) — a partial subset would, for a row that is both
non-finite and out-of-universe, report `out_of_universe` while the loop reports the finite breach
first. This is also safe and scope-respecting: well-formed strategies pass every rail (no behavior
change), and a strategy whose CONSTRUCT output breaks a value rail already raises in `_fast_weights`
(engine.py ~222) BEFORE `_assert_parity`/`_canonical_row` is ever reached — so the only new rejection
this surfaces is the per-bar-`signal` out-of-universe case #179 targets. The exhaustive promotion
gate already covers that case (it runs the loop); this extends coverage to the ordinary-backtest
bounded guard for defense-in-depth.

### Error flow / parity

- Backtest loop & fast path already wrap `RiskBreach → BacktestError(f"{breach.detail} at {t}")`
  (engine.py:161-162, 224-225). The `out_of_universe` breach hard-fails the backtest exactly like
  the other rails and like the old PIT behavior. The existing PIT test asserts only `match="BBB"`,
  which the new message still satisfies.
- Live/paper: the `RiskBreach` propagates through `decide()` exactly as short/cap/gross breaches do
  today → the weight is rejected, never traded.
- `build_intents` is **unchanged** — the rejected (nonzero out-of-universe) weight never reaches it.
  "Never traded" is scoped to NONZERO out-of-universe targets: a ZERO target for an out-of-universe
  symbol passes the check by design, so `build_intents` may still emit a flattening SELL to exit an
  EXISTING out-of-universe holding (a position the universe dropped) — which is the desired
  behavior, not a parity gap.

## Components touched

- `algua/risk/limits.py` — new `check_universe_membership`; `validate_decision_weights` gains the
  required `allowed_symbols` param and calls the new check (order: finite → universe → short → cap →
  gross).
- `algua/backtest/engine.py` *(CODEOWNERS — human design call; this spec + GATE-1 serve it)* —
  `_decision_weights` (static + PIT) and `_fast_weights` validate calls gain `allowed_symbols`; the
  inline PIT `non_members` block deleted; `_canonical_row` gains a full `validate_decision_weights`
  call (faithful loop-proxy); `simulate` fails closed when the static operating universe is empty.
  `verify_signal_panel_parity` is untouched — it inherits via `_decision_weights` + `_fast_weights`.
- `algua/live/paper_loop.py` — `decide()` passes `allowed_symbols=strategy.universe`.
- `tests/test_risk_limits.py` — 5 existing direct `validate_decision_weights` calls gain an
  `allowed_symbols` arg.

## Testing

New tests (TDD — fail first):

1. **`check_universe_membership`** unit: nonzero out-of-universe weight → `RiskBreach`
   (kind `out_of_universe`, message names offender + `strategy_name`); in-universe weights pass; a
   zero weight for an out-of-universe symbol passes (mirrors `!= 0.0`); empty series passes; empty
   `allowed_symbols` + a nonzero weight → breach; a non-string symbol label in the offenders renders
   via `key=str` without a bare `TypeError`.
2. **`validate_decision_weights`** unit: out-of-universe weight raises with the new param; an
   in-universe set passes; ordering — a weight that is both out-of-universe and non-finite surfaces
   the finite breach first (finite precedes universe).
3. **Static backtest** integration: a static-mode strategy returning a weight for a symbol outside
   `strategy.universe ∩ adj.columns` raises `BacktestError` naming the symbol (the issue's core
   acceptance) — for BOTH `_decision_weights` (loop) and the fast path (`_decision_weights_fast` /
   `_fast_weights`, which has no prior out-of-universe coverage).
4. **`_canonical_row` / bounded parity guard**: a strategy whose per-bar `signal` emits an
   out-of-universe weight (clean `signal_panel`) fails the fast-path run closed via the
   `_canonical_row` membership check (proves the bounded guard stays a faithful loop-proxy).
5. **Live/paper** integration: `decide()` with an out-of-universe weight raises (rejected, not
   traded) against `strategy.universe`.
6. **Empty static operating universe**: a (synthetic) provider that returns bars only for
   undeclared symbols → `simulate` raises `BacktestError` (declared universe non-empty, intersection
   empty) rather than running a flat backtest.
7. **No regression**: existing in-universe strategies/tests unaffected; the existing PIT non-member
   test (`test_decision_weights_rejects_non_member_weight`, asserts `match="BBB"`) still passes via
   the unified rail; existing fast-path breach tests (gross/long-only/cap/inf) still raise in
   `_fast_weights` as before.

## Acceptance

- A static-mode strategy returning a weight for an out-of-universe symbol hard-fails the backtest
  the same way PIT mode does, and live rejects it identically.
- Existing strategies/tests that only weight in-universe symbols are unaffected.
- `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.

## Out of scope / non-goals

- No change to `build_intents` or live breach handling beyond what the shared rail already does.
- No vectorized PIT fast path (unrelated; deferred elsewhere).
- The declared-but-no-data edge case is fail-closed in backtest by design; **live keeps using
  `strategy.universe`** (it does NOT intersect with the latest bar's available symbols). Rejecting a
  declared symbol on a transient missing bar would change live semantics well beyond #179 and is too
  harsh; the backtest-stricter divergence is intentional and documented above.
- No fetch-time "returned symbols ⊆ requested" guard — that is a data-layer concern; the
  `strategy.universe ∩ adj.columns` intersection already neutralizes its parity impact at the rail.

Ref: PR #174 GATE-2 comment.
