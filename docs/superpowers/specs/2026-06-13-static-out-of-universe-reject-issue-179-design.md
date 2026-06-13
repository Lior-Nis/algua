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
  the check a strict superset of prior PIT behavior and avoids a semantic change.
- `allowed_symbols` is converted to a `set` for O(1) membership.
- Raises `RiskBreach("out_of_universe", ...)`. The new kind joins the existing breach kinds; in
  live it trips/flattens like any breach, in backtest it is wrapped to `BacktestError` (below).
- Empty `weights` → no-op (consistent with the other checks).

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

| Caller | `allowed_symbols` |
|---|---|
| `paper_loop.decide()` | `strategy.universe` (declared static universe) |
| `_decision_weights` — static (`universe_by_date is None`) | `adj.columns` (realized fetched/tradeable universe) |
| `_decision_weights` — PIT | `members` (as-of members at `t`) |
| `_decision_weights_fast` | `adj.columns` |

Each path enforces against the universe it can actually act on. The truly-out-of-universe bug
(symbol not in `strategy.universe` at all) fails under every choice, because `adj.columns ⊆
strategy.universe`. The only behavioral edge is a symbol DECLARED in `strategy.universe` but with no
fetched price data (absent from `adj.columns`): it fails-closed in static backtest (cannot be
simulated without a price) — an acceptable, stricter-in-backtest divergence, not the parity gap this
issue targets.

### Delete the bespoke PIT block

The inline non-member rejection in `_decision_weights` (engine.py:150-156) is **removed** — now
subsumed by passing `allowed_symbols=members` to the rail. This is the unification: one check, no
second code path to drift.

### Error flow / parity

- Backtest loop & fast path already wrap `RiskBreach → BacktestError(f"{breach.detail} at {t}")`
  (engine.py:161-162, 224-225). The `out_of_universe` breach hard-fails the backtest exactly like
  the other rails and like the old PIT behavior. The existing PIT test asserts only `match="BBB"`,
  which the new message still satisfies.
- Live/paper: the `RiskBreach` propagates through `decide()` exactly as short/cap/gross breaches do
  today → the weight is rejected, never traded.
- `build_intents` is **unchanged** — the rejected weight never reaches it.

## Components touched

- `algua/risk/limits.py` — new `check_universe_membership`; `validate_decision_weights` gains the
  required `allowed_symbols` param and calls the new check.
- `algua/backtest/engine.py` *(CODEOWNERS — human design call; this spec + GATE-1 serve it)* —
  three call sites updated; the inline PIT non-member block deleted.
- `algua/live/paper_loop.py` — `decide()` passes `allowed_symbols=strategy.universe`.
- `tests/test_risk_limits.py` — 5 existing direct `validate_decision_weights` calls gain an
  `allowed_symbols` arg.

## Testing

New tests (TDD — fail first):

1. **`check_universe_membership`** unit: nonzero out-of-universe weight → `RiskBreach`
   (kind `out_of_universe`, message names offender); in-universe weights pass; a zero weight for an
   out-of-universe symbol passes (mirrors `!= 0.0`); empty series passes.
2. **`validate_decision_weights`** unit: out-of-universe weight raises with the new param; an
   in-universe set passes; ordering — a weight that is both out-of-universe and non-finite surfaces
   the finite breach first (finite precedes universe).
3. **Static backtest** integration: a static-mode strategy returning a weight for a symbol not in
   `adj.columns` raises `BacktestError` naming the symbol (the issue's core acceptance) — for BOTH
   `_decision_weights` (loop) and `_decision_weights_fast`.
4. **Live/paper** integration: `decide()` with an out-of-universe weight raises (rejected, not
   traded) against `strategy.universe`.
5. **No regression**: existing in-universe strategies/tests unaffected; the PIT non-member test
   still passes via the unified rail.

## Acceptance

- A static-mode strategy returning a weight for an out-of-universe symbol hard-fails the backtest
  the same way PIT mode does, and live rejects it identically.
- Existing strategies/tests that only weight in-universe symbols are unaffected.
- `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.

## Out of scope / non-goals

- No change to `build_intents` or live breach handling beyond what the shared rail already does.
- No vectorized PIT fast path (unrelated; deferred elsewhere).
- The declared-but-no-data edge case is fail-closed in backtest by design; no special handling.

Ref: PR #174 GATE-2 comment.
