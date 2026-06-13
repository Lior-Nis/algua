# Exhaustive `signal_panel` parity gate at promotion (#178)

**Status:** design approved, pre-implementation
**Date:** 2026-06-13
**Issue:** #178 — Fast-path parity: make exhaustive at promotion/approval (bounded 16-bar sample can miss a buggy `signal_panel`)
**Ref:** PR #174 GATE-2 comment (pre-existing; the bounded sample is from the #114 fast-path guard)

## Problem

`algua/backtest/engine.py::_assert_parity` checks the vectorized fast path against the canonical
per-bar `construct(signal(view), view)` on only a **bounded deterministic sample**
(`_PARITY_SAMPLE = 16` bars). A strategy whose `signal_panel` diverges from its per-bar `signal` on
an **unsampled** bar can therefore have the divergent (wrong) weights become the **backtest**
result, while paper/live always use the per-bar `decide()` (canonical). That is a backtest↔live
fidelity gap for a buggy/custom `signal_panel`.

Mitigations today: the full-parity test asserts exhaustive agreement for the bundled example, and
live/paper never use `signal_panel`. So the runtime exposure is limited to **research fidelity** for
custom panel strategies.

## Two findings that shape the design

1. **The agent `research promote` backtest cannot be relied on to exercise `signal_panel`.** An agent
   *cannot pass* promotion without PIT — but PIT is enforced at the **gate** (`run_gate` →
   `resolve_pit_ok`/`pit_required`), *not* at preflight. If an agent omits `--universe`,
   `walk_forward` still runs with `universe_by_date=None`, which permits `_decision_weights_fast`
   (the panel) — and the PIT refusal only fires later, after the holdout is touched. Conversely, the
   normal PIT path forces the canonical loop (`universe_by_date is not None`) and never runs the
   panel at all. **Either way**, "make the existing backtest exhaustive" does not reliably close the
   gap. The panel must be checked by a **standalone verifier** that runs it in *static* mode over the
   window and compares it to the canonical loop on every bar — robust to whichever path the actual
   backtest took. (Pre-existing, orthogonal: a non-PIT agent promote burns the holdout before the
   PIT gate refuses it. Moving that refusal into preflight is out of scope for #178 — noted below.)

2. **Go-live runs no backtest and never touches `signal_panel`.** The `forward_tested → live`
   ceremony verifies a signature + a forward certificate minted from *paper* trading, which uses the
   canonical per-bar `decide()`. There is no panel computation in that path to gate. The genuinely
   load-bearing moment is `research promote` (backtested→candidate) — the one place a panel-derived
   backtest number gates a stage transition.

**Scope decision:** enforce the exhaustive parity gate at `research promote` only. Go-live is a
no-op for this property (no panel ever runs there). Ordinary backtests keep the bounded sample.

## Design

Three small changes; one new engine function.

### 1. `algua/backtest/engine.py` — new `verify_signal_panel_parity`

```python
def verify_signal_panel_parity(
    strategy: LoadedStrategy, provider: DataProvider, start: datetime, end: datetime
) -> None:
    ...
```

- **No-op** when `strategy.signal_panel_fn is None` (nothing to verify).
- Otherwise: fetch the strategy's declared **static** universe bars over `[start, end]` (the same
  fetch + `adj_close` pivot `simulate` does; mirror `simulate`'s empty-bars guard — a bad/empty
  provider surfaces as `BacktestError`), then compute **both**:
  - the vectorized fast-path weights via a new `_fast_weights(strategy, bars, adj)` helper —
    `_decision_weights_fast`'s body **without** the internal bounded `_assert_parity` call (see the
    refactor below); and
  - the canonical per-bar loop `_decision_weights(strategy, bars, adj)` — **static** (no
    `universe_by_date`, no `fundamentals`),

  assert the two matrices share index + columns, then assert they agree on **every** bar within
  `WEIGHT_TOL` (rtol=0, the same tolerance the runtime guard uses), with a **NaN-safe** comparison
  (an `isna()` mismatch counts as divergence — defensive only, since both paths `fillna(0.0)` so a
  NaN cannot structurally survive). On any divergence, raise `BacktestError` naming the **first**
  divergent bar and the offending symbol(s), with both sides' weights — mirroring `_assert_parity`'s
  message style.

**Refactor (no behavior change):** split `_decision_weights_fast` into `_fast_weights` (the
panel→per-bar-construct computation, returning the full weights matrix) + the existing bounded
`_assert_parity` call. `_decision_weights_fast` becomes `_fast_weights(...)` followed by
`_assert_parity(...)` — identical to today. The exhaustive verifier calls `_fast_weights` directly,
so it does the full comparison *itself* (single clean error path) rather than tripping the bounded
sample's message first when a divergence happens to fall on a sampled bar.
- **Static by design.** This verifies a *code property* — that `signal_panel` agrees with its
  per-bar `signal` twin — independent of universe masking. Because the agent's promote backtest runs
  under PIT (which forces the loop and never exercises the panel), the panel must be checked here
  directly, in the static mode where the fast path is actually used.
- Provider/empty-bar errors surface as `BacktestError` (mirror `simulate`'s fetch guards), so a bad
  provider fails the gate cleanly rather than crashing.

The comparison of the two full frames **is** the exhaustive check: if they agree everywhere, the
bounded sample (a subset of bars, run internally by `_decision_weights_fast`) trivially also agrees.
No new sampling mode or flag is threaded through the engine.

### 2. `algua/registry/promotion.py` — `promotion_preflight` calls the verifier

`promotion_preflight` already `load_strategy(name)`s `_loaded` for the fundamentals wall. Extend its
signature with `provider: DataProvider`, `start: datetime`, `end: datetime`. Immediately after the
fundamentals refusal, if `_loaded is not None`, call the verifier **on the already-loaded `_loaded`**
(do not reload by name — verify the same object the rest of the flow uses):

```python
verify_signal_panel_parity(_loaded, provider, start, end)
```

A divergence raises **before the holdout is touched and before any gate row / token is minted** — a
hard pre-peek refusal, in the same phase and character as the existing relaxation / stage /
fundamentals walls. For synthetic test names that do not resolve to a bundled module,
`_loaded is None` and the verifier is skipped (provider unused). Fundamentals strategies never reach
the verifier (the `needs_fundamentals` wall above refuses them first; the loader also rejects a
`signal_panel` on a fundamentals strategy) — so the verifier only ever runs on non-fundamentals
static-universe strategies, exactly where the fast path applies.

### 3. `algua/cli/research_cmd.py` — pass the new args

The `promotion_preflight(...)` call (currently `research_cmd.py:100`) passes `provider=provider`,
`start=start_dt`, `end=end_dt` alongside the existing arguments.

## Why this is not a holdout peek

The check computes decision **weights** and compares two **code paths**; it records nothing,
evaluates no out-of-sample **returns**, and does not influence the gate metric or decision beyond a
binary refuse-or-proceed. It must see all bars (including the holdout span) to be exhaustive — the
same character as the fundamentals/stage walls that already inspect the strategy pre-peek. The
holdout "burn" concerns evaluating strategy *performance* on OOS returns; a parity check consumes no
statistical degrees of freedom.

## Ordinary backtests unchanged

`_PARITY_SAMPLE`, `_parity_sample_positions`, `_assert_parity`, and the fast path's bounded runtime
guard are untouched. Only the promotion preflight runs the exhaustive comparison. This satisfies the
acceptance criterion that ordinary backtests keep the bounded-sample cost.

## Testing (TDD)

- **Unit — catches the unsampled divergence:** construct a `LoadedStrategy` whose `signal_panel`
  diverges from its per-bar `signal` on a bar that is **not** in `_parity_sample_positions(warmup,
  n)`. Assert (a) the bounded `_assert_parity` does *not* flag it (documents the gap), and (b)
  `verify_signal_panel_parity` raises `BacktestError` naming that bar.
- **Unit — faithful panel passes:** a panel that equals its per-bar twin everywhere → verifier
  returns without raising.
- **Unit — no-op:** `signal_panel_fn is None` → verifier returns immediately (no provider call).
- **Integration — promote refuses:** `research promote` of a strategy with a divergent panel fails
  with no promotion and no `holdout_evaluations` / `gate_evaluations` row recorded (the refusal
  precedes `walk_forward`).
- **Integration — faithful promote unaffected:** a faithful panel strategy promotes exactly as
  before.
- **Regression:** existing bounded-sample backtest tests stay green (the `_fast_weights` refactor is
  behavior-preserving — `_decision_weights_fast` still runs the bounded `_assert_parity`); the four
  existing `promotion_preflight` test callers get the new args (synthetic names → `_loaded is None`
  → no-op, provider unused).
- **Edge — empty provider:** a provider returning no bars for a panel strategy → `verify_signal_
  panel_parity` raises `BacktestError` (mirrors `simulate`'s guard), refusing promotion.

## Acceptance (from the issue)

- A `signal_panel` that diverges from `signal` on any bar cannot pass promotion. ✅ (verifier in
  preflight)
- Ordinary backtests keep the bounded-sample cost. ✅ (runtime guard untouched)
- `pytest && ruff check . && mypy algua && lint-imports` green.

## Out of scope / deferred

- Go-live "approval" parity (no panel runs in that path — would gate a property that no longer
  affects capital).
- Flipping ordinary/candidate-stage backtests to exhaustive (contradicts the acceptance criterion).
- Any change to the fast-path algorithm or `_PARITY_SAMPLE` value.
- Moving the agent **PIT refusal** from `run_gate` into preflight (so a non-PIT agent promote is
  refused *before* the holdout is touched). Real and surfaced by the GATE-1 review, but pre-existing
  and orthogonal to the parity gate — a separate follow-up.
