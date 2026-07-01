# Spec — ADV / capacity participation cap in sizing (#344)

**Status:** Design. **Date:** 2026-07-01.

## Problem
Bar `volume` is in the schema but never consulted for sizing. A strategy sized purely by
gross + per-symbol `|weight|` caps can size a position into an untradeable fraction of a name's
liquidity. Lior's KB makes a capacity gate non-negotiable (New Backtest Procedure Step 8 GATE:
`max_participation_rate` + `min_adv_filter`; Backtest Sanity item 6: position ≤ ~5% ADV/trade).
algua has no ADV / participation constraint anywhere. Fix: make sizing capacity-aware.

## Design (KISS)

### Where — one chokepoint, full parity
`LoadedStrategy.construct(scores, view)` (`algua/strategies/base.py`) is the SINGLE place every
path resolves weights:
- backtest per-bar loop → `strategy.target_weights` → `construct`
- backtest vectorized fast path → `strategy.construct` (and its parity twin `_canonical_row`)
- live/paper `decide` → `strategy.target_weights` → `construct`

Applying the cap **inside `construct`, after the construction policy runs**, makes it apply
identically in backtest AND live/paper (parity is automatic and the fast-path parity guard keeps
holding because it compares `construct` outputs). It does NOT touch `algua/backtest/engine.py`
(CODEOWNERS-protected) — the engine consumes already-capped weights.

### What — scale, not reject
Capacity is a *sizing* limit, not a strategy bug: a position that would exceed its liquidity
budget is **scaled down** (capped), leaving the freed weight as cash. We do NOT renormalize the
vector (renormalizing would push notional back onto neighbours and could re-breach — KISS: cap and
hold cash). Scaling only ever reduces `|weight|`, so already-passing gross / per-symbol rails still
pass.

### Scope — POSITION capacity, evaluated at a declared reference AUM
This gates **position size** (a held position ≤ `max_participation_rate` of the name's ADV),
matching the operator directive ("cap each position at a configurable fraction of ADV") and
Backtest-Sanity item 6. It is **not** a per-trade order-delta participation cap — that needs the
current→target order delta from the execution/fill path (engine.py — protected; build_intents in
live). Per-trade participation is a deferred follow-up (see Non-goals).

`reference_aum` is the **declared capital the capacity is evaluated at** — the AUM-breakeven framing
the KB asks for ("at $X AUM, is this position ≤ X% of ADV?"). The backtest is unitless
(`size_type="targetpercent"`, equity-relative), so it carries no absolute AUM of its own;
`reference_aum` supplies the one needed to convert a weight fraction into a notional. The cap is
therefore defined **at reference_aum by construction** — there is no hidden claim about the
backtest's compounding equity. Assessing capacity at a fixed assumed AUM (rather than compounding
equity) is the conventional, more-interpretable capacity analysis. Live/paper is assumed to run
within the declared envelope (account equity ≤ reference_aum); asserting live account equity against
reference_aum at trade time is a deferred follow-up (decide() does not currently receive equity).

### The cap
Config lives on a new frozen `CapacityLimit` nested on `ExecutionContract`
(`algua/contracts/types.py`), so it is folded into `config_hash` automatically (`asdict`) and is
part of strategy identity. `capacity=None` (default) ⇒ cap disabled ⇒ existing strategies/tests
byte-unchanged.

```
CapacityLimit(reference_aum: float, max_participation_rate: float, adv_window_bars: int)
```
- `reference_aum` — the capital the capacity is evaluated at (dollars, finite > 0).
- `max_participation_rate` — max fraction of a name's ADV one position may occupy
  (finite, 0 < r ≤ 1).
- `adv_window_bars` — trailing window length for ADV (int ≥ 1).

Per symbol `s`, from `view` (the exact PIT frame the signal saw — already up to and including the
fully-closed decision bar `t`, never `t+1`):
```
dollar_adv_s   = mean over the last adv_window_bars rows of (close_s * volume_s)   # raw dollars
max_notional_s = max_participation_rate * dollar_adv_s
max_weight_s   = max_notional_s / reference_aum
capped_w_s     = copysign( min(|w_s|, max_weight_s), w_s )
```
- Uses **raw `close`** (not `adj_close`): real historical dollar volume = the actual liquidity.
- Trailing window on `view` (which ends at `t`) ⇒ **no look-ahead** (never sees the fill bar `t+1`
  or later); this is the same frame the signal already legitimately consumes.
- **Fail-closed illiquidity**: `dollar_adv_s ≤ 0`, NaN/inf, or a weighted symbol absent from
  `view` ⇒ `max_weight_s = 0` ⇒ weight forced to 0. No division by ADV anywhere (we divide by the
  validated positive `reference_aum` and multiply by ADV), so no divide-by-zero.
- **Full trailing window required (fail-closed)**: a symbol with **fewer than `adv_window_bars`
  observations** in `view` has no established ADV ⇒ `max_weight_s = 0` ⇒ weight forced to 0. A
  newly-listed / sparse name is thus treated as un-sized, not sized off one or two noisy bars
  (this is the `min_adv_filter` spirit).
- **Shorts**: `|weight|` with sign preserved via `copysign`.
- **Input contract (fail-closed)**: if `view` lacks the `close` or `volume` column (or is empty
  while weights are non-empty), `apply_capacity_cap` raises `ConstructionError` rather than silently
  zeroing every position — so a live/paper `view` that dropped `volume` or supplied no history is a
  loud failure, not a silent flatten. A test asserts the live/paper `view` carries volume + history.

### Implementation
- `algua/portfolio/construction.py`: pure `apply_capacity_cap(weights, view, capacity)` +
  `_dollar_adv(view, window)` helper. `algua.portfolio` may import `algua.contracts` (import-linter
  allows it), so `CapacityLimit` is imported for typing.
- `algua/contracts/types.py`: `CapacityLimit` frozen dataclass with `__post_init__` validation;
  `ExecutionContract.capacity: CapacityLimit | None = None` (last field, keyword-only usage in the
  whole codebase — safe).
- `algua/strategies/base.py`: `LoadedStrategy.construct` calls `apply_capacity_cap` when
  `execution.capacity is not None`.

### Validation (fail-closed, mirrors existing contract guards)
`CapacityLimit.__post_init__`: `reference_aum` finite > 0; `max_participation_rate` finite, in
(0, 1]; `adv_window_bars` int ≥ 1 (reject bool). A non-finite/zero value silently disabling the
rail is the exact footgun `ExecutionContract` already guards against — same treatment.

## Non-goals / deferred
- **Per-trade order-delta participation**: capping the current→target *trade* (not just the held
  position) at X% ADV needs the order delta from the fill path (engine.py — protected; live
  build_intents). Deferred; the position cap is the KB GATE and ships here.
- **Live account-equity vs reference_aum assertion**: fail-closed check that live/paper account
  equity ≤ declared `reference_aum` at trade time (decide() would need equity threaded in). Deferred.
- **AUM-breakeven figure in the report** (issue's secondary ask): surfacing a capacity number in
  metrics/report touches the reporting layer and is advisory, not the GATE. Deferred as a
  follow-up; the sizing cap (the actual KB GATE) ships here.
- `min_adv_filter` as a *universe* pre-filter: the participation cap subsumes the sizing concern
  (an illiquid / sparse name is forced to 0 weight); a separate universe filter is a later refinement.
- Out of scope: execution/paper reconcile (#312), kb/CLI doc-sync (#331).

## Test plan
- `apply_capacity_cap`: caps an oversized long; scales proportionally to ADV; leaves an
  under-budget weight untouched; forces zero-ADV / missing-symbol / short-history (< window) names
  to 0; preserves short sign; respects `adv_window_bars` (only the trailing window feeds ADV);
  no-op when nothing exceeds; raises on a `view` missing `close`/`volume`.
- `CapacityLimit` validation rejects non-finite/≤0 aum, rate outside (0,1], window < 1, bool window.
- `config_hash` changes when `capacity` is set, unchanged when `None`.
- Engine parity: a capacity-configured strategy runs through both the loop and the fast path with
  the parity guard green (cap applied identically).
- Live/paper `decide` applies the same cap (parity with backtest).
- No-capacity default leaves all existing behavior byte-identical.
