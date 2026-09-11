# Portfolio overlays — a tighten-only stage after construction (design)

**Status:** design agreed 2026-09-10 between the operator and the agent. Implementation follows
the normal plan → review-gated flow.
**PRD anchor:** `docs/PRD.md` §6 (a strategy is a pure function of point-in-time inputs) and §4
(gates do not change per strategy kind). Nothing here touches a gate or a wall.

## Context

A FinRL-X teardown (2026-09-10) found one idea worth taking: a strategy is a pipeline
`selection → allocation → timing → risk overlay`, every stage contract-preserving on the weight
vector, so a regime gate or a stop-loss is a swappable component rather than logic smuggled into
the alpha. Algua already has the first two stages as `signal()` (scores) and a named construction
policy (weights). It has no seam for the last two: an author who wants "cut exposure when the
market is stressed" or "drop a name that has fallen 15% off its high" has to fold that into the
score, where it pollutes the alpha, escapes the sweep namespace, and is invisible to identity.

This design adds that seam. It also gives the ideation engine two new axes to vary per hypothesis
(a regime policy, a stop policy) without a new strategy kind.

Three decisions were taken with the operator and are not reopened below:

1. **Stateless.** Overlays are pure functions of the PIT view. A trailing stop is "price below
   its rolling high", a cooldown is "the trigger fired within the last C bars". No entry price,
   no position ledger, no absolute-from-entry stop.
2. **Universe-derived market state.** The regime gate reads the strategy's own universe (an
   equal-weight index of members, cross-sectional turbulence). No reference symbols (SPY, VIX).
3. **One seam.** A single ordered `overlays` list, not separate timing and risk slots. A timing
   policy is an overlay that happens to scale every weight by the same factor.

## The contract

### Config

`StrategyConfig` gains one field:

```python
overlays: list[OverlaySpec] = []
```

where `OverlaySpec` is a pydantic model defined in `algua/portfolio/overlays.py`:

```python
class OverlaySpec(BaseModel):
    policy: str                       # id resolved against OVERLAY_POLICIES
    params: dict[str, Any] = {}       # validated per-policy at load
```

`StrategyConfig` imports `OverlaySpec` from the overlays module (portfolio is already below
strategies in the layering: `strategies` imports `portfolio.construction` today). Default empty:
every existing strategy loads unchanged.

### Policy interface

```python
OverlayFn = Callable[[pd.Series, pd.DataFrame, dict[str, Any]], pd.Series]
#                     weights     view          params        -> weights
```

`weights` is the construction output (symbol-indexed, may be empty). `view` is the same long-format
PIT bar frame the signal saw, ending at the fully-closed decision bar. Return the transformed
weights. An overlay must be pure: no I/O, no global state, no clock.

### Invariants (enforced centrally, fail closed)

After every overlay in the chain, `LoadedStrategy.construct()` asserts:

- **No new symbols.** `set(out.index) ⊆ set(in.index)`. An overlay may drop or zero a name, never
  add one.
- **Tighten only.** For every symbol, `|out[s]| <= |in[s]|` and `sign(out[s]) ∈ {0, sign(in[s])}`.
  An overlay reduces exposure; it never scales up or flips a side.
- **Finite.** No NaN/inf in the output (a symbol with no opinion is passed through, not NaN-ed).

A violation raises `OverlayError(ValueError)` naming the policy index and id. The consequences
are the point: a weight vector that passed the gross-exposure and per-symbol rails inside
construction still passes them after any overlay chain, so the #135 risk walls need no change,
and a buggy or malicious policy cannot lever up. Freed weight is left as cash; nothing is
renormalised (the same cap-and-hold-cash rule the capacity cap uses).

### Composition

Inside `LoadedStrategy.construct()`, today:

```
weights = construct_fn(scores, view, construction_params)
weights = apply_capacity_cap(weights, view, capacity)   # if declared
```

becomes:

```
weights = construct_fn(scores, view, construction_params)
for i, (fn, spec) in enumerate(overlay_fns):
    weights = _checked_overlay(i, spec, fn(weights, view, spec.params), before=weights)
weights = apply_capacity_cap(weights, view, capacity)   # if declared, still last
```

The capacity cap stays last: it is the hardest liquidity wall and must see the final vector.
`construct()` is the single chokepoint every path already resolves weights through (the per-bar
backtest loop, the vectorised fast path and its parity guard, the paper tick, the live tick), so
overlays are enforced identically everywhere with no change to `backtest/decision_path.py`,
`backtest/engine.py`, or either lane. Parity is preserved by construction: the fast path calls
`construct(scores_row, view_t)` per row, which now includes the overlays.

An empty weights vector short-circuits: overlays are not called (there is nothing to tighten).

### Validation at load

`strategies/loader.py` resolves every `OverlaySpec` against the registry and calls
`validate_overlay_params(policy, params)`, which fails closed on: unknown policy id, unknown
param key, missing required key, non-finite or non-JSON value, out-of-domain value. Same shape
as `validate_construction_params`.

One new cross-check: when `feature_lookback` is declared, it must be `>=` the longest window any
declared overlay reads (`overlay_lookback(spec)`, a per-policy function). A declared lookback
smaller than an overlay window is an author bug of the same kind as an under-declared signal
lookback (#345): it would size the walk-forward embargo too small. Undeclared (`None`) stays
undeclared; the agent promote path already fails closed on that.

### Identity, approvals, sweeps

- **`config_hash`** folds `"overlays": [spec.model_dump() ...]` **only when the list is non-empty**,
  so every existing strategy's hash is byte-identical (no live-approval or result-identity churn;
  the same rule the model lane used in #376).
- **Approvals closure** (`registry/approvals.py`) is rooted from the overlays module in addition
  to the signal and construction modules, so editing a policy body invalidates a prior live
  approval exactly as editing a construction policy does.
- **Sweeps** (`backtest/sweep.py`): a grid key `overlay.<i>.<key>` tunes `overlays[i].params[key]`
  and is re-validated by the policy. `<i>` out of range or a key the policy rejects fails the sweep
  before any combo runs, matching the `construction.` namespace behaviour.
- **CODEOWNERS.** `algua/portfolio/overlays.py` is added to CODEOWNERS and to the integrity set in
  `tests/test_repo_hygiene.py`. So is `algua/portfolio/construction.py`, which is hashed into
  identity today but is not protected; that gap is closed in the same change.

## Features

New pure module `algua/features/regime.py` (no I/O, imports only pandas/numpy and
`algua.contracts`):

| Function | Returns | Notes |
|---|---|---|
| `equal_weight_index(view) -> pd.Series` | index level by timestamp, base 1.0 | mean of member simple returns from `adj_close`; a member missing a bar contributes nothing that bar |
| `rolling_drawdown(level, window) -> pd.Series` | `level / rolling_max(window) - 1` | NaN until the window is full |
| `turbulence(view, window) -> pd.Series` | Mahalanobis distance of each bar's cross-sectional return vector vs the trailing `window` covariance | pseudo-inverse; symbols without a full window are excluded that bar; NaN until `window` observations exist; the current bar is excluded from its own covariance |
| `robust_zscore(x, window) -> pd.Series` | `(x - rolling_median) / (1.4826 * rolling_MAD)` | NaN until full; zero MAD yields NaN, not inf |

All derive from `adj_close`, never raw `close` (#521). These are also usable inside `signal()` as
ordinary features.

## The first two policies

Both live in `algua/portfolio/overlays.py`, registered in `OVERLAY_POLICIES` (a
`MappingProxyType` over a static dict, like `CONSTRUCTION_POLICIES`; no dynamic registration,
because identity rests on the module's static source).

### `regime_gate` (the timing tenant)

A two-speed exposure multiplier computed from the universe. Design principle carried over from
FinRL-X: **the regime controls the risk budget, not asset selection.**

Slow gate, evaluated at each of the last `persistence` bars from the view:

| stress | true when |
|---|---|
| trend | index level `<` its `trend_window`-bar mean |
| drawdown | `rolling_drawdown(level, dd_window) < -dd_threshold` |
| volatility | `robust_zscore(turbulence(view, turb_window), z_window) > turb_z` |

`risk_score` = count of true stresses, mapped to a state: 0 → risk-on, 1 → neutral, ≥ 2 →
risk-off. The state **in effect** at bar t is the state of the most recent run of `persistence`
consecutive bars (ending at or before t) that all share one state; if no such run exists in the
view, risk-on. This is computed from the view alone by evaluating the score on each trailing bar.
State → multiplier: risk-on → `1.0`, neutral → `neutral_exposure`, risk-off →
`risk_off_exposure`.

Fast overlay: if within the last `fast_lookback` bars the index's `shock_window`-bar return was
below `-shock_return` **or** turbulence exceeded `fast_turb_z` (robust z), multiplier
`fast_exposure` applies.

Effective multiplier = `min(slow, fast)`; every weight is scaled by it. Insufficient history for
any component means that component is **not stressed** (a gate cannot fire on data it does not
have); the whole policy is a no-op until the first component can be evaluated.

Params and domains (all required, no defaults in the policy; the example strategy carries a
sensible set):

| param | domain |
|---|---|
| `trend_window`, `dd_window`, `turb_window`, `z_window`, `shock_window`, `fast_lookback` | positive int |
| `persistence` | positive int, `<= dd_window` |
| `dd_threshold`, `shock_return` | float in `(0, 1)` |
| `turb_z`, `fast_turb_z` | float `> 0` |
| `neutral_exposure`, `risk_off_exposure`, `fast_exposure` | float in `[0, 1]`, with `risk_off_exposure <= neutral_exposure` |

`overlay_lookback` = `max(trend_window, dd_window, turb_window + z_window, shock_window +
fast_lookback) + persistence`.

### `trailing_stop` (the risk-overlay tenant)

Per symbol: if `adj_close_t < (1 - stop_pct) * max(adj_close over the last lookback bars incl. t)`,
the weight is set to zero. Cooldown: if that condition was true on any of the last
`cooldown_bars` bars (evaluated on the view, no state), the weight is also zero. A symbol with
fewer than `lookback` bars uses the bars it has (`min_periods=1`); the overlay never zeroes a name
for lack of data. A symbol in `weights` but absent from `view` is passed through unchanged.

| param | domain |
|---|---|
| `lookback` | positive int |
| `stop_pct` | float in `(0, 1)` |
| `cooldown_bars` | int `>= 0` |

`overlay_lookback` = `lookback + cooldown_bars`.

## Example strategy

`algua/strategies/momentum/momentum_regime_stop.py` (the `examples/` family was retired in #121;
bundled examples live beside `cross_sectional_momentum`): the existing momentum signal with
`top_k_equal_weight`, `overlays=[regime_gate{...}, trailing_stop{...}]`, and a `signal_panel`, so
the exhaustive parity gate exercises the overlay chain. Marked `GENERATED_BY = "agent"` like the
other examples; it is a fixture, not a candidate.

## Ratchet carves

Two touched modules sit exactly on their size pins:

- `algua/strategies/base.py` (380/380): move `assert_tradable_without_fundamentals/news/model`
  (~30 lines) to a new `algua/strategies/tradable.py` as a pure move; `loader.py` and the four
  tests that import them follow. That funds the ~15 lines `construct()` and `config_hash` gain.
- `algua/backtest/sweep.py` (461/461): move `_coerce`, `_coerce_values`, `parse_grid`,
  `validate_sweep_grid` (~70 lines) to `algua/backtest/sweep_grid.py` as a pure move. That funds
  the `overlay.` namespace in `_override`.

Both moves are separate commits ahead of the feature commits so the diff reviewer sees a move,
then a change. The ratchet pins are not edited: a pinned module may shrink freely, and neither
module ends more than the stale-slack below its pin.

## Non-goals (recorded so they are not re-derived)

- Absolute stop from entry price, peak-since-entry, or any position ledger in the strategy.
- Reference symbols outside the universe.
- Turnover cap: needs previous weights, belongs at the execution boundary. Separate spec.
- A gym-style environment wrapper for RL policies: roadmap step 4 (#627).
- Weekly or monthly rebalance cadence: the engine still rejects anything but `1d`.
- Vol-targeting or any overlay that scales *up*. The tighten-only invariant is deliberate; a
  scale-up policy is a construction policy.
- Sector caps: no sector data in the view today.

## Testing

- `tests/test_features_regime.py`: each feature on synthetic panels with planted structure (a
  known drawdown, a planted covariance and a known outlier bar for turbulence, a constant series
  for zero-MAD); NaN until full window; `adj_close` only.
- `tests/test_portfolio_overlays.py`: `regime_gate` on a synthetic universe with a planted
  risk-on → risk-off → shock sequence, asserting the multiplier path and the persistence rule;
  `trailing_stop` with a planted peak-and-fall and the cooldown window; the no-data pass-through;
  invariant enforcement (a fake policy that adds a symbol, scales up, or flips sign is rejected
  with the policy index in the message); param validation domains.
- `tests/test_strategy_overlays_identity.py`: empty `overlays` reproduces today's `config_hash`
  byte-for-byte; non-empty changes it; reordering two overlays changes it; the approvals closure
  includes the overlays module.
- Loader: unknown id, bad param, `feature_lookback` smaller than `overlay_lookback` all fail
  closed with the expected messages.
- Sweep: `overlay.0.stop_pct=0.1,0.2` produces the combos; `overlay.5.x` and an unknown key fail
  before any run.
- End to end: `backtest run` on the example strategy over a fixture panel completes, and the
  exhaustive parity gate passes with overlays in the chain.
- Ratchet, lint-imports (the existing contract "portfolio construction layer is pure" already
  bounds `algua.portfolio` to `contracts` + `features`, so `overlays.py` is covered with no new
  contract), repo hygiene (CODEOWNERS integrity set), lane parity: unchanged tests must stay green.

## Files

New: `algua/portfolio/overlays.py`, `algua/features/regime.py`, `algua/strategies/tradable.py`
(move), `algua/backtest/sweep_grid.py` (move), the example strategy, the four test modules.

Modified: `algua/strategies/base.py`, `algua/strategies/loader.py`, `algua/backtest/sweep.py`,
`algua/registry/approvals.py`, `CODEOWNERS`, `tests/test_repo_hygiene.py`,
`docs/architecture.md`
("An overlay" under *How to add things*; the walls section notes tighten-only), the
`author-a-strategy` skill (a section on overlays with the two policies' param tables), and the
`interpret-results` skill (a note that a regime gate shows up as time-varying gross exposure).

## Deviations recorded during implementation

- **Three files, not one.** The overlay seam ended up split across `algua/portfolio/overlays.py`
  (types, invariants, `_OVERLAYS` registry, `resolve_overlays`), `overlay_policies.py`
  (`trailing_stop`, `regime_gate` + their validators/lookbacks) and `overlay_validation.py`
  (`OverlayError` + shared param helpers), instead of the single module this spec assumed — the
  module-size ratchet forced the carve.
- **`turbulence` returns NaN, not a huge finite number, on a rank-deficient covariance.** A
  Mahalanobis distance through `pinv` over a singular trailing covariance produces an arbitrarily
  large but finite value (observed ~1e29), which would silently pass any `> turb_z` threshold as a
  "real" stress; `turbulence` now detects the rank deficiency and returns NaN so the guard that
  reads "not stressed" on NaN is correct instead of accidentally always tripped.
- **`regime_gate` fails closed with `OverlayError` when `turb_window` doesn't exceed the universe
  size**, rather than running with the volatility leg silently degraded. An undersized
  `turb_window` makes the turbulence covariance singular on every bar, which — absent the
  precondition — would mean the volatility leg of the slow gate is always NaN/not-stressed with no
  error surfaced; a strategy would believe it has three stress checks when it only has two.
- **The "state in effect" persistence search is bounded to the last `REGIME_SEARCH_BARS` (63)
  bars**, not searched back indefinitely. An unbounded backward-fill could resolve to a same-state
  run far outside the tail the turbulence/z-score arrays are actually computed over, i.e. a
  back-fill reaching beyond the computed tail; bounding the horizon (and sizing that tail to cover
  it) guarantees every state the search can select from is one the inputs actually define.

## Rollout order

1. The two pure-move carves.
2. `features/regime.py` with tests.
3. `portfolio/overlays.py`: interface, registry, invariants, validation, `trailing_stop`.
4. Contract wiring: `OverlaySpec` on config, `construct()` chain, loader validation, identity,
   approvals, CODEOWNERS.
5. `regime_gate`.
6. Sweep namespace.
7. Example strategy, end-to-end and parity tests, docs and skills.

One PR, reviewed as a whole; the order is for the plan's task boundaries.
