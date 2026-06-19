# Weight-space risk conventions — sizing and protection in allocation space

This note is algua's **risk-design judgment layer** for the authoring agent. It sits *above* the code
walls that enforce hard capital protection (#135). It is **advisory** — sizing and protection are
**context-dependent judgment** with legitimate exceptions, which is exactly why they are prose here and
not hard checks (a hard check would wrongly reject valid strategies).

algua is **weight-based**: a strategy returns target portfolio weights, and risk discipline is
expressed in **weight space**. **Stop-loss, take-profit, and fixed risk-reward are trade-centric
constructs and are NOT used here.** The execution-side stop overlay was considered and **rejected**: it
is path-dependent and breaks backtest↔live parity unless fully simulated. Everything below expresses
protection by *shaping the target weights*, not by bolting an order-level stop on top.

Like its sibling [[research-methodology]], this note lives by three rules:

1. **It never restates a wall as the enforcement.** Where a control is walled in code, the code is the
   source of truth; this note explains it and points at it.
2. **It never claims a control the code has not built.** Where something is judgment-only, it says so.
3. **It never frames a wall as optional.** The advisory tone is about *judgment above the floor*, never
   about the floor itself.

## The walls are the floor (enforced today)

The **decision-weight** checks below — the concentration cap, long/short gate, gross cap, and
finite-weights guard (the first four rows) — run on **every** decision (backtest, paper, live) through
`algua/risk/limits.py::validate_decision_weights`, so those rails can't drift between research and live.
**Execution-timing look-ahead is a separate hard wall** (last row): the engine applies the `t→t+1` shift
in `algua/backtest/engine.py::simulate`, and the `decision_lag_bars >= 1` floor is enforced on the
contract itself. The conventions in the next section sit *above* all of these and never replace them:

| Hard control | Code wall (`path::symbol`) | Contract field |
|---|---|---|
| **Single-name concentration cap** — bounds `\|weight\|` per symbol (caps the largest position; gross caps the sum). | `algua/risk/limits.py::check_max_weight_per_symbol` | `ExecutionContract.max_weight_per_symbol` |
| **Long/short gate** — long-only by default; negative weights breach unless shorts are declared. | `algua/risk/limits.py::check_short_policy` | `ExecutionContract.allow_short` |
| **Gross-exposure cap** — bounds the sum of `\|weight\|`. | `algua/risk/limits.py::check_gross_exposure` | `ExecutionContract.max_gross_exposure` |
| **Finite-weights fail-closed** — NaN/inf/non-numeric/duplicate-symbol weights hard-breach (no silent fillna). | `algua/risk/limits.py::check_finite_weights` | — |
| **Execution-timing look-ahead** — the `t→t+1` decision lag; orders fill no earlier than `t + lag`. | `algua/backtest/engine.py::simulate` (shifts by `decision_lag_bars`) | `ExecutionContract.decision_lag_bars >= 1` |

There is also a hard **equity/NAV drawdown breaker**, `algua/risk/limits.py::check_drawdown`, enforced
in the paper/live loops (`algua/live/paper_loop.py`, `algua/live/live_loop.py`) — **not** in
`validate_decision_weights`. It is an account-level circuit breaker, a different thing from the *soft,
per-name* drawdown-based weight decay below. Don't conflate them.

## Weight-space conventions

Post-#141 (PR #174), weight-space risk **composes across two layers**, and each convention below lives
in one of them:

- **Score-shaping — `signal(view, params)`.** Judgment about *how strong each score is*: conviction
  sizing, slow-moving (turnover-aware) signals, decaying a name's score on drawdown or a dead thesis.
  `signal` returns cross-sectional **scores** (higher = more attractive), not weights.
- **Weights-from-scores — the named construction policy.** Judgment that turns scores into target
  weights: selection, the weighting scheme (equal-, inverse-vol-, score-proportional-), and gross
  normalization. You declare a policy in `CONFIG.construction`; a bespoke scheme is **added as a named
  policy** in `algua/portfolio/construction.py`, never inlined.

`view` holds bars only **up to the current decision bar**; the engine applies the `t→t+1` lag
centrally, so both layers work in *decision-time* (pre-lag) terms and **never shift, and never read a
future bar.** The snippets below operate strictly on info up to `t`; where a snippet normalizes scores
into weights, that final step is the construction policy's, not the signal's.

### Conviction-scaled sizing
Size by signal strength, not equal-weight by default. A stronger signal earns a larger target weight; a
marginal one earns a small one. Equal-weighting throws away the information in *how* strong each signal
is. **Composes across both layers:** emit the conviction in the **score** (`signal`), then pick a
magnitude-weighting policy — `score_proportional_long` is the shipped version of the sketch below (it
drops missing/non-finite scores and weights only the strictly-positive ones) — to turn it into weights.

```python
raw = scores.clip(lower=0.0)            # scores: conviction per symbol, info up to t (signal's job)
w = raw / raw.sum() if raw.sum() > 0 else raw * 0.0   # scores → weights: the construction policy's job
```

### Inverse-volatility sizing
Spread *risk*, not dollars, evenly: size inversely to each name's **trailing** realized vol
(risk-parity-style), so a quiet name and a wild one don't carry the same risk. Compute vol on a
**trailing window**, never the full sample (full-sample stats leak the future — see
[[research-methodology]]).

```python
rets = wide_close.pct_change()         # wide_close: closes UP TO t, no future bar
vol = rets.tail(lookback).std()        # trailing realized vol
inv = 1.0 / vol.replace(0.0, pd.NA)
w = (inv / inv.sum()).fillna(0.0)      # inverse-vol weights (sum to 1)
```

This **normalizes** by inverse vol; it does not scale gross exposure to hit a target σ. Inverse-vol
sizing is a **weighting scheme**, so it belongs in a construction policy — but the starter library
(`top_k_equal_weight`, `equal_weight_positive`, `score_proportional_long`) doesn't ship one, so it's a
policy you **add** to `algua/portfolio/construction.py` and name in `CONFIG`. True portfolio-**vol
targeting** (levering up/down to a target volatility) is **not a policy yet** — and even as one it must
respect the hard gross / per-name caps, never bypass them.

### Drawdown-based weight decay
The allocation-native "stop": as a name's drawdown-from-its-high grows, **decay its target weight
toward zero** — cut the allocation rather than hold and hope. This is NOT a stop-loss order: there is no
intrabar trigger and no same-bar fill. It uses only info available by the decision bar and rides the
central `decision_lag_bars >= 1` lag like everything else. **Score-shaping:** decay the name's
**score** in `signal` (the snippet's `base_w * scale` shape applies equally to a score), and the
construction policy sizes off the decayed scores.

```python
dd = 1.0 - close / close.cummax()      # close: a symbol's closes UP TO t
scale = (1.0 - dd.iloc[-1] / decay_full_at).clip(0.0, 1.0)  # soft per-name threshold
score = base_score * scale             # decay the score in signal; construction sizes off it
```

`decay_full_at` is a **soft per-name** decay threshold you choose — not the hard equity breaker
`check_drawdown` (above), which is account-level and enforced.

### Signal-invalidation
When the condition that justified the position disappears, **decay the score toward "no opinion"**
rather than holding the position on inertia. A thesis that no longer holds is not a position to keep
"until it comes back". **Score-shaping:** drop the name from the score (a missing score is "not
selectable"), so the construction policy stops weighting it — no weight math in `signal`.

```python
score = base_score.where(thesis_holds)  # thesis_holds: bool per symbol, info up to t; else NaN = drop
```

### Turnover / transaction-cost awareness
Weights are sticky for a reason: an edge that evaporates under realistic churn and costs is not an edge.
The stateless lever you have inside `signal` is to make the **signal itself** slow-moving — smooth or
threshold the score so a marginal change doesn't thrash the book.

`signal(view, params)` is **stateless per bar**: it receives no prior scores or weights. Do **not** read
a previous weight, a broker position, or any module-global to band turnover — that smuggles in hidden
state and breaks per-bar / backtest parity. Explicit previous-weight no-trade banding needs the prior
decision weights threaded in, which belongs to a **construction policy** (the layer is built, but no
turnover-banding policy ships yet), not to a per-bar signal.

### Exposure caps (per-name / per-sector / gross-net awareness)
Be deliberate about concentration *before* the hard caps catch you: diversify across names, watch sector
clustering, and stay aware of gross vs net. The **hard** versions — per-name |weight| and gross — are
walls (#135, table above) and are non-negotiable; this convention is the *soft* judgment that keeps you
well inside them. Per-sector and net-exposure caps are **not** walled today and are **not yet
construction policies** — they are yours to honor, and natural additions to
`algua/portfolio/construction.py`.

## The right yardstick (R:R is the wrong one)

For cross-sectional / portfolio strategies, **risk-reward ratio is the wrong measure** — it's a
trade-centric yardstick that doesn't describe a target-weight book. Judge a strategy on:

- **Return per unit risk** — Sharpe / Sortino on out-of-sample windows.
- **Max drawdown** — the depth you'd have lived through.
- **Turnover-adjusted Sharpe** — net of realistic costs, not gross.
- **Capacity** — how much capital the edge survives.
- **Tail exposure** — the shape of the worst outcomes, not just the average.
- **Degradation across walk-forward splits** — a consistent edge, not one lucky window.

The promotion gate's thresholds and how to read them live in the `interpret-results` skill and the gate
(#137); this note doesn't restate them.

## The layer is built — what's a policy and what's still judgment

#141 — *formalize portfolio construction as a distinct layer* — **shipped** (PR #174): a strategy now
returns **scores** from `signal`, and a **named, library-provided construction policy** maps scores →
weights. The starter library is `top_k_equal_weight`, `equal_weight_positive`, and
`score_proportional_long`; a bespoke weighting scheme is **added as a named policy** in
`algua/portfolio/construction.py` (additions-only), not inlined.

Some conventions above are now expressible as policies (selection, equal- / score-proportional
weighting); others have **no starter policy yet** — inverse-vol sizing and vol-targeting, per-sector and
net caps, and previous-weight turnover banding. For those, until a policy is added, the score-shaping
you can do in `signal` plus deliberate judgment is the only guard. The #135 hard walls still run after
construction regardless.

## Related principles

- [[research-methodology]] — the anti-leakage / generalization sibling: the leakage vectors no wall
  catches, and how to read results honestly.
