# Weight-space risk conventions — kb doc + author-a-strategy reference (#136)

## Context & decision

algua is **weight-based** (target-portfolio), confirmed by design review with an independent second
opinion. **Stop-loss / take-profit / fixed risk-reward are trade-centric constructs and are
explicitly NOT used** — risk discipline is expressed in **weight space**. The previously-floated
execution-side stop overlay is **rejected**: it is path-dependent, breaks backtest↔live parity unless
fully simulated, and conflicts with the allocation-native decision.

`author-a-strategy` today has essentially no risk-design guidance (one line deferring to the engine's
gross cap). This adds the missing **weight-space risk conventions** the authoring agent applies inside
`compute_weights`.

## The dividing principle (why kb doc, not code)

Hard capital-protection limits are **code walls** (#135, merged). The conventions here are
**context-dependent judgment** with legitimate exceptions — encoding them as hard checks would wrongly
reject valid strategies. They live in **prose the agent weighs, above the walls** — exactly like the
sibling `research-methodology.md` (#138).

## Artifact & placement

- **New file:** `kb/principles/risk-conventions.md`, peer to `kb/principles/research-methodology.md`.
- It resolves the existing reserved `[[risk-conventions]]` wikilink in `research-methodology.md`
  (currently *"reserved for issue #136; not yet authored"*) — that line is updated to a real
  one-line description.
- **Governing rules** (adopted from `research-methodology.md`, stated at the top of the new doc):
  1. Never restate a wall as the enforcement — the code is the source of truth; the doc points at it.
  2. Never claim a control the code has not built — be honest about what is judgment-only.
  3. Never frame a wall as optional — the advisory tone is about *judgment*, never about the floor.
- **Tone:** advisory judgment *above* the walls; prose-led, with **targeted PIT-safe code snippets**
  only where a convention has a look-ahead trap.

## Doc structure

### 1. Framing
Stop-loss / take-profit / fixed risk-reward are trade-centric and NOT used; risk is expressed in
weight space. Name the rejected execution-side stop overlay as rejected (path-dependence,
backtest↔live parity). One paragraph.

### 2. The walls are the floor (enforced today)
A table mirroring `research-methodology.md`'s, citing the real #135 code by `path::symbol`. Every
convention below sits **above** these and never replaces them:

| Risk control | Wall (`path::symbol`) | Contract field |
|---|---|---|
| Single-name concentration cap | `algua/risk/limits.py::check_max_weight_per_symbol` | `ExecutionContract.max_weight_per_symbol` |
| Long/short gate (long-only default) | `algua/risk/limits.py::check_short_policy` | `ExecutionContract.allow_short` |
| Gross-exposure cap | `algua/risk/limits.py::check_gross_exposure` | `ExecutionContract.max_gross_exposure` |
| Finite-weights fail-closed | `algua/risk/limits.py::check_finite_weights` | — |
| Execution-timing look-ahead | `algua/backtest/engine.py::simulate` (shifts by `decision_lag_bars`) | `ExecutionContract.decision_lag_bars >= 1` |

All **decision-weight** walls funnel through `algua/risk/limits.py::validate_decision_weights`, called
by backtest + paper + live + fast-path — so the rails cannot drift between research and live.

**Not a decision-weight convention:** `algua/risk/limits.py::check_drawdown` is a separate hard
**equity/NAV drawdown breaker** enforced in the paper/live loops (`algua/live/paper_loop.py`,
`algua/live/live_loop.py`), not in `validate_decision_weights`. The doc names it once to keep it
distinct from the *soft, per-name* drawdown-based weight decay in §3.3 — they are different things
(account-level circuit breaker vs. per-symbol allocation judgment).

### 3. The six weight-space conventions
Each: **what / why / the PIT trap to avoid**, plus a 3–5 line snippet where it earns its place. All
snippets operate on info **up to the decision bar**; the engine applies the `t→t+1` lag centrally, so
authors return decision-time (pre-lag) weights and never shift themselves.

1. **Conviction-scaled sizing** — weight by signal strength, not equal-weight by default.
   ```python
   raw = scores.clip(lower=0.0)            # scores: info up to t, indexed by symbol
   w = raw / raw.sum() if raw.sum() > 0 else raw * 0.0
   ```
2. **Inverse-volatility sizing** — size inversely to *trailing* realized vol so risk, not dollars, is
   spread evenly (risk-parity-style). The snippet computes inverse-vol *normalized* weights; it does
   **not** scale gross to hit a vol number. True portfolio-vol *targeting* (scaling exposure to a
   target σ) is **judgment-only today** and a candidate for the #141 layer — and it must still respect
   the hard gross / per-name walls, never bypass them.
   ```python
   rets = wide_close.pct_change()          # wide_close: closes UP TO t, no future bar
   vol = rets.tail(lookback).std()
   inv = 1.0 / vol.replace(0.0, pd.NA)
   w = (inv / inv.sum()).fillna(0.0)       # inverse-vol weights (sum to 1; NOT vol-targeted)
   ```
3. **Drawdown-based weight decay** — the allocation-native "stop": cut a name's weight as its
   drawdown-from-high grows, **info-by-decision-bar only**, `decision_lag_bars >= 1`, no same-bar
   trigger+fill. This is NOT a stop-loss order, and `decay_full_at` is a *soft per-name* threshold —
   not the hard equity breaker `check_drawdown` (see §2).
   ```python
   dd = 1.0 - close / close.cummax()       # close: a symbol's closes UP TO t
   scale = (1.0 - dd.iloc[-1] / decay_full_at).clip(0.0, 1.0)  # soft per-name decay threshold
   w = base_w * scale
   ```
4. **Signal-invalidation** — when the thesis condition disappears, decay to zero rather than
   hold-and-hope.
   ```python
   w = base_w.where(thesis_holds, 0.0)     # thesis_holds: bool Series, info up to t
   ```
5. **Turnover / transaction-cost awareness** — weights are sticky for a reason; an edge that evaporates
   under realistic churn/costs is not an edge. **Prose, not a snippet:** `compute_weights(view,
   params)` is *stateless per bar* — it receives no prior target weights, so it must **not** read a
   `prev_w`, a broker position, or any module-global to band turnover (that introduces hidden state
   and breaks per-bar/backtest parity). The stateless lever an author *does* have is to make the
   **signal itself** slow-moving — smooth or threshold the score so marginal changes don't thrash the
   book. Explicit previous-weight no-trade banding needs the prior decision weights threaded in, which
   is **#141 portfolio-construction-layer** scope, not a per-bar `compute_weights` concern.
6. **Exposure caps (per-name / per-sector / gross-net awareness)** — soft discipline *above* the hard
   #135 caps; cross-links the hard versions as the floor (never restated as optional).

### 4. The right yardstick (reject R:R)
R:R / fixed risk-reward is the **wrong** measure for cross-sectional/portfolio strategies. The right
criteria: **return per unit risk, max drawdown, turnover-adjusted Sharpe, capacity, tail exposure, and
degradation across walk-forward splits.** Points to `interpret-results` (gate thresholds live there)
and #137 (merged); does not duplicate them.

### 5. Not a layer yet (honest, mirrors the sibling's "Not yet enforced")
These are **per-strategy prose** today, not a composable/enforceable layer. **#141 (open)** —
"Formalize portfolio construction as a distinct layer" — is where several of these conventions move
from judgment into code (vol-target, exposure caps, turnover bands become a portfolio-construction
step). Until then, judgment is the only guard for the soft versions.

### 6. Related principles (cross-links)
- `[[research-methodology.md]]` — the anti-leakage/generalization sibling (#138).
- `interpret-results` skill — the metrics/gate yardstick.

## Skill edit (lightweight)
Extend the existing **"Read the methodology before authoring"** bullet in
`.codex/skills/author-a-strategy/SKILL.md` to also point at `kb/principles/risk-conventions.md`
(weight-space risk: inverse-vol sizing, drawdown decay, conviction sizing — judgment above the #135 walls).
`.claude/skills/author-a-strategy` is a **symlink** to the `.codex` copy, so one edit propagates.
No new "Risk design" subsection — the skill stays lean; the doc holds the substance.

## Out of scope / explicit non-goals
- **No code walls, no new enforcement.** Pure docs.
- **No new tests** (no code changed). The repo quality gate still runs before the PR.
- **`interpret-results` left untouched** — the yardstick lives in the new doc and is referenced, not
  duplicated.
- **No SL/TP/R:R machinery**, per the design decision.
- The enforceable portfolio-construction layer is **#141's** scope, not this issue's.

## Acceptance criteria
- `kb/principles/risk-conventions.md` exists and:
  - States the no-SL/TP framing and the three governing rules.
  - Cites the #135 walls by accurate `path::symbol` and never frames one as optional.
  - Covers all six conventions with PIT-safe snippets where snippeted.
  - States the right yardstick and rejects R:R.
  - Honestly marks the soft conventions as not-yet-a-layer and forward-points to #141.
- `research-methodology.md`'s `[[risk-conventions]]` line is updated (no longer "not yet authored").
- `author-a-strategy` SKILL.md references the new doc (one lightweight bullet); the symlinked
  `.claude` copy reflects it.
- Quality gate green: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

## Issue cross-references
- **#135** (CLOSED) — hard risk walls; the floor this sits above.
- **#137** (CLOSED) — gate / DS-integrity walls; the yardstick coordination.
- **#138** (CLOSED) — `research-methodology.md`, the sibling doc and tone model.
- **#141** (OPEN) — formalize portfolio construction as a distinct layer; where conventions become
  enforceable.
