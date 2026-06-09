# Weight-space risk conventions (#136) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `kb/principles/risk-conventions.md` — an advisory weight-space risk-design doc the authoring agent weighs *above* the hard #135 code walls — and reference it from `author-a-strategy`.

**Architecture:** Pure docs. One new kb note (sibling to `research-methodology.md`, same three governing rules + walls-are-the-floor structure), one cross-link update in `research-methodology.md`, one lightweight reference bullet in the `author-a-strategy` skill (edited at its `.codex` source; `.claude` is a symlink). No code, no enforcement, no tests.

**Tech Stack:** Markdown / Obsidian vault (`kb/`), algua CLI for verification.

Spec: `docs/superpowers/specs/2026-06-09-weight-space-risk-conventions-issue-136-design.md` (GATE-1 approved).

---

## File structure

- **Create:** `kb/principles/risk-conventions.md` — the advisory doc (framing → walls-are-the-floor table → six conventions → right-yardstick → not-a-layer-yet → related).
- **Modify:** `kb/principles/research-methodology.md` — update the reserved `[[risk-conventions]]` line (drop "not yet authored").
- **Modify:** `.codex/skills/author-a-strategy/SKILL.md` — extend the "Read the methodology before authoring" bullet to also point at the new doc.

---

### Task 1: Author `kb/principles/risk-conventions.md`

**Files:**
- Create: `kb/principles/risk-conventions.md`

- [ ] **Step 1: Write the file verbatim**

````markdown
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

These hard checks run on **every** decision — backtest, paper, and live — through
`algua/risk/limits.py::validate_decision_weights`, so the rails cannot drift between research and live.
The conventions in the next section sit *above* these and never replace them:

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

Each convention shapes the target weights your `compute_weights(view, params)` returns. `view` holds
bars only **up to the current decision bar**; the engine applies the `t→t+1` lag centrally, so you
return *decision-time* (pre-lag) weights and **never shift, and never read a future bar.** The snippets
below operate strictly on info up to `t`.

### Conviction-scaled sizing
Size by signal strength, not equal-weight by default. A stronger signal earns a larger target weight; a
marginal one earns a small one. Equal-weighting throws away the information in *how* strong each signal
is.

```python
raw = scores.clip(lower=0.0)            # scores: conviction per symbol, info up to t
w = raw / raw.sum() if raw.sum() > 0 else raw * 0.0
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

This **normalizes** by inverse vol; it does not scale gross exposure to hit a target σ. True
portfolio-**vol targeting** (levering up/down to a target volatility) is **judgment-only today** and a
candidate for the portfolio-construction layer (#141) — and even then it must respect the hard gross /
per-name caps, never bypass them.

### Drawdown-based weight decay
The allocation-native "stop": as a name's drawdown-from-its-high grows, **decay its target weight
toward zero** — cut the allocation rather than hold and hope. This is NOT a stop-loss order: there is no
intrabar trigger and no same-bar fill. It uses only info available by the decision bar and rides the
central `decision_lag_bars >= 1` lag like everything else.

```python
dd = 1.0 - close / close.cummax()      # close: a symbol's closes UP TO t
scale = (1.0 - dd.iloc[-1] / decay_full_at).clip(0.0, 1.0)  # soft per-name threshold
w = base_w * scale
```

`decay_full_at` is a **soft per-name** decay threshold you choose — not the hard equity breaker
`check_drawdown` (above), which is account-level and enforced.

### Signal-invalidation
When the condition that justified the position disappears, **decay the weight to zero** rather than
holding it on inertia. A thesis that no longer holds is not a position to keep "until it comes back".

```python
w = base_w.where(thesis_holds, 0.0)    # thesis_holds: bool per symbol, info up to t
```

### Turnover / transaction-cost awareness
Weights are sticky for a reason: an edge that evaporates under realistic churn and costs is not an edge.
The stateless lever you have inside `compute_weights` is to make the **signal itself** slow-moving —
smooth or threshold the score so a marginal change doesn't thrash the book.

`compute_weights(view, params)` is **stateless per bar**: it receives no prior target weights. Do
**not** read a previous weight, a broker position, or any module-global to band turnover — that smuggles
in hidden state and breaks per-bar / backtest parity. Explicit previous-weight no-trade banding needs
the prior decision weights threaded in, which belongs to the portfolio-construction layer (#141), not to
a per-bar signal.

### Exposure caps (per-name / per-sector / gross-net awareness)
Be deliberate about concentration *before* the hard caps catch you: diversify across names, watch sector
clustering, and stay aware of gross vs net. The **hard** versions — per-name `|weight|` and gross — are
walls (#135, table above) and are non-negotiable; this convention is the *soft* judgment that keeps you
well inside them. Per-sector and net-exposure caps are **not** walled today — they are yours to honor,
and candidates for the #141 layer.

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

## Not a layer yet

These conventions are **per-strategy prose** today, applied by judgment inside each `compute_weights`.
They are **not** a composable, enforceable portfolio-construction layer. #141 — *formalize portfolio
construction as a distinct layer* — is where several of them (inverse-vol / vol-targeting, per-sector and
net caps, previous-weight turnover banding) move from judgment into code. Until then, for the soft
versions, **judgment is the only guard.**

## Related principles

- [[research-methodology]] — the anti-leakage / generalization sibling: the leakage vectors no wall
  catches, and how to read results honestly.
````

> **Note for the executor:** in the actual file the single-name / gross rows use a literal pipe in
> `|weight|`; write it as `\|weight\|` *inside the table cell* (escaped, as shown) so the Markdown table
> parses, but plain `|weight|` in the prose paragraph under "Exposure caps". The fenced block above is
> the file content verbatim.

- [ ] **Step 2: Verify the file exists and renders**

Run: `ls -la kb/principles/risk-conventions.md && head -5 kb/principles/risk-conventions.md`
Expected: file present; title line `# Weight-space risk conventions — sizing and protection in allocation space`.

---

### Task 2: Update the cross-link in `research-methodology.md`

**Files:**
- Modify: `kb/principles/research-methodology.md` (the "Related principles" line, currently reserved)

- [ ] **Step 1: Replace the reserved line**

Find:
```markdown
- [[risk-conventions]] — weight-space risk conventions *(reserved for issue #136; not yet authored)*.
```
Replace with:
```markdown
- [[risk-conventions]] — weight-space risk conventions: sizing and protection in allocation space
  (inverse-vol sizing, drawdown-based weight decay, conviction sizing) — judgment above the #135 walls.
```

- [ ] **Step 2: Verify**

Run: `grep -n "risk-conventions" kb/principles/research-methodology.md`
Expected: the new line; no "not yet authored".

---

### Task 3: Reference the doc from `author-a-strategy`

**Files:**
- Modify: `.codex/skills/author-a-strategy/SKILL.md` (the "Read the methodology before authoring" bullet, ~lines 54-57). `.claude/skills/author-a-strategy` is a symlink to this — one edit covers both.

- [ ] **Step 1: Replace the bullet**

Find:
```markdown
- **Read the methodology before authoring.** `kb/principles/research-methodology.md` covers the
  leakage vectors no wall catches — full-sample fitting, target leakage inside a custom feature,
  `adj_close`/provenance leaks, and the `compute_weights_panel` parity-vs-validity trap. The rules
  here are the floor, not the whole job.
```
Replace with:
```markdown
- **Read the methodology AND the risk conventions before authoring.**
  `kb/principles/research-methodology.md` covers the leakage vectors no wall catches — full-sample
  fitting, target leakage inside a custom feature, `adj_close`/provenance leaks, and the
  `compute_weights_panel` parity-vs-validity trap. `kb/principles/risk-conventions.md` covers
  weight-space risk — inverse-vol sizing, drawdown-based weight decay, conviction sizing, and the
  "R:R is the wrong yardstick" point — the judgment that sits above the #135 walls. The rules here are
  the floor, not the whole job.
```

- [ ] **Step 2: Verify the edit propagated through the symlink**

Run: `grep -n "risk-conventions" .claude/skills/author-a-strategy/SKILL.md`
Expected: the new reference is visible via the `.claude` symlink (proves one edit covers both).

---

### Task 4: Verification + commit

**Files:** none (verification only)

- [ ] **Step 1: Quality gate (no code changed — must stay green)**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green (docs-only change touches no Python).

- [ ] **Step 2: Vault link integrity**

Run: `grep -rn "\[\[risk-conventions\]\]\|\[\[research-methodology\]\]" kb/principles/`
Expected: `research-methodology.md` links to `[[risk-conventions]]`; `risk-conventions.md` links back to `[[research-methodology]]` — both targets exist (`kb/principles/{risk-conventions,research-methodology}.md`), so no dangling wikilink.

- [ ] **Step 3: doctor still healthy**

Run: `uv run algua doctor`
Expected: non-error JSON; `knowledge_base` check unaffected (it tracks `strategies/`, not `principles/`).

- [ ] **Step 4: Commit**

```bash
git add kb/principles/risk-conventions.md kb/principles/research-methodology.md \
        .codex/skills/author-a-strategy/SKILL.md \
        docs/superpowers/specs/2026-06-09-weight-space-risk-conventions-issue-136-design.md \
        docs/superpowers/plans/2026-06-09-weight-space-risk-conventions-issue-136.md
git commit -m "docs(136): weight-space risk conventions kb doc + author-a-strategy reference"
```

---

## Self-review

- **Spec coverage:** framing/no-SL-TP ✔ (Task 1 intro); three governing rules ✔; walls-are-the-floor table with accurate `path::symbol` ✔; `check_drawdown` distinction ✔; six conventions (conviction, inverse-vol, drawdown-decay, signal-invalidation, turnover-prose, exposure-caps) ✔; reject-R:R yardstick ✔; not-a-layer-yet + #141 forward-pointer ✔; cross-links updated ✔ (Task 2); skill reference ✔ (Task 3); quality gate ✔ (Task 4).
- **Placeholder scan:** none — full doc content embedded verbatim.
- **Consistency:** convention named "Inverse-volatility sizing" throughout; `decay_full_at` (not `max_dd`) in the snippet; turnover is prose (no `prev_w`); exec-lag cites `engine.py::simulate`. Matches the GATE-1-approved spec.
