# #529 — Recalibrate the `backtested→candidate` promotion gate (triple-counted multiplicity)

Status: DESIGN, **round 4b — GATE-1 (Codex) PASS** (2 non-blocking GATE-2 watchpoints folded in: state
the standard LORD++ null-p-value/predictability assumption behind `p_cohort`; word the throttle as a
per-decision cap on PRIOR committed in-window rows, not a retrospective-window property). **Resolves
the round-3 GATE-1 BLOCK by making the near-term exposure budget an AGENT-ENFORCED CAP** (Codex
recommendation path (a)): the schema-free windowed throttle that
round 3 sketched and declined in §3.5 is now ADOPTED and REAL as a **promotion-eligibility cap**
(§3.5). The 5% near-term cumulative false-discovery budget derived in §3.1 is no longer an operational
hope backed only by human sign-off — it is enforced in code by a hard cap on the number of binding
FDR tests that are PROMOTION-ELIGIBLE per rolling window (at most `FDR_NEAR_TERM_BINDING_BUDGET = 16`;
the 17th+ still commit a fail-closed binding row and still advance the LORD++ stream but CANNOT
promote), computed inside the existing `BEGIN IMMEDIATE` (no schema change, no new lock), with a
HUMAN-ONLY #329-signed bypass as the escape hatch. Because only the first 16 in-window binding tests
can promote, and those 16 CONSECUTIVE binding tests touch at most 3 always-fresh-W0 cohorts (#324),
the windowed all-null FALSE-PROMOTION probability is bounded — for ANY rolling window, under actual
LORD++ wealth dynamics — by `3·p_cohort ≈ 7.2%` (union bound dominating each windowed slice by its
full fresh cohort; typical cohort-aligned case ≈ 4.9%), and equivalently the exposure RATE is capped
at ≤16 promotion-eligible tests / 365d — bounded by code, not hoped. Round 3 was BLOCKED because it
presented §3.1–3.2 as having CLOSED the round-2 cumulative-exposure finding when it had only
QUANTIFIED-AND-ACCEPTED it. The first round-4 GATE-1 pass BLOCKED a draft that claimed to cap "binding
tests that COMMIT" (false — the 17th still commits); the second BLOCKED an "EXACT S(16)=4.9% per
rolling window" claim (false — a rolling window does not align to cohort boundaries and can inherit
pre-window discoveries). This revision reframes the cap as promotion-eligibility (Codex option C) and
replaces the exact claim with the RIGOROUS worst-case `≤3·p_cohort` over any window (Codex option 2),
scoping every "invariant" as an agent-enforced cap with a signed human bypass. Prior rounds: round 1 (α-floor) BLOCKED; round 2 (N=4, W0=α, "land α in [0.01,0.05]") BLOCKED
(unbudgeted cumulative exposure); round 3 (budget-derived N=8, reverted W0, quantified retry surface,
honest rescope framing) BLOCKED (budget not enforced). Round 4 keeps every round-3 gain (γ-fix,
W0=α/2, N=8 budget derivation, honest rescope) and adds the enforcing promotion-eligibility throttle.
CODEOWNERS-protected surface (`algua/research/gates.py`, `algua/research/fdr_lord.py`,
`algua/registry/promotion.py`, `algua/registry/store.py`) — PR stays OPEN for human merge.

## 1. Problem (unchanged from the issue)

The `backtested→candidate` gate hard-ANDs three overlapping multiplicity corrections:

- **(a)** breadth-deflated holdout-Sharpe bar (`effective_min_holdout_sharpe`, haircut by
  `n_funnel`) — in `evaluate_gate`. Corrects for the strategy's OWN measured search breadth.
- **(b)** DSR confidence ≥ 1−`DSR_ALPHA` (= 0.95) — in `evaluate_gate` (`passed_dsr`). Per-strategy
  selection-inflation-corrected confidence that `SR_true > SR*`.
- **(c)** LORD++ per-cohort FDR AND-check: `p = 1 − dsr_confidence ≤ α_t` — in
  `store.record_gate_with_fdr_and_maybe_promote` (`fdr_rejected`, folded into `final_passed`). The
  ACROSS-STRATEGY funnel-wide false-discovery-rate control.

The three are NOT redundant — they correct three DIFFERENT multiplicities (own-search, own-selection
-vs-null, across-strategy-funnel). The bug is narrowly in **(c)**: its per-test α is Bonferroni-tiny
(current code: `α_1 ≈ 0.00165`, `α_2 ≈ 0.00082`, `α_64 ≈ 4.6e-5` — a ~99.8%-individual-confidence
bar), STRICTLY stronger than (b)'s 0.95, so (c) silently raises the effective DSR bar from 0.95 to
~0.998 for every strategy. The first real verdict (`liquid10_adj_momentum`: holdout Sharpe 1.06 vs
deflated bar 2.065, dsr 0.902, fdr p 0.098 vs α_t ≈ 0.00082) fails (a) AND (b) AND (c) — but for a
GENUINELY strong strategy that clears (a) and (b), (c) at α ≈ 0.001 is the sole remaining wall, and
it is effectively unpassable.

## 2. Root cause of the tiny α (two genuine mis-calibrations, both inside valid LORD++)

The dry-spell (no in-cohort discoveries) level is `α_t = γ_t · W0`. Two parameters make it tiny:

1. **γ normalized over the wrong horizon (a genuine bug).** Cohorts restart every
   `FDR_COHORT_SIZE` binding tests (#324), so a cohort is a self-contained LORD++ stream of length
   ≤ `FDR_COHORT_SIZE`. But `γ` is normalized over `FDR_GAMMA_TRUNCATION = 10 000` terms. A cohort
   therefore only ever spends the first-`FDR_COHORT_SIZE` fraction of the γ-mass — at N=64 the
   cohort spends `Σ_{j≤64} γ_j = 0.449` of `W0`, wasting HALF its budget. The restart horizon and
   the normalization horizon are mismatched. Fixing it (normalize γ over the restart horizon)
   alone lifts `α_1` from 0.00165 → 0.00366 at N=64 — still tiny, but the honest first step.
2. **`FDR_COHORT_SIZE = 64` spreads even the full budget across 64 tests**, so the average per-test
   α is `≤ 0.025/64 ≈ 0.0004`. 64 was calibrated for a HIGH-throughput funnel; the real funnel is
   low-throughput. Recalibrating N is legitimate — BUT (this is the round-2 lesson) N is not free:
   smaller N restarts the FDR budget more often, so it trades per-test power against **cumulative
   lifetime false-discovery exposure**, which MUST be budgeted, not ignored.

`FDR_W0 = FDR_ALPHA/2 = 0.025` is the standard Ramdas et al. 2017 choice and is retained (round 2's
bump to `W0 = FDR_ALPHA` is REVERTED — see §3.3).

## 3. Decision (round 4): budget-derive the cohort size AND enforce the budget

Keep all three checks HARD. Recalibrate (c)'s LORD++ **parameters** and add a hard windowed throttle
that turns the derived budget into an agent-enforced cap — four moves:

1. **Fix γ-normalization** — normalize γ over `FDR_COHORT_SIZE` terms (the restart horizon), not
   10 000. Each independent cohort is a length-N LORD++ stream; its γ must sum to 1 over its own N
   terms. Guarantees `Σ_{t≤N} α_t^dry = W0`. Uncontested (round-2 GATE-1 accepted this).
2. **`FDR_W0 = FDR_ALPHA/2 = 0.025`** — UNCHANGED from the codebase (round 2's `W0 = FDR_ALPHA`
   doubling is reverted, §3.3).
3. **`FDR_COHORT_SIZE = 8`** (was 64) — **derived from an explicit cumulative-exposure budget**
   (§3.1), NOT from "lands α in the issue's band." The issue's [0.01, 0.05] band is explicitly
   REJECTED as the target because no N that reaches it keeps the retry-attack surface acceptable
   (§3.1, §3.2).
4. **A hard windowed PROMOTION-ELIGIBILITY throttle that ENFORCES the §3.1 budget** (§3.5) — the
   missing piece round 3 declined. It bounds the number of binding FDR tests that are PROMOTION-
   ELIGIBLE per rolling window to `FDR_NEAR_TERM_BINDING_BUDGET = 16 = H_near` (the 17th+ still commit
   a fail-closed binding row and still advance the LORD++ stream, but cannot promote), so the windowed
   all-null FALSE-PROMOTION probability is bounded, for ANY rolling window under actual LORD++
   dynamics, by `≤ 3·p_cohort ≈ 7.2%` (typical cohort-aligned case ≈ 4.9%; §3.5), i.e. the exposure
   RATE is capped — enforced in code for agents, not a hope. Human-only #329-signed bypass is the
   escape hatch.

### 3.1 Deriving N from a cumulative-exposure budget (round-2 GATE-1 finding 1)

**The per-cohort `Σα_t = W0` identity is NOT a safety proof.** Each cohort restart is a *policy
rescope*: a new, independent hypothesis family with a fresh `W0` budget (this rescope was introduced
by #324, not by #529 — see §3.4). What #529 changes is the cohort FREQUENCY, i.e. how MANY
independent 5%-budget families a lifetime of attempts is chopped into. The honest safety question is
therefore the **cumulative all-null false-discovery probability over a realistic number of binding
attempts**, `S(H, N) = 1 − (1 − p_cohort)^{H/N}`, where `p_cohort = 1 − Π_{t≤N}(1 − α_t^dry)` ≈ W0.

**Budget.** Adopt an explicit, human-signable, AND CODE-ENFORCED exposure budget instead of the band:

- **Near-term realistic horizon `H_near = 16` binding tests** (≈ one year at this low-throughput
  funnel). Justification: (c) binds on every MEASURED run carrying a DSR p-value, and each such run
  costs a FRESH single-use OOS holdout interval (the holdout burn, #193/#205) — non-overlapping OOS
  history is physically scarce, so ~16 binding tests/yr is a generous low-throughput estimate; the
  adjacent `AGENT_NOVEL_MINT_CAP ≈ 8 mints / 90 days` (#524) corroborates the order of magnitude.
- **Budget target: near-term cumulative all-null FD `S(16, N) ≤ 0.05`** — one cohort-budget's worth
  (≈ 2× the #324 per-cohort baseline of 2.5%), an EXPLICIT, signed relaxation, not a hidden one.
- **`H_near` is not just the derivation horizon — it is the ENFORCED cap.** Round 3 treated `H_near`
  as an assumption about how the funnel *would* behave; round 4 makes the system *guarantee* it via
  the throttle (§3.5). The budget arithmetic below is therefore load-bearing on a bound the code
  holds, not on operator behaviour.

**Solve.** With `W0 = 0.025`, γ normalized over N (measured, this worktree):

| N | α_1 | α_2 | α_N(dry) | p_cohort | S(16) | S(32) | S(64) | S(128) |
|---|-----|-----|----------|----------|-------|-------|-------|--------|
| 4 | 0.0110 | 0.0055 | 0.0039 | 0.0247 | 0.096 | 0.182 | 0.331 | 0.552 |
| 6 | 0.0088 | 0.0044 | 0.0023 | 0.0247 | 0.065 | 0.125 | 0.235 | 0.414 |
| **8** | **0.00764** | **0.00382** | **0.00156** | **0.0247** | **0.049** | **0.095** | **0.182** | **0.330** |
| 12 | 0.0064 | 0.0032 | 0.0009 | 0.0247 | 0.033 | 0.065 | 0.125 | 0.234 |
| 16 | 0.0057 | 0.0028 | 0.0006 | 0.0247 | 0.025 | 0.049 | 0.095 | 0.181 |
| 64 (baseline) | 0.0037 | 0.0018 | 0.0001 | 0.0247 | 0.006 | 0.012 | 0.025 | 0.049 |

`N = 8` is the SMALLEST cohort (⇒ largest per-test α ⇒ most passable) whose near-term surface
`S(16) = 0.049` stays within the 5% budget. It lifts `α_1` from the current **0.00165 → 0.00764**
(a **4.6×** improvement; the effective DSR bar drops from 0.99835 → 0.99236, i.e. an annualized
Sharpe of ≈1.81 on 452 obs instead of ≈2.1), which is a materially more passable first-test bar for
a genuinely strong strategy, WITHOUT chasing the unsafe band. With the throttle (§3.5) capping
promotion-eligible binding tests at `H_near = 16` per window (≤3 always-fresh cohorts touched), the
ENFORCED per-window worst-case false-promotion probability is `≤ 3·p_cohort ≈ 7.2%` (typical aligned
`S(16) = 4.9%`) — the `S(32)/S(64)/S(128)` columns are the exposure the throttle now PREVENTS from
accumulating within any one window, not an accepted operating point.

### 3.2 Retry-attack surface at N=8 — now BOUNDED by the throttle (round-2 GATE-1 finding 3)

The concrete all-null probability that a determined autonomous "retry until (c) passes" loop lands
≥1 false PROMOTION, for `H` promotion-eligible binding attempts — the number round 3 quantified and
accepted, and round 4 now CAPS at the `H = H_near = 16` column via the throttle:

```
N=8, W0=0.025:   S(16)=4.9%   S(32)=9.5%   S(64)=18.2%   S(128)=33.0%
baseline N=64:   S(16)=0.6%   S(32)=1.2%   S(64)= 2.5%   S(128)= 4.9%
                 └── ENFORCED  └── per-window unreachable for PROMOTIONS (throttle) ────┘
```

The throttle (§3.5) makes at most 16 binding tests PROMOTION-ELIGIBLE per rolling window; those 16
consecutive tests touch ≤3 always-fresh cohorts, so the attacker's realized false-promotion surface
over ANY window is ≤ `3·p_cohort ≈ 7.2%` (typical aligned case ≈ 4.9% = the `S(16)` cell) — the
`S(32..128)` rows are per-window unreachable for promotions. Across windows the exposure is RATE-CAPPED
at ≤16 promotion-eligible tests per 365-day window (§3.4), not uncapped. This is the structural fix
Codex asked for: the safety property is now "windowed all-null false-PROMOTION probability ≤ 7.2% for
agents (rigorous worst-case)", enforced in code, rather than "≤ 4.9% only if the attempt horizon stays
near 16", hoped. (A "false discovery" in this real-money system IS a null strategy PROMOTED to
`candidate`; bounding promotions, not committed binding rows, is the precise safety statement — §3.5.
The `S(H,N)` table rows are cohort-ALIGNED idealizations; the enforced object is the ≤3·p_cohort
worst-case + the rate cap.)

- **The residual is explicitly accepted with human sign-off.** N ∈ {6, 8, 12}, the throttle window
  (`FDR_THROTTLE_WINDOW_DAYS = 365`) and cap (`FDR_NEAR_TERM_BINDING_BUDGET = 16`), the agent-ENFORCED
  worst-case per-window false-promotion surface `≤ 3·p_cohort ≈ 7.2%` (typical aligned ≈ 4.9%; vs
  baseline 0.6%), and the existence of the human-only signed bypass are a CODEOWNERS, real-money
  calibration call; the PR body MUST record human sign-off on all of them with THIS table in view
  (round-3 GATE-1 nit 1 + round-2 blocking finding 3 + round-1 nit 5). What the human accepts for agent
  operation is now BOUNDED (a per-window ≤7.2% worst-case false-promotion ceiling + a ≤16/window rate
  cap the code holds for agents), not the round-3 unbounded-in-count posture; each human bypass is a
  separate, individually-signed acceptance of one marginal promotion.
- **(c) + throttle is the near-term defense; the external controls are now defense-in-depth.** The
  retry loop is additionally bounded by controls that scale with attempt count: (i) the **single-use
  holdout burn** — 16 binding attempts require 16 distinct non-overlapping OOS intervals; (ii)
  **accumulating breadth-deflation of (a)** — every attempt raises `windowed_total_combos`,
  monotonically LIFTING the (a) Sharpe bar; (iii) **accumulating DSR trial variance** on (b); (iv)
  `AGENT_NOVEL_MINT_CAP` (≈8 mints/90d) on the #524 family-mint path. Round 3 leaned on these as the
  SOLE lifetime bound (Codex objection 2); round 4 carries the near-term bound in the throttle and
  demotes (i)–(iv) to belt-and-suspenders, so a soft/path-specific assumption no longer voids the
  near-term guarantee.
- **N=8 is 4× LESS exposed than the BLOCKED round 2, and now capped.** Round 2 (N=4, W0=α) was 16×
  more cohorts × 2× budget = **32×** baseline rate. Round 4 (N=8, W0=α/2) is 8× more cohorts × 1×
  budget = **8×** the rate, but the throttle bounds the WINDOWED realization of that rate to ≤ S(16).

**External-control assumptions (now defense-in-depth, still worth confirming at sign-off).** These
NO LONGER carry the near-term budget (the throttle does), so a failure of one degrades the depth of
defense rather than voiding the guarantee:

1. **Holdout supply is finite, single-use, and non-overlapping** (#193 peek/burn, #205 identity).
   VERIFY the burn is per-interval on the `research promote` path.
2. **Breadth-deflation accumulates across retries** in the relevant family/window — `windowed_total
   _combos` (and family-lifetime breadth, #222/#524) monotonically rises and feeds (a)'s bar.
3. **`AGENT_NOVEL_MINT_CAP` bounds ONLY the #524 family-MINT path**, not promotion into existing
   families — it is not a general promotion throttle (the §3.5 throttle now is).
4. **DSR trial variance is a soft statistical penalty on (b), not a hard retry throttle.**

**Why not the issue's [0.01, 0.05] band?** Reaching `α_1 ≥ 0.01` needs N ≤ 4 (W0=0.025), whose
S(64) ≥ 33% — but more to the point, the band was never a safety target; the throttle + budget are.

### 3.3 Why W0 stays at FDR_ALPHA/2 (reverting a round-2 change)

Round 2 set `W0 = FDR_ALPHA = 0.05` to "spend the full budget for early power." That DOUBLES
`p_cohort` (2.5% → 5%) and hence DOUBLES the per-window surface the throttle must bound, for a
marginal early-α gain — the wrong trade. Round 4 keeps the standard `W0 = FDR_ALPHA/2 = 0.025`
(Ramdas et al. 2017); the passability improvement comes from the budget-derived N and the γ-fix.
`w0 ≤ α` remains satisfied.

### 3.4 The per-cohort rescope, argued honestly + now rate-capped (round-2 GATE-1 finding 2)

**Cohort restarts are a policy rescope, not a formal lifetime guarantee — and #529 inherits, not
introduces, that rescope.** #324 already re-scoped FDR from lifetime to per-cohort (the
`FDR_COHORT_SIZE` docstring: "FDR is controlled PER COHORT … NOT per lifetime. Cumulative exposure
over K completed cohorts is bounded by FDR_ALPHA·K"). The reason was intrinsic: any valid *lifetime*
online-FDR target on a garbage-dominated funnel drives the per-test level to 0 (anti-scaling) — a dry
spell of flops lowers everyone's bar. The rescope trades an unattainable lifetime guarantee for a
per-cohort one plus a throughput-independent floor.

Round 3 acknowledged the GLOBAL exposure `≈ FDR_ALPHA · (total binding tests / N)` grows without
bound in attempt count and merely argued it "acceptable." **Round 4 does not remove the unbounded-
in-count lifetime posture (that would require the impossible lifetime guarantee), but it RATE-CAPS
it:** the §3.5 throttle bounds binding tests to ≤ `H_near` per rolling window, so lifetime exposure
now accrues at a BOUNDED RATE (≤ one near-term budget per window over calendar time) rather than the
uncapped rate an unthrottled retry loop could realize. We therefore claim, precisely:

1. **Near-term / per-window exposure is an agent-enforced bounded cap** — worst-case false-promotion
   probability `≤ 3·p_cohort ≈ 7.2%` per throttle window (typical aligned 4.9%) and a ≤16-eligible/
   window rate cap, held by code (§3.5), not by sign-off (a human bypass is a separate signed
   exception).
2. **Lifetime exposure remains unbounded-in-count but is now rate-limited** by the throttle to ≤16
   promotion-eligible tests per calendar window; it is no longer realizable at burst speed.
3. **A genuine lifetime FDR guarantee is provably incompatible with a passable gate** on this funnel
   (anti-scaling) — which is why #324 rescoped and why we rate-cap rather than promise it.

This is a rescope-of-a-rescope, argued on its merits AND now with an enforced near-term floor under
it — not a `Σα_t = W0` identity dressed up as a proof, and not (round 3's error) a quantified budget
presented as if enforced.

### 3.5 The hard windowed PROMOTION-ELIGIBILITY throttle (ADOPTED)

Round 3 sketched a schema-free windowed throttle and DECLINED it (fail-closed wedge). Round-3 GATE-1
BLOCKED on exactly that decline: the near-term budget was quantified and accepted, not enforced.
Round 4 adopts the throttle. It is precisely a **promotion-eligibility cap** — NOT a cap on committed
binding rows (the first round-4 GATE-1 draft claimed the latter, which is false: throttled rows still
commit). The objections are re-answered below.

**Mechanism (schema-free, inside the existing critical section).** In
`record_gate_with_fdr_and_maybe_promote`, inside the SAME `BEGIN IMMEDIATE` that already reads the
FDR stream and does the stage CAS, before folding the FDR verdict into `final_passed`, count the
funnel-wide binding gate rows already committed in the trailing throttle window and trip PROMOTION
(not the row insert) when the budget is already spent:

```
throttle_window_binding = COUNT(*) FROM gate_evaluations
    WHERE fdr_binding = 1 AND created_at >= now − FDR_THROTTLE_WINDOW_DAYS   -- PRIOR rows only
promotion_eligible = (throttle_window_binding < FDR_NEAR_TERM_BINDING_BUDGET) or human_override
throttle_tripped   = fdr_binding and not promotion_eligible
final_passed       = provisional_passed and fdr_rejected and promotion_eligible
```

with two new CODEOWNERS-protected constants in `fdr_lord.py`:

- `FDR_NEAR_TERM_BINDING_BUDGET = 16` (= `H_near`; the per-window promotion-eligibility cap), and
- `FDR_THROTTLE_WINDOW_DAYS = 365` (the rolling window over which `H_near` was estimated, §3.1).

**A throttled test STILL commits its binding row and STILL advances the LORD++ stream** (it performed
a real FDR test — a holdout was burned at peek, a p-value computed); it only loses promotion
eligibility (`final_passed=False`). Prior binding rows count regardless of pass/fail: every binding
test spent α, so all of them count toward the promotion-eligibility budget (conservative). The count
is a pure SELECT over existing columns (`fdr_binding`, `created_at`) — **no schema change, no new
lock**. Adopting Codex's option (C): this is explicitly a promotion throttle, and the bound below is
proven over PROMOTIONS, so throttled rows remaining in the LORD++ stream is BY DESIGN and does not
perturb the arithmetic (they are tests 17+, after the promotion-eligible window; see the proof).

**Why the windowed false-promotion probability is RIGOROUSLY BOUNDED — and why it is NOT the exact
`S(16)` (answers findings 1, 2 & 4, and the round-4b window/cohort-misalignment BLOCK).** A "false
discovery" in the real-money sense is a null strategy PROMOTED to `candidate`. A promotion needs (a)
AND (b) AND (c) AND promotion-eligibility; adversarially GRANT the attacker (a) and (b), so a false
promotion ⟺ `fdr_rejected=True` on a promotion-eligible binding test. Only the FIRST 16 in-window
binding tests are promotion-eligible (the count is of PRIOR in-window rows, so tests at positions
1..16 see count 0..15 <16 and are eligible; positions 17+ see count ≥16 and are throttled). The bound
is built on the ONE thing that is exact under the real LORD++ wealth dynamics:

- **Per fresh cohort, `P(≥1 false discovery) = p_cohort` EXACTLY.** Every cohort of N=8 starts with
  fresh `W0` and no inherited discoveries (#324). Up to its FIRST rejection there are no in-cohort
  discoveries, so its level is exactly the dry-spell `α_t^dry = γ_t·W0`; post-discovery reward
  inflation is irrelevant to the ≥1 event (once the first fires, the event is satisfied). So
  `p_cohort = 1 − Π_{t≤8}(1−α_t^dry) ≈ 2.47%`, computed from the ACTUAL `lord_plus_plus_level(t, [])`
  (test §7.3/§7.4 pins the full dry-spell vector against the real level function — Codex finding 2).
- **A rolling wall-clock window does NOT align to cohort boundaries** (the round-4b BLOCK): its first
  eligible test can be mid-cohort, and a discovery in that cohort BEFORE the window elevates levels
  for the in-window tail. So `1−(1−p_cohort)^2 = S(16)=4.9%` is the COHORT-ALIGNED IDEALIZED case, NOT
  a valid bound for an arbitrary window. We do not claim it as enforced.
- **Rigorous worst-case for ANY rolling window (dominance, not direct computation).** The 16
  promotion-eligible tests are 16 CONSECUTIVE binding tests in arrival order (a wall-clock window's
  binding tests are contiguous in the id-ordered stream; the eligible ones are its first 16). 16
  consecutive tests on period-8 cohorts touch AT MOST `⌈16/8⌉+1 = 3` cohorts. For each touched
  cohort k, the event "≥1 false discovery among cohort k's IN-WINDOW tests" ⊆ "≥1 false discovery
  among ALL of cohort k's tests", whose probability is exactly `p_cohort` (fresh cohort, above) —
  INCLUDING any pre-window discovery in k, which is already part of the full-cohort event. Dominating
  the windowed subset by the full fresh cohort is what makes this immune to wealth dynamics and
  misalignment. Union bound over ≤3 touched fresh cohorts:
  `P(≥1 false promotion / ANY rolling window) ≤ 3·p_cohort ≈ 7.2%`. This is the ENFORCED, agent-level,
  worst-case guarantee (the cohort-aligned typical case is the smaller 4.9%).
- **Throttled tests (17+) are non-eligible and cannot promote**, so they never add a false promotion;
  keeping them in the LORD++ stream is BY DESIGN (option C) and does not affect the ≤3·p_cohort bound
  (Codex finding 4).

This is the structural fix, honestly bounded: capping promotion-eligibility at 16 caps the windowed
worst-case false-promotion probability at `≤3·p_cohort ≈ 7.2%` (typical/aligned 4.9%), for agents, in
code — a valid bound over ANY rolling window under actual LORD++ dynamics (Codex round-4b option 2),
NOT an overclaimed exact 4.9%. It closes Codex round-3 objection 1 (unenforced) and objection 4
(sign-off ≠ guardrail), and downgrades objection 2 (external bounds) to defense-in-depth (§3.2).

**Preserves `fdr_rejected` meaning (round-1 finding 5).** `fdr_rejected = p ≤ α_t` stays the pure
LORD++ FDR verdict — the throttle does NOT overload it (a throttled row keeps its true `fdr_rejected`,
merely `final_passed=False`). The throttle is a SEPARATE, explicitly-named `final_passed` term
surfaced as its own `checks[]` entry (`fdr_throttle`, op `<`, value `throttle_window_binding`,
threshold `FDR_NEAR_TERM_BINDING_BUDGET`, `passed = promotion_eligible`), plus `fdr_throttle_tripped`
/ `fdr_throttle_window_binding` / `fdr_throttle_override` in `raw_decision`.

**Agent-enforced cap with a signed human bypass — NOT an unconditional invariant (answers finding 3).**
The cap is unconditional FOR AGENTS: an agent can never promote a 17th+ in-window binding test. It is
conditional overall: a human with a #329 signature can bypass it via `--fdr-throttle-override` on
`research promote`, REJECTED on the agent path exactly like `--allow-non-pit`/`--allow-holdout-reuse`.
The correct claim is therefore "agent-enforced ≤16/window promotion cap with an individually-signed
human bypass", NOT "unconditional system invariant". Each bypass is one signed, audited acceptance of
one marginal promotion (the signature binds the run, so it cannot be replayed). A genuinely productive
year of >16 promotions wedges agent promotions until older tests age out of the 365-day window — a
rare, human-recoverable false-negative at the estimated ~16 tests/yr throughput (§3.1); strictly
better than round 3's "no bound, just sign off on the number".

**No new TOCTOU (round-2's throttle objection).** The count runs inside the SAME write-locked
`BEGIN IMMEDIATE` that already serializes binding evaluations (the stream read + stage CAS live
there). Two concurrent binding tests cannot both read `throttle_window_binding = 15` and both promote
as the 16th/17th — the second re-reads 16 after the first commits. Round 2 rejected a throttle partly
on a "new DB field / TOCTOU surface" concern; this throttle adds neither.

**What it does and does NOT guarantee (honest scope).** It ENFORCES, for agents, a bounded windowed
worst-case: all-null false-PROMOTION probability ≤ `3·p_cohort ≈ 7.2%` per rolling year (typical
cohort-aligned case ≈ 4.9%), AND — equivalently and more simply — an operational RATE cap of ≤16
promotion-eligible binding tests / 365d, which is precisely what the round-2/round-3 finding demanded
(the cohort-restart FREQUENCY, hence the exposure RATE, is no longer unbounded). It does NOT
manufacture a lifetime FDR guarantee (provably incompatible with a passable gate, §3.4) — it rate-caps
lifetime exposure instead of promising it — and it does NOT claim an EXACT per-window probability
(window/cohort misalignment + wealth dynamics preclude that; only the ≤3·p_cohort worst-case is
claimed). Round 4 structurally fixes the NEAR-TERM slice of the round-2 finding (agent-enforced rate
cap + bounded worst-case) and rate-limits the lifetime slice; that lifetime residual is named as an
explicit CODEOWNERS-signed risk acceptance (§9).

**Two watchpoints folded in for GATE-2 (round-4b GATE-1 nits, non-blocking).** (1) The `p_cohort`
exactness rests on the STANDARD LORD++ assumptions — null p-values (super-)uniform and the LORD++
predictability/independence conditions; the `fdr_lord.py` docstring and test §7.4 MUST state this
assumption explicitly (it is the same assumption the pre-existing per-cohort FDR guarantee already
relies on, not a new one). (2) The throttle enforces promotion-eligibility AT EACH EVALUATION TIME
using the count of PRIOR committed binding rows in the trailing window — it is a per-decision
promotion cap, not a retrospective property of an arbitrary window; a retrospectively-chosen window
may contain shifted/fewer eligible rows, which does NOT weaken the per-decision cap. Doc/comment
wording must say "prior committed in-window rows at decision time", not "any window".

## 4. Audit exposure — surface partial-cohort spend + throttle state (round-2 NON-blocking note)

`fdr_expected_false_discoveries = FDR_ALPHA · cohorts_completed` counts only COMPLETED cohorts, so at
small N an in-progress cohort (up to N−1 binding tests that have already SPENT α) reads as 0 exposure
until it closes — the note Codex raised. Round 4 adds, to the audit-only block (no gate effect beyond
the throttle term itself, which IS surfaced in `checks[]`):

- `fdr_active_cohort_position` — the within-cohort ordinal of the current (in-progress) cohort
  (1..N), already available as `within_cohort_t`.
- `fdr_active_cohort_applied_alpha` — `Σ` of the stored `fdr_alpha_level` over the binding rows of
  the current in-progress cohort INCLUDING this row (the α actually spent so far in the open cohort).
- `fdr_expected_false_discoveries_incl_active = FDR_ALPHA · cohorts_completed +
  fdr_active_cohort_applied_alpha` — the completed-plus-active exposure, so partial spend at small N
  is never hidden.
- `fdr_throttle_window_binding` / `fdr_throttle_tripped` / `fdr_throttle_override` — the windowed
  binding-test count, whether the throttle fired, and whether a human override lifted it, so the
  enforced-invariant state is fully auditable per row.

The existing `fdr_expected_false_discoveries` (completed-only) is UNCHANGED and remains the
conservative per-cohort-bound field; the new fields are additive audit context.

## 5. #524 NOVEL-family mint dependency (point 5)

The agent-NOVEL mint fires in `record_gate_with_fdr_and_maybe_promote` inside
`if final_passed: … if pending_novel_family is not None: self._mint_agent_novel_family(…)`, within
the SAME `BEGIN IMMEDIATE`, keyed on:

```
final_passed = provisional_passed AND fdr_rejected AND promotion_eligible   (binding case)
             = (a breadth-deflated Sharpe) AND (b DSR ≥ 0.95) AND (economic/stability floors)
               AND (c: p_value ≤ α_t, RECALIBRATED valid LORD++, α_1 ≈ 0.0076 at N=8)
               AND (throttle: < H_near PRIOR promotion-eligible binding tests this window, or
                    human #329-signed override)
```

**"Pass" for mint-seeding is unchanged in definition** — `final_passed`, all hard checks ANDed. Two
changes vs the current dead path: (c) becomes reachable (α_1 lifts 4.6×), and the throttle adds one
more AND term. Precisely:

- The mint keys on `final_passed`, NOT `provisional_passed`. A strategy clearing (a)+(b)+floors+(c)
  but tripping the throttle has `final_passed=False` → no promotion AND no mint, atomically. So the
  throttle STRENGTHENS the mint-path retry defense (a throttled retry loop can never seed a family)
  — the invariant "the mint can never seed a family for a strategy the gate rejected" holds literally
  for all three of the FDR bar, the DSR bar, and the throttle.
- No change to `_mint_agent_novel_family`, the pending-NOVEL CAS, `AGENT_NOVEL_MINT_CAP`, or the
  graph-fingerprint drift checks. #524 composes unchanged; #529 only alters which strategies reach
  the shared `final_passed=True` gate #524 hangs off (wider via (c), narrower under burst via the
  throttle).
- #524 is unmerged (tasks #63–66 pending) and touches the same method. The two changes are in
  disjoint regions of the one transaction (#529: the `α_t`/constants/γ + throttle + audit block;
  #524: the mint call after the stage-CAS) and do not conflict semantically. Second-to-merge rebases.

## 6. Files touched

- `algua/research/fdr_lord.py` — `FDR_COHORT_SIZE = 8` (was 64); `FDR_W0` UNCHANGED at
  `FDR_ALPHA / 2`; **ADD** `FDR_NEAR_TERM_BINDING_BUDGET = 16` and `FDR_THROTTLE_WINDOW_DAYS = 365`
  (with the §3.1/§3.5 derivation in their docstring); normalize `_compute_lord_gamma` / `_LORD_GAMMA`
  over `FDR_COHORT_SIZE` terms (a cohort never indexes past N). **REMOVE `FDR_GAMMA_TRUNCATION`** —
  vestigial once γ is normalized over the restart horizon; delete it and update its re-export in
  `gates.py` and its test in the SAME PR (round-1 nit 7). Rewrite the `FDR_COHORT_SIZE` rationale
  block to the round-4 story: the γ-normalization bug, the budget-derived N=8 (near-term S(16)≤5%
  budget, §3.1 table), the ENFORCING throttle (§3.5), the honest per-cohort rescope + rate-cap
  (§3.4), and the LORD++-assumptions/operating-target caveat. `lord_plus_plus_level` math unchanged.
- `algua/registry/store.py` — binding site: `alpha_t = level_fn(t_next, stream.discovery_indices)`
  already computes the recalibrated α_t; `fdr_rejected` UNCHANGED (stays pure LORD++). **ADD the
  throttle**: a helper (e.g. `_windowed_binding_test_count(window_days)`) that COUNTs `fdr_binding=1`
  rows with `created_at >= now − FDR_THROTTLE_WINDOW_DAYS`, called inside the existing
  `BEGIN IMMEDIATE`; compute `throttle_tripped` (respecting the human-override arg) and AND it into
  `final_passed`; write the `fdr_throttle` `checks[]` entry + `fdr_throttle_*` audit fields into
  `raw_decision`. **ADD the §4 audit fields**: `_read_fdr_stream` returns the current in-progress
  cohort's applied-α sum + position; write `fdr_active_cohort_position`,
  `fdr_active_cohort_applied_alpha`, `fdr_expected_false_discoveries_incl_active`. Accept a new
  `fdr_throttle_override: bool = False` parameter threaded from the promote path. No schema change
  (all in `decision_json`); `fdr_rejected`/`fdr_expected_false_discoveries` formulas UNCHANGED.
- `algua/registry/promotion.py` — thread the human-only `fdr_throttle_override` flag through to
  `record_gate_with_fdr_and_maybe_promote`; `level_fn = partial(lord_plus_plus_level, alpha=
  FDR_ALPHA, w0=FDR_W0)` picks up the constants automatically (W0 unchanged). Enforce that the
  override is HUMAN-ONLY: reject it on the agent path (mirror the existing `--allow-non-pit` /
  `--allow-holdout-reuse` human-only handling) and bind it to the #329 signed run.
- `algua/research/gates.py` — re-export `FDR_NEAR_TERM_BINDING_BUDGET` / `FDR_THROTTLE_WINDOW_DAYS`
  alongside `FDR_W0`/`FDR_COHORT_SIZE`; drop the `FDR_GAMMA_TRUNCATION` re-export.
- The `research promote` CLI wiring — add the human-only `--fdr-throttle-override` flag (rejected for
  `--actor agent`, requiring the #329 signature like the other human-only relaxations). Non-CODEOWNERS
  file; keep the flag plumbing minimal.
- `CLAUDE.md` — update the `research promote` LORD++ paragraph: cohort size recalibrated to the
  low-throughput domain (8) under an explicit near-term cumulative-exposure budget; γ normalized over
  the restart horizon; `W0` unchanged at `FDR_ALPHA/2`; the budget is ENFORCED by a hard windowed
  throttle (≤16 binding tests / 365d) with a human-only signed `--fdr-throttle-override`; FDR stays a
  per-cohort operating target at `FDR_ALPHA` with lifetime exposure now RATE-CAPPED (not unbounded-
  at-burst); per-test α now O(0.5–0.8%), ~4.6× the old first-test level.
- Tests — see §7.

## 7. Test plan (point 7 — update, do not delete, the FDR-binding coverage)

`tests/test_research_gates.py`, `tests/test_promotion.py`, `tests/test_registry_store.py`:

1. `test_fdr_constants` — `FDR_W0 == FDR_ALPHA/2` (0.025, UNCHANGED); `FDR_COHORT_SIZE == 8`;
   `FDR_NEAR_TERM_BINDING_BUDGET == 16`; `FDR_THROTTLE_WINDOW_DAYS == 365`.
2. `test_gamma_weights_sum_to_at_most_one` / `_are_positive` / `_are_eventually_decreasing` — retune
   to the N=8-term γ (Σ=1 over N, all positive, decreasing across the N terms). Keep the properties.
3. **NEW** `test_lord_cohort_spends_full_budget` — `Σ_{t=1..N} lord_plus_plus_level(t, []) ==
   pytest.approx(FDR_W0)` AND pin the exact levels `lord_plus_plus_level(1, []) ==
   pytest.approx(0.00764, abs=1e-4)` and `lord_plus_plus_level(2, []) == pytest.approx(0.00382,
   abs=1e-4)` (round-3 GATE-1 nit 2). **Pin the FULL dry-spell vector** `[lord_plus_plus_level(t, [])
   for t in 1..8]` against the §3.1 table row so the S(16) arithmetic is proven to use the ACTUAL
   level function, not table constants (round-4 GATE-1 finding 2).
4. **NEW / adversarial** `test_lord_all_null_first_discovery_probability` — compute `p_cohort` from
   the ACTUAL `lord_plus_plus_level(t, [])` (empty-discovery ⇒ the pre-first-discovery state that the
   ≥1-false-discovery event lives in): assert `1 − Π_{t≤8}(1−lord_plus_plus_level(t, [])) ≤
   FDR_ALPHA/2 + ε` (per-cohort ≈2.5%, NOT round-1's 72%). Documents WHY the dry-spell product is
   EXACT for P(≥1): levels are dry-spell up to the first discovery — and STATES the standard LORD++
   assumption it rests on (null p-values super-uniform + LORD++ predictability/independence; the same
   assumption the pre-existing per-cohort guarantee uses — round-4b GATE-1 watchpoint 1).
5. **NEW / budget-guard + worst-case bound** `test_lord_retry_surface_within_near_term_budget` —
   assert the cohort-ALIGNED idealized surface `1 − (1 − p_cohort)^{H_near/N} ≤ 0.05` at
   `H_near=FDR_NEAR_TERM_BINDING_BUDGET, N=8` (the typical case), AND — the load-bearing one — assert
   the RIGOROUS worst-case for ANY rolling window: `max_touched_cohorts = ceil(H_near/N)+1 == 3` and
   `1 − (1 − p_cohort)^{max_touched_cohorts} ≤ 0.08` (regression-pins the ≤3·p_cohort≈7.2% enforced
   ceiling so a future N/budget change can't silently blow it), AND `H_near ==
   FDR_NEAR_TERM_BINDING_BUDGET` (cap can't desync from the arithmetic).
6. **NEW / throttle promotion-eligibility** `test_fdr_throttle_blocks_promotion_beyond_budget` — after
   `FDR_NEAR_TERM_BINDING_BUDGET` binding rows land within `FDR_THROTTLE_WINDOW_DAYS`, the NEXT binding
   evaluation that would otherwise pass (a)+(b)+(c) has `final_passed=False`, `fdr_throttle_tripped=
   True`, NO stage advance, and (if pending-NOVEL) NO mint — BUT it STILL commits a `fdr_binding=1`
   gate row with its TRUE `fdr_rejected` (pure LORD++, still True) and STILL advances the LORD++ stream
   (assert `fdr_test_index`/cohort position advanced) — proving it is a PROMOTION throttle, not a
   binding-row cap (round-4 GATE-1 findings 1 & 4). A binding row whose `created_at` is OUTSIDE the
   window does NOT count toward the cap; the 16th (count 0..15) DOES promote and the 17th does not
   (off-by-one pin).
7. **NEW / throttle-override** `test_fdr_throttle_human_override_promotes` — the same over-budget
   state with `fdr_throttle_override=True` (human path) promotes; assert the override is REJECTED on
   the agent path (mirrors the `--allow-non-pit` agent-rejection test).
8. `test_lord_no_discoveries_alpha_decreasing` — unchanged property, new values.
9. `test_run_gate_fdr_binding_accept_promotes` (sharpe=7.0, p≈0) — still promotes (within budget);
   α_1 comment → 0.00764.
10. `test_run_gate_fdr_binding_reject_no_promotion` — REJECT case with `p≈0.02 > 0.00764` (genuine
    LORD++ rejection, not throttle); update the threshold comment.
11. **NEW** `test_run_gate_fdr_recalibration_promotes_previously_unpassable` — dsr≈0.995 (`p≈0.005`),
    clears (a)+(b), within budget: OLD α_1≈0.00165 rejected it; recalibrated α_1≈0.00764 accepts →
    promoted. The regression test that the fix works.
12. **NEW** `test_fdr_active_cohort_exposure_surfaced` — after k<N binding rows in an open cohort,
    `fdr_active_cohort_position == k`, `fdr_active_cohort_applied_alpha == Σ stored α_levels`,
    `fdr_expected_false_discoveries_incl_active == FDR_ALPHA·cohorts_completed + that sum`; and
    `fdr_expected_false_discoveries` (completed-only) UNCHANGED (§4).
13. `test_run_gate_declared_breadth_omits_fdr_entirely` / `_missing_dsr_stats_omits_fdr` /
    `_non_binding_decision_json_has_fdr_skip_reason` — unchanged (recalibration + throttle only affect
    binding). Confirm a NON-binding decision never trips the throttle (no `fdr_binding` row counted).
14. `tests/test_registry_store.py` FDR-exposure asserts — `fdr_expected_false_discoveries` formula
    stays `FDR_ALPHA · cohorts_completed`, `== 0.0` at 0 completed cohorts still holds.
15. `test_fdr_cohort_position` / cohort-boundary tests referencing 64 — retune constants to 8.
16. #524-composition assertion (after #524 lands, or stubbed): a pending-NOVEL strategy failing the
    recalibrated (c) OR tripping the throttle is NOT minted (`final_passed=False`); one clearing all
    hard checks within budget IS.

## 8. GATE-1 findings → resolutions

### Round-4b GATE-1 (second pass) — blocking finding, resolved in this revision

| # | Round-4b finding (Codex) | Resolution in this revision |
|---|--------------------------|-----------------------------|
| A | `S(16)=4.9%` is NOT exact for a ROLLING window: it does not align to cohort boundaries, so the window can start mid-cohort and inherit pre-window discoveries that elevate in-window levels — the "two fresh cohorts of 8" proof is invalid for an arbitrary window | §3.5 — the exact claim is WITHDRAWN. Replaced with a rigorous worst-case valid for ANY rolling window under actual wealth dynamics (Codex option 2): the 16 eligible tests are 16 CONSECUTIVE binding tests touching ≤3 always-fresh cohorts (#324); each windowed slice is DOMINATED by its full fresh cohort (`P=p_cohort` exact, inheriting-discovery-immune); union bound ⇒ `P(≥1 false promotion / any window) ≤ 3·p_cohort ≈ 7.2%`. Equivalently an enforced ≤16-eligible/window RATE cap — which is exactly what the round-2/3 finding demanded |

### Round-4 GATE-1 (first pass on the throttle draft) — blocking findings, resolved

| # | Round-4 first-pass finding (Codex) | Resolution |
|---|-----------------------------------|------------|
| 1 | Throttle claimed to cap "binding tests that COMMIT ≤16" — FALSE; the 17th still commits a fail-closed binding row | §3.5 — reframed as a PROMOTION-ELIGIBILITY cap (Codex option C); throttled rows commit + advance the stream but cannot promote; the bound is stated over PROMOTIONS |
| 2 | S(16) via dry-spell product presented as an approximation; post-discovery α behaviour not pinned | §3.5 — the ONLY exact object is per FRESH cohort `P(≥1)=p_cohort` (dry-spell up to first discovery); the windowed bound is the ≤3·p_cohort dominance worst-case (finding A), NOT the dry-spell product on a rolling window; test §7.3/§7.4 pins the FULL dry-spell vector against the actual `lord_plus_plus_level` |
| 3 | Human override makes it conditional, not an unconditional invariant | Status + §3.5 — reworded to "agent-enforced ≤16/window promotion cap with an individually-signed human bypass"; every "invariant" claim scoped to agents |
| 4 | Throttled rows staying `fdr_binding=1` change the statistical process vs the budget arithmetic | §3.5 — throttled rows (positions 17+) are non-eligible, cannot promote, and are dominated-out of the ≤3·p_cohort bound BY CONSTRUCTION; keeping them in the stream is intentional (option C), not a mismatch |

### Round-3 blocking finding (resolved by adopting the throttle)

| # | Round-3 BLOCKING finding (Codex) | Round-4 resolution |
|---|--------------------------|--------------------|
| 1 | Safety budget depends on `H_near=16` but system does not enforce `H ≤ 16`; N=8 is an operational hope, not an invariant | §3.5 — hard windowed throttle caps promotion-eligible binding tests at `H_near=16` per 365d ⇒ enforced ≤16/window rate cap + worst-case windowed false-promotion `≤3·p_cohort≈7.2%` |
| 2 | External retry bounds are load-bearing but merely reviewer-verified | §3.2 — throttle now carries the near-term bound; (i)–(iv) demoted to defense-in-depth, not the sole line |
| 3 | No-throttle under-justified vs the admitted 18.2%@H=64 surface | §3.5 — throttle ADOPTED; the 18.2%/33.0% surfaces are now per-window unreachable; wedge cost bounded by a human-only signed override |
| 4 | "Human sign-off" substituted for a guardrail | §3.5 — sign-off now approves a BOUNDED residual (worst-case per-window ≤7.2% the code holds for agents); the guardrail is the throttle, not the signature |

### Round-2 blocking findings (carried, still resolved)

| # | Round-2 finding | Resolution |
|---|-----------------|------------|
| 1 | N justified SOLELY by "lands α in band"; no cumulative-exposure budget / cross-cohort throttle | §3.1 — N=8 DERIVED from an explicit near-term budget; §3.5 — the throttle is now BUILT, not declined |
| 2 | `Σα_t=W0` treated as sufficient safety proof; lifetime exposure unargued | §3.4 — honest rescope; near-term now an agent-enforced bounded cap (≤3·p_cohort worst-case + rate cap), lifetime rate-capped by the throttle |
| 3 | retry-attack surface at chosen N unquantified/unaccepted | §3.2 — quantified AND bounded (worst-case ≤3·p_cohort≈7.2% per window) + human-signed |
| — (non-blocking) | exposure field hides partial-cohort spend at small N | §4 — added active-cohort + throttle audit fields; completed-only field unchanged |

### Round-1 findings (carried, still resolved)

| # | Round-1 finding | Resolution |
|---|-----------------|------------|
| 1 | `max(LORD++,0.02)` floor invalidates the spending rule (72.5% all-null FD) | no floor; valid LORD++, `Σα_t^dry = W0`; per-cohort all-null ≈2.5% (test §7.4) |
| 4 | Σ-applied-α exposure not an FDR bound | reverted; `FDR_ALPHA·cohorts_completed` retained (guarantee intact) |
| 5 | `fdr_rejected` silently changes meaning | preserved exactly; `α_t` is genuine LORD++; throttle is a SEPARATE named term, not an overload |
| 7 | nit — remove vestigial `FDR_GAMMA_TRUNCATION` | §6 — removed with re-export + test in the same PR |

## 9. Risk & safe-failure posture

This LOOSENS a real-money multiplicity control, deliberately and in a budgeted, NOW-ENFORCED
direction. The loosening: (c)'s first-test bar goes from dsr≥0.99835 to dsr≥0.99236 (α_1
0.00165→0.00764). The counterweight round 3 lacked: the funnel-wide all-null FALSE-PROMOTION
probability, instead of growing at an uncapped burst rate under sign-off alone, is now bounded to a
worst-case `≤3·p_cohort≈7.2%` per rolling 365-day window (typical aligned 4.9%) — equivalently a
≤16-eligible/window RATE cap — by a hard promotion-eligibility throttle enforced in code FOR AGENTS
(the ≤3·p_cohort worst-case is a rigorous bound over ANY window, not an exact per-window number, §3.5),
with a human-only #329-signed bypass as the sole (and rare) escape hatch. The other two hard
multiplicity defenses (a, b) and the economic/stability floors are untouched; the specific over-fit
case still fails (a)+(b)+(c); per-cohort FDR is a preserved operating target at ≈`FDR_ALPHA`; the
near-term retry-attack surface is OWNED, BOUNDED, and agent-enforced, with holdout scarcity,
breadth-deflation, DSR-variance, and `AGENT_NOVEL_MINT_CAP` now defense-in-depth rather than the sole
bound.

**Honest scope of the fix.** Round 4 structurally closes the NEAR-TERM slice of the round-2/round-3
cumulative-exposure finding (per-window false-promotion probability is now an agent-enforced cap, not
a hope) and rate-caps the lifetime slice; it does NOT claim to make lifetime FDR bounded-in-count,
which is provably incompatible with a passable gate on this funnel (§3.4), nor to be an unconditional
invariant (a human can sign a bypass). That residual — an unbounded-in-count but now rate-limited
lifetime posture, plus the individually-signed human-bypass path — is an intentional, CODEOWNERS-
signed risk acceptance, named plainly here rather than presented as closed. Confined to
CODEOWNERS-protected files; PR stays OPEN for human merge (including sign-off on N ∈ {6,8,12}, the
throttle window/cap, and the agent-enforced per-window worst-case ≤7.2% false-promotion surface); full
GATE-2 + full quality gate before merge.
