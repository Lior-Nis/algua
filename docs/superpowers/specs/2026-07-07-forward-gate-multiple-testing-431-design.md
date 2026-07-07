# Forward-gate multiple-testing / optional-stopping correction (#431)

Date: 2026-07-07
Branch: `fix/forward-gate-mt-431`
Issue: #431 (lane:ds, severity:medium, north-star: safe_scale)

## Round-2 GATE-1 revision summary (Codex BLOCK — this pass closes it)

Codex (gpt-5.5, read-only adversarial) returned BLOCK on two race-safety findings plus five
IMPORTANT items. This revision resolves all of them; the changes are threaded into the sections
below and flagged inline:

- **CRITICAL/HIGH — serialize look-count read + evaluate + insert (findings 1 & 2).** The
  `n_prior_looks` count and the `forward_multiplicity` re-evaluation now happen INSIDE the same
  `BEGIN IMMEDIATE` write transaction that inserts the row, via a new store method
  `record_forward_gate_with_multiplicity_and_maybe_promote` modeled on
  `record_gate_with_fdr_and_maybe_promote` (#339/#324) and `reserve_holdout` (#161). SQLite's
  DB-wide single-writer serializes concurrent promotes so no two see a stale `L` — in-flight
  uncommitted looks included. New section **§3a "Concurrency & atomicity"** + a `busy_timeout`
  concurrency test. This is the gate the reviewer said must be re-run before implementation.
- **`n_concurrent_forward` global vs family-scoped** — explicit decision: kept PLATFORM-GLOBAL as an
  intentional conservative superset of the family count (over-counts, never under-counts); the
  sequential factor stays lineage-scoped. Documented in the multiplicity-term section.
- **Family-lookup fallback** — no longer silently degrades to own-name-only: a missing family seed
  sets `family_seed_present=False` and **fails the check closed at `m>1`** (new fail-closed rule 7),
  not a silent fallback that reopens the clone/sibling hatch.
- **Active-only lineage membership** — the counting join now includes historical membership within
  the horizon (`removed_at IS NULL OR removed_at >= cutoff`), defeating reassign-after-peek. (**[R3]
  widened further:** the CTE SEED now also retains within-horizon-removed membership — see round-3.)
- **Clock inconsistency** — one `now` is threaded through assembly, the in-lock count, and the row's
  `created_at` (replacing `_now()` for this insert), so horizon boundaries are replay-consistent.
- **Precondition failures counted as looks** — the count filters to GENUINE performance peeks.
  (**[R3] SUPERSEDED PREDICATE:** round-2 wrote `realized_sharpe IS NOT NULL AND holdout_sharpe IS
  NOT NULL`, but `realized_sharpe` is never null even for empty runs; round-3 corrects this to
  `holdout_sharpe IS NOT NULL AND n_forward_observations >= min_forward_observations`.)
- **`--family-alpha` #329 wiring** — elevated from "parity polish" to an all-or-nothing hard
  requirement (auth criteria list + `ForwardGateCriteria` + signature challenge) if implemented (§4).
- **decision_json burial** — a `forward_multiplicity_summary` accessor + certificate-summary and
  CLI-payload surfacing (§5) with tests prove a human reviewer sees `m`/`conf_floor`/`psr`.

## Round-3 GATE-1 revision summary (Codex BLOCK — this pass closes it)

Round-2 closed the `n_prior_looks` race but a second GATE-1 pass found the in-lock section was
still too NARROW and three smaller correctness gaps. This pass resolves all five; changes are
threaded into the sections below and flagged `[R3]` inline:

- **CRITICAL — ALL DB-derived multiplicity inputs are now recomputed IN-LOCK, not just
  `n_prior_looks`.** Round-2 recomputed only the sequential look count inside the `BEGIN IMMEDIATE`
  section; `n_concurrent_forward`, `family_seed_present`, and the lineage-component membership were
  still lock-free assembly-time reads that the decision bound on. That reopens the same
  read-evaluate-write divergence one factor over (a racing sibling's ticks or a racing family
  reassignment could push `m` over the bar between the lock-free read and the commit, yet the
  committed row would still carry the stale pass). Fix: the in-lock closure now recomputes a
  **bundle** — `(n_prior_looks, n_concurrent_forward, family_seed_present)` — against the
  write-locked connection, and the lineage-component membership (the CTE) is re-derived there too.
  ALL FOUR lock-free assembly-time values are now purely provisional CLI-payload seeds, exactly as
  `n_prior_looks` was in round-2. The store method's return value IS the in-lock authoritative
  decision (the lock-free `decision` is discarded), and the persisted `n_concurrent_forward` COLUMN
  is set from the in-lock recount so the column, `decision_json`, the certificate summary, and the
  CLI payload can never diverge from what promoted. See §3a.
- **Internal contradiction removed (missing family seed).** Rule 7 says a missing family seed
  FAILS CLOSED at `m > 1`; the "Escape-hatch closure" section still said the count "falls back to
  the `strategy_id`-only tally via the `MAX`". Those are mutually exclusive. Fail-closed (rule 7)
  is the single behavior; the conflicting fallback prose is deleted. The `MAX(component_count,
  strategy_id_count)` floor survives ONLY as the own-looks floor when a seed IS present.
- **Lineage CTE seed retains historical membership within the horizon.** The round-2 CTE seed read
  only the strategy's CURRENTLY-active family (`removed_at IS NULL`), so a disconnected reassignment
  (old membership `removed_at` set, new active membership in an unconnected family) erased the old
  component's recent looks even inside the horizon. The seed predicate is widened to
  `removed_at IS NULL OR removed_at >= <horizon_cutoff>`, so both the old and new components seed the
  walk and their within-horizon looks are retained. (A disconnected reassignment is itself only
  reachable via a human `--new-family` transition — an agent cannot mint a disconnected family, #222
  — so this closes a human-only residual, and a test documents that the agent path cannot exercise
  it.)
- **"Genuine performance peek" predicate corrected to actual code behavior.** `metrics_from_returns`
  returns `sharpe = 0.0` (finite, NON-NULL) on empty / zero-vol / underpowered input
  (`algua/backtest/metrics.py:100`), and `realized_sharpe = float(m["sharpe"])` is therefore ALWAYS
  written non-null — so the round-2 `realized_sharpe IS NOT NULL` predicate excluded nothing and the
  prose claim ("NULL for underpowered runs") was false. The peek predicate is corrected to
  `holdout_sharpe IS NOT NULL AND n_forward_observations >= min_forward_observations` (a bar existed
  to compare against AND the window cleared its own observations floor — the two conditions under
  which a real, powered `realized_sharpe >= bar` comparison actually happened). Empty/underpowered
  rows now genuinely drop out. Tests assert a `T < min_forward_observations` row is NOT counted even
  though its `realized_sharpe` is `0.0`.
- **Re-run Codex GATE-1 after these changes** before implementation (done at the end of this pass).

## Problem (grounded in the code)

`evaluate_forward_gate` (`algua/research/forward_gates.py:141`) is a **single-shot** performance +
integrity gate. Its performance check is the whole statistical story:

```
bar      = max(degradation_factor * holdout_sharpe, sharpe_floor)
passes  <=> realized_sharpe >= bar          # a point-estimate comparison
```

There is **no** multiple-testing / optional-stopping term — no FDR, no DSR, no breadth haircut —
even though the gate is re-runnable along two independent axes:

1. **Sequential re-looks (optional stopping).** At `PAPER` an agent may call `paper promote` after
   every new tick session, each a fresh test of `realized_sharpe >= bar` on a **growing** window,
   stopping at the first pass. At `FORWARD_TESTED` a passing run "refreshes the certificate"
   (`forward_promotion.py:560-582`) with no stage change — and the live wall trusts the newest
   certificate for `CERTIFICATE_FRESH_SESSIONS = 10` sessions. Peeking-until-pass on one fixed
   hypothesis inflates the false-pass probability of the paper->forward_tested edge — the last
   statistical gate before the (human-signed) live wall.

2. **Concurrent breadth (family-wise).** `n_concurrent_forward` — distinct strategies with paper
   ticks overlapping the window — is **computed and persisted** (`forward_promotion.py:282-288`,
   `db.py:495`), stored on `AssembledEvidence` (`:80`) and in the gate row (`run_forward_gate` at
   `:553`), and is even surfaced in the CLI payload and the certificate summary
   (`forward_promotion.py:466`). But it is **not a field of `ForwardEvidence`**
   (`forward_gates.py:76-102`), so `evaluate_forward_gate` never sees it. The code says so
   explicitly: "Concurrency breadth (recorded, not yet enforced)" (`forward_promotion.py:279`) and
   "recorded, not yet enforced" (`db.py`/comment). Running many forward tests at once and taking
   the one that passes is exactly multiple testing, uncorrected.

The backtested->candidate research gate has LORD++ FDR + DSR + bootstrap + regime checks. The
forward gate — structurally the twin gate one lane over, and the LAST gate before live — has none.

## Design decision

Implement the issue's **option (b) strengthened with a sequential term**: promote
`n_concurrent_forward` to a **binding** `ForwardEvidence` field, add a **bounded sequential
re-look count**, fold both into a single effective multiplicity `m`, and add ONE new
`forward_multiplicity` check that **deflates the confidence required to clear the existing bar**,
Bonferroni-splitting a fixed family-wise false-pass budget across the `m` tests in the family.

The correction is **exactly a no-op when `m == 1`** (a fresh identity, run once, with no concurrent
siblings) — so it introduces **zero regression** on the honest single-shot path — and tightens
monotonically as re-looks and/or concurrency grow.

### Why not a full LORD++ ledger for the forward gate (option (a))

Two reasons, both of which a reviewer should hold us to:

- **Statistical mismatch.** LORD++/FDR controls false *discoveries* across a stream of **distinct**
  hypotheses. Sequential re-runs of the *same* strategy+identity on a growing window are **not**
  distinct hypotheses — they are repeated **looks** at one hypothesis. FDR is the wrong tool for
  optional stopping; alpha-spending / group-sequential control is the right one. Bolting the
  research gate's per-combo DSR p-value + LORD++ ledger onto the forward gate would conflate the
  two and mis-state the guarantee. (The forward gate also has no parameter *sweep*, hence no
  trial-variance to build a DSR `SR*` from — the DSR machinery does not map.)

- **Anti-scaling (the #324 lesson).** A lifetime-cumulative online-FDR level over forward
  evaluations would recreate exactly the pathology #324 fixed for the research gate: the live wall
  *mandates* periodic re-certification (certificate must be <=10 sessions old), so routine,
  required re-runs would ratchet a lifetime level toward 0 and eventually make the gate
  unpassable — punishing a strategy for *complying* with the freshness wall. Any correction we add
  MUST be immune to this. Ours is (see "Anti-scaling guarantees" below).

### The multiplicity term (exact math)

Let, at evaluation time:

- `n_concurrent_forward` — distinct strategies (`DISTINCT strategy`) with paper ticks overlapping
  the window (already computed; the strategy counts itself, so it is `>= 1` whenever there are
  observations). **Scope decision (round-2 item): this factor is deliberately PLATFORM-GLOBAL, not
  family-scoped, even though the sequential factor is lineage-scoped.** The asymmetry is intentional
  and both directions err conservative (tighten-only): (i) the global concurrent count is a
  strict SUPERSET of the family-scoped count (every family member overlapping the window is also a
  platform member overlapping it), so using it can only OVER-count the family's concurrent
  pass-opportunities, never under-count — the safe direction; whereas (ii) the sequential factor
  MUST be lineage-scoped because a single-name scope there would UNDER-count re-registered clones
  (the unsafe direction — see "Escape-hatch closure"). Framed plainly: the concurrent term is
  platform-wide multiplicity throttling (running many forward tests at once anywhere raises
  everyone's bar), which dominates the narrower family-wise concurrency the prose motivates it with.
  This is a design choice, not an oversight; a future refinement could tighten it to a
  family-scoped-but-still-conservative count, but the global count is already on the safe side.
- `n_prior_looks` >= 0 — count of prior **genuine performance-peek** `forward_gate_evaluations`
  rows (pass OR fail, ANY identity) **across this strategy's entire weakly-connected family LINEAGE
  COMPONENT** — the transitive closure over `#222` `family_parents` edges walked in BOTH directions
  (ancestors AND descendants/siblings), so every family reachable from this strategy's family
  through parent edges is included — **within a trailing horizon** of
  `FORWARD_RELOOK_HORIZON_SESSIONS` sessions (by `created_at`). Excludes the current run (its row is
  written after the count, in the SAME transaction — see "Concurrency & atomicity" below).
  **Lineage-component-scoped, NOT identity-hash-, single-name-, or ancestor-only-scoped** — see
  "Escape-hatch closure" below. Three refinements from round 2, each stated explicitly:
  - **[R3] ALL DB-derived multiplicity inputs are computed IN-LOCK, not just this count.** The
    plain pre-transaction reads in `assemble_forward_evidence` — `n_prior_looks`,
    `n_concurrent_forward`, AND `family_seed_present` (plus the lineage-component membership that
    `n_prior_looks` is derived from) — are ALL INFORMATIONAL only (they seed the CLI payload). The
    values that BIND the decision are re-derived as a bundle inside the `BEGIN IMMEDIATE` write
    transaction that inserts the row, and the multiplicity check is re-evaluated against them. Round-2
    only re-counted `n_prior_looks`, leaving `n_concurrent_forward`/`family_seed_present`/membership as
    lock-free reads the decision still bound on — the same read-evaluate-write divergence one factor
    over (a racing sibling's ticks pushing the concurrent count up, or a racing family reassignment
    changing membership, between the lock-free read and the commit). Recomputing the whole bundle
    in-lock closes it: two concurrent `paper promote` runs on the same lineage cannot both pass on a
    stale `m` on ANY of its factors. The store method returns the IN-LOCK decision; the lock-free one
    is discarded. See "Concurrency & atomicity".
  - **Only genuine performance peeks count (precondition/underpowered failures do NOT). [R3
    corrected predicate.]** A row counts as a look iff a real, POWERED `realized_sharpe >= bar`
    comparison actually happened on that run —
    `holdout_sharpe IS NOT NULL AND fge.n_forward_observations >= fge.min_forward_observations`.
    Rationale, grounded in the code: `metrics_from_returns` returns `sharpe = 0.0` (finite, NON-NULL)
    on empty / zero-vol / underpowered input (`algua/backtest/metrics.py:100`), and
    `realized_sharpe = float(m["sharpe"])` is therefore ALWAYS non-null — so `realized_sharpe IS NOT
    NULL` excludes nothing and the round-2 prose ("NULL for underpowered runs") was simply wrong. The
    two conditions under which a real, powered comparison DID run are instead: (i) a bar existed to
    compare against — `holdout_sharpe IS NOT NULL` (the exact condition the performance check needs to
    avoid its own bar-undefined fail-closed); and (ii) the run's window cleared its OWN observations
    floor — `n_forward_observations >= min_forward_observations` (both are `NOT NULL` columns already
    on every row), which excludes the empty (`T=0`) and underpowered (`T < floor`) runs whose
    `realized_sharpe=0.0` reveals no performance information and carries no optional-stopping risk.
    This is a deliberate, bounded RELAXATION vs the round-1 "count every row": it stops the correction
    from over-penalizing operational failures (missing holdout, broker fetch error surfacing as a
    hygiene fail, too-short window) that are unrelated to peeking. A hygiene/broker failure that
    occurred AFTER a valid, powered Sharpe was computed still counts — the information was revealed.
    The predicate uses only real columns, so it is enforced in the count SQL, not in JSON.
  - **Historical membership within the horizon counts (seed AND join); absent membership fails
    closed. [R3: seed widened.]** BOTH the CTE seed and the final counting join now retain
    within-horizon historical membership, `removed_at IS NULL OR removed_at >= <horizon_cutoff>`:
    - The final **join** includes any strategy that WAS a component member within the horizon, so
      reassigning a clone OUT of the family after it peeks does not erase its recent looks (a
      membership-churn dodge).
    - The CTE **seed** (round-3 fix) is likewise widened from active-only to
      `removed_at IS NULL OR removed_at >= <horizon_cutoff>`, so a DISCONNECTED reassignment of the
      CURRENT strategy — its old membership `removed_at`-set, a new active membership minted in an
      unconnected family — still seeds the walk from the OLD component and retains that component's
      within-horizon looks. (A disconnected reassignment is only reachable via a human `--new-family`
      transition — an agent cannot mint a disconnected family, #222 — so this closes a human-only
      residual; a test asserts the agent path cannot exercise the gap.)

    `family_seed_present` is `True` iff this WIDENED seed returns any row (an active membership OR a
    within-horizon-removed one). If the current strategy has NO membership at all within the horizon
    (its widened seed is empty — only reachable via a human forced transition into this stage), the
    `family_seed_present=False` evidence flag makes the multiplicity check **fail closed whenever it
    would otherwise engage (`m > 1`)** rather than silently degrading to an own-name-only tally (which
    would reopen the exact clone/sibling escape hatch the design claims closed — see fail-closed rule
    7 and "Escape-hatch closure"). This is UNIFORM (no actor split — the evaluator is actor-agnostic);
    it stays inert at `m == 1`, so an honest single-shot strategy that a human forced in without a
    family is unaffected, and the human's recourse under real multiplicity is to restore the
    strategy's family membership (the correct fix). When the seed IS present the count is floored at
    this `strategy_id`'s own peek tally (`MAX(component_count, strategy_id_count)`) — never below its
    own looks. There is NO fallback-to-own-tally when the seed is ABSENT: an absent seed is
    fail-closed, full stop.

Effective multiplicity — a **multiplicative** union bound over the concurrent family (so it
genuinely dominates the family's pass-opportunity count under a symmetric-look approximation,
rather than the additive `N+L` which undercounts the `N*L` family):

```
looks = n_prior_looks + 1                          # this look plus prior looks at THIS strategy
m     = looks * n_concurrent_forward               # = 1 * 1 = 1 on the honest single-shot path
```

With `N` concurrent strategies each looked at `L+1` times, the family has `~N*(L+1)` pass
opportunities; this strategy's `m = (L+1)*N` matches that (exactly, under equal looks across
siblings). If a sibling was looked at *more* than this strategy, `m` under-counts — so the
combined figure is an **operating target, not a proven FWER bound** (see next paragraph).

Required confidence floor (split a fixed family false-pass budget `alpha0` across `m`):

```
conf_floor(m) = 1 - alpha0 / m,   alpha0 = FORWARD_FAMILY_ALPHA = 0.5
```

- `m = 1` -> check is **inert** (see fail-closed rule 1; no PSR computed, status quo exactly)
- `m = 2` -> `conf_floor = 0.75`   `m = 5` -> `0.90`   `m = 10` -> `0.95`   `m = 20` -> `0.975`

**Guarantee framing (operating target, not a formal proof).** Like the research gate's FDR ledger
— which `fdr_lord.py` documents as "an operating target (shared-holdout dependence breaks the
formal guarantee)" — this is a **conservative, tighten-only operating penalty**, NOT a proven
family-wise-error-rate bound. Two known gaps, both stated honestly rather than papered over: (i)
serial dependence in the return series and cross-strategy return correlation break the independence
Bonferroni assumes; (ii) multiplying `looks * n_concurrent_forward` double-counts concurrent
siblings (whose looks appear in both factors) — an **over**-count, i.e. conservative. Note the
sequential factor is now the TRUE total prior-look count across the whole lineage component (not a
symmetric-look approximation), so it does not *under*-count on the sequential axis. It is deployed
because it is **monotone tighten-only in every agent-controllable input** and reduces to the status
quo at `m = 1`, not because it certifies a numeric false-pass rate.

**Why `alpha0 = 0.5`.** The *existing* gate passes iff the point estimate is above the bar, i.e.
iff the Probabilistic-Sharpe-Ratio confidence that the true Sharpe exceeds the bar is `>= 0.5`
(the point estimate sits at the 50th percentile of its sampling distribution). So the status quo's
implicit per-look false-pass tolerance is exactly 50%; `alpha0 = 0.5` anchors the correction so it
is inert at `m = 1` (zero regression) and engages only when there is real multiplicity. 50% is a
loose *family budget* on its own — but it is not the whole gate: the raw `realized_sharpe >= bar`
wall, the vol floor, the drawdown cap, and the full integrity/hygiene battery ALL still apply
unchanged and unconditionally. `family_alpha` is a protected constant a human may only **lower**
(tighten); an agent can never raise it.

The confidence itself is the **Probabilistic Sharpe Ratio** that the true per-period Sharpe exceeds
the bar, using the same Bailey–López de Prado formula the DSR layer already uses
(`algua/research/dsr.py`), with the realized series' higher moments:

```
sr_pp   = realized_sharpe / sqrt(ANN)          # de-annualize (ANN = 252)
bar_pp  = bar / sqrt(ANN)
z       = (sr_pp - bar_pp) * sqrt(T - 1)
          / sqrt(1 - skew*sr_pp + (kurt - 1)/4 * sr_pp^2)
psr     = Phi(z)                                # in [0, 1]
```

`T = n_return_observations`. **Conservative higher moments** (so noisy realized skew/kurtosis can
only ever TIGHTEN, never loosen — favorable positive skew shrinks the PSR denominator and would
inflate confidence, which we refuse to credit): the check uses `skew_used = min(skew, 0.0)` and
`kurt_used = max(kurt, 3.0)`, where `skew`/`kurt` are the realized series' Pearson skew/kurtosis
(already produced by `metrics_from_returns` as `skewness`/`kurtosis`, fisher=False). The
`forward_multiplicity` check passes iff `psr` is finite AND `psr >= conf_floor(m)`.

Note the desirable coupling to `T`: more honest observations shrink the Sharpe standard error, so a
longer track record clears the same `conf_floor` more easily — a genuinely good strategy with a
long window is barely affected, while peeking early on a just-past-63-obs window with many looks is
taxed hard. This is the correct incentive.

### Fail-closed rules (match the module's existing philosophy)

1. **`m == 1` -> the check is INERT (exact zero regression).** When `n_prior_looks == 0` AND
   `n_concurrent_forward == 1`, there is no multiplicity to correct: the check is emitted
   `passed=True` with `detail="m=1: no multiplicity correction"` and **no PSR is computed at all**.
   This makes the correction a literal no-op on the honest single-shot path — a currently-passing
   single-shot run cannot be flipped to fail by any PSR/moment degeneracy (resolving the round-1
   "m==1 is not an exact no-op" objection). The multiplicity check only ever *engages* at `m > 1`.
2. **Bar undefined -> FAIL.** `holdout_sharpe is None`/non-finite, or `degradation_factor`/
   `sharpe_floor` non-finite (the exact conditions the existing performance check already fails
   closed on): at `m > 1` the check is emitted **failed** with a null threshold and a `detail`.
3. **`psr is None` -> FAIL.** Degenerate PSR (`T <= 1`, non-finite `sr_pp`, non-positive variance
   term) at `m > 1` fails closed.
4. **Corrupt concurrency -> FAIL, not floor.** If observations exist (`T >= 1`) but
   `n_concurrent_forward < 1` (impossible by construction), the check **fails closed** with a
   `detail` — it does NOT floor to 1 (a floor would silently loosen relative to true concurrency;
   resolving the round-1 finding). `n_prior_looks < 0` (impossible) likewise fails closed.
5. **`family_alpha` hard-validated.** If `family_alpha` is non-finite or not in `(0, 1]`, the check
   fails closed at `m > 1` (in addition to the direction guard that already blocks an agent from
   *relaxing* it above the protected 0.5 default).
6. **Tighten-only composition.** The new check is **ANDed** into `decision.passed` alongside every
   existing check — it can only ever turn a pass into a fail, never rescue a fail.
7. **No family seed -> FAIL at `m > 1`.** If `family_seed_present is False` the lineage count cannot
   be trusted to include clones/siblings; at `m > 1` the check fails closed (uniform, no actor
   split) rather than silently degrade to an own-name-only tally. Inert at `m == 1`. A strategy
   reaching this stage without a family membership is only produced by a human forced transition;
   the human recourse under real multiplicity is to restore the family membership.

## Changes

### 1. `algua/research/dsr.py` (NOT CODEOWNERS-protected — pure-maths leaf)

Add a pure, fail-closed helper (reuses the scipy `norm` already imported), generalizing
`dsr_confidence`'s `SR*` to an arbitrary threshold and accepting **annualized** inputs so the
protected file never has to touch `ANN`:

```python
def psr_above_threshold_annualized(
    realized_sharpe_ann: float, threshold_sharpe_ann: float, t: int,
    skew: float, raw_kurtosis: float,
) -> float | None:
    """Probabilistic Sharpe Ratio that the true per-period Sharpe exceeds a threshold.
    Annualized Sharpes in; probability in [0, 1] out; None (fail closed) on any degenerate
    input (t<=1, non-finite inputs, non-positive variance term)."""
```

(De-annualizes both Sharpes by `sqrt(ANN)`, then applies the Bailey–López de Prado z/Phi above.
The caller is responsible for passing the conservative-clamped moments; the helper itself does not
clamp, so it stays a faithful PSR primitive.)

### 2. `algua/research/forward_gates.py` (CODEOWNERS-protected)

- New protected wall constants (documented as walls, agent may only tighten):
  - `FORWARD_FAMILY_ALPHA = 0.5` — family-wise false-pass budget, Bonferroni-split across `m`.
  - `FORWARD_RELOOK_HORIZON_SESSIONS = 10` — trailing window over which repeated looks at one
    fixed hypothesis count as optional stopping; aligned with `CERTIFICATE_FRESH_SESSIONS` so
    routine re-certification contributes <= ~1 look while burst re-runs accumulate.
- `ForwardEvidence` (frozen) gains five fields: `n_concurrent_forward: int`,
  `n_prior_looks: int`, `realized_skew: float`, `realized_kurtosis: float`, and
  `family_seed_present: bool` (whether the strategy has a `#222` family membership — active OR
  within-horizon-removed — from which to resolve a lineage component; `False` fails the multiplicity
  check closed at `m > 1` — fail-closed rule 7). **[R3]** THREE of these are DB-derived multiplicity
  inputs — `n_prior_looks`, `n_concurrent_forward`, and `family_seed_present` — and all three are
  re-derived IN-LOCK as a bundle before the row insert; their assembly-time values are provisional
  (payload only). `realized_skew`/`realized_kurtosis` are pure functions of this run's own return
  series (not DB-shared state) and need no in-lock recompute.
- `ForwardGateCriteria` gains `family_alpha: float = FORWARD_FAMILY_ALPHA`.
- `evaluate_forward_gate`: after the existing performance check (unchanged — the raw
  `realized_sharpe >= bar` wall stays), append a stable-named `forward_multiplicity` check:
  - `looks = n_prior_looks + 1`; `m = looks * n_concurrent_forward`.
  - **Rule 1 (inert):** if `n_prior_looks == 0 and n_concurrent_forward == 1` (`m == 1`), emit
    `passed=True`, `detail="m=1: no multiplicity correction"`, compute no PSR.
  - **Rule 4 (corrupt):** if `n_concurrent_forward < 1` or `n_prior_looks < 0`, emit failed.
  - **Rule 5 (alpha):** if `family_alpha` non-finite or `not (0 < family_alpha <= 1)`, emit failed.
  - **Rule 7 (no family seed):** if `family_seed_present is False` (the strategy has no family
    membership — active OR within-horizon-removed — to scope the lineage count), emit failed — the
    count could not be trusted to include clones/siblings, so fail closed rather than proceed on a
    possibly-under-scoped `m`.
  This whole check is **re-evaluated inside the write transaction** with the in-lock bundle
  `(n_prior_looks, n_concurrent_forward, family_seed_present)` **[R3 — all three, not just the look
  count]** (via `dataclasses.replace` on the frozen evidence) — the lock-free evaluation in
  `run_forward_gate` is provisional and is overwritten by the in-lock decision that is actually
  persisted (see "Concurrency & atomicity").
  - Otherwise compute `bar` exactly as the performance check does (rule 2 fail-closed on undefined
    bar), `conf_floor = 1 - family_alpha / m`, `skew_used = min(skewness, 0.0)`,
    `kurt_used = max(kurtosis, 3.0)`, and
    `psr = dsr.psr_above_threshold_annualized(realized_sharpe, bar, T, skew_used, kurt_used)`;
    emit `passed = psr is not None and psr >= conf_floor` (rule 3 fail-closed on `psr is None`).
  - The check payload carries `m`, `n_prior_looks`, `n_concurrent_forward`, `conf_floor`, and `psr`
    for the audit row. `decision.passed` remains `all(c["passed"] ...)`.

### 3. `algua/registry/forward_promotion.py` (CODEOWNERS-protected)

The single-count-and-write is split into **provisional reads at assembly** (payload only) and an
**authoritative re-derive + re-evaluate INSIDE the write transaction** that inserts the row (the
round-2/round-3 CRITICAL concurrency fix — see "Concurrency & atomicity" below). **[R3]** ALL THREE
DB-derived multiplicity inputs are re-derived in-lock; the SQL for each is identical in both places,
only the call site (and the connection's lock state) differs.

- `assemble_forward_evidence`: populate the five new `ForwardEvidence` fields —
  `realized_skew = m["skewness"]`, `realized_kurtosis = m["kurtosis"]` (already computed by
  `metrics_from_returns`; the conservative clamp is applied in the evaluator, not here), and the
  three **provisional** DB-derived inputs seeded here for the CLI payload but NOT bound on:
  - `n_concurrent_forward` — the existing DISTINCT-strategy overlap count (§ assembly step 8),
    computed over the fixed window `[admissible[0].recorded_at, now_iso]`; also stays on
    `AssembledEvidence` for the row payload. **[R3]** provisional — re-counted in-lock.
  - `family_seed_present` — whether the **widened** CTE seed `SELECT family_id FROM family_members
    WHERE strategy_name=? AND (removed_at IS NULL OR removed_at >= <horizon_cutoff>)` returns any row.
    **[R3]** provisional — a racing family reassignment can flip it, so it is re-derived in-lock.
  - `n_prior_looks` — from the lineage-component query below. **[R3]** provisional — re-run in-lock.

  None of the three seeds the decision; the store re-derives all three in-lock and re-evaluates (a
  currently-passing lock-free decision is DISCARDED if the in-lock recompute pushes `m` over the bar
  on ANY factor). The lock-free `decision` computed in `run_forward_gate` never binds.
- The horizon cutoff is the calendar date `FORWARD_RELOOK_HORIZON_SESSIONS` sessions before `now`
  (reuse the `SessionCalendar` session arithmetic already imported), computed ONCE from `now` and
  passed as a fixed ISO string into both the assembly read and the in-lock recount — it depends only
  on `now`, never on DB write state, so it is safe to precompute outside the lock. `now` is the SAME
  clock threaded into the row's `created_at` (see clock note below).
- **Lineage-component count SQL** (the weakly-connected family component via a recursive CTE over
  `family_members`/`family_parents` walked in BOTH edge directions; a new
  `repo.family_component(family_id)` helper generalizing the ancestor-only `family_ancestry` is an
  acceptable factoring):

  ```sql
  WITH RECURSIVE fam(fid) AS (
    -- [R3] seed widened from active-only to include within-horizon-removed memberships, so a
    -- disconnected reassignment of THIS strategy still seeds the walk from its old component.
    SELECT family_id FROM family_members
      WHERE strategy_name = ? AND (removed_at IS NULL OR removed_at >= ?)   -- horizon cutoff
    UNION                                          -- UNION (not ALL) dedups => cycle-safe, finite
    SELECT fp.parent_family_id FROM family_parents fp JOIN fam ON fp.child_family_id = fam.fid
    UNION
    SELECT fp.child_family_id  FROM family_parents fp JOIN fam ON fp.parent_family_id = fam.fid
  )
  SELECT COUNT(DISTINCT fge.id) FROM forward_gate_evaluations fge
    JOIN strategies s ON s.id = fge.strategy_id
    JOIN family_members fm ON fm.strategy_name = s.name
   WHERE fm.family_id IN (SELECT fid FROM fam)
     AND (fm.removed_at IS NULL OR fm.removed_at >= ?)   -- historical membership within horizon
     AND fge.created_at >= ?                             -- horizon cutoff (same string)
     AND fge.holdout_sharpe IS NOT NULL                  -- [R3] a bar existed to compare against ...
     AND fge.n_forward_observations >= fge.min_forward_observations  -- ... AND the window was powered
  ```

  Both recursive branches (up via `child->parent`, down via `parent->child`) give the undirected
  connected component. `COUNT(DISTINCT fge.id)` (not `COUNT(*)`) is REQUIRED: dropping the
  `removed_at IS NULL` filter on the join means one strategy can match several `family_members` rows
  (a current + a historical membership), which would multiply-count a single look under `COUNT(*)` —
  the DISTINCT collapses each look to one regardless of how many membership rows it joins. The final
  join drops the `removed_at IS NULL` filter (round-2 fix) — a strategy that WAS a component member
  within the horizon still counts, defeating the reassign-after-peek dodge. **[R3]** The peek
  predicate is `holdout_sharpe IS NOT NULL AND n_forward_observations >= min_forward_observations`
  (a bar existed AND the window was powered), NOT `realized_sharpe IS NOT NULL` — because
  `realized_sharpe = float(m["sharpe"])` is written `0.0` (non-null) even for empty/underpowered runs
  (`metrics_from_returns` returns `sharpe=0.0` on empty/zero-vol input,
  `algua/backtest/metrics.py:100`), so a `realized_sharpe IS NOT NULL` filter would count empty and
  underpowered runs that revealed no powered performance comparison. `n_prior_looks =
  max(component_count, strategy_id_count)` where `strategy_id_count` is the same horizon+peek count
  filtered to this `strategy_id` alone (the own-looks floor). When the widened CTE seed is empty
  (`family_seed_present=False`) the count is NOT trusted and there is NO own-tally fallback: the
  evaluator fails the multiplicity check closed at `m > 1` (rule 7). The strategy name for the CTE is
  already in scope (`name` parameter).
- Replace the stale "recorded, not yet enforced" comment (`:279-282`) with the enforcement note.
- `guard_forward_relaxations`: add `family_alpha` to the direction guard — a **higher**
  `family_alpha` is looser, so it is a relaxation an agent may not make (human-only), a **lower**
  value is a permitted tightening.
- No new DB columns: `n_concurrent_forward` is already a column; `n_prior_looks`, `m`, `conf_floor`,
  and `psr` ride in `decision_json` (the full audit record). **No `SCHEMA_VERSION` bump.**

### 3a. Concurrency & atomicity (round-2 CRITICAL/HIGH fix, round-3 widened to ALL inputs)

**The race being closed.** The pre-round-2 flow computed `n_prior_looks` as a plain read in
`assemble_forward_evidence`, evaluated the gate lock-free, and only THEN inserted the row in a
separate write transaction. Two concurrent `paper promote` runs on the same lineage component both
read the same `L`, both evaluate against the same `m`, and both pass — the classic
read-evaluate-write gap. **[R3]** Round-2 closed this for `n_prior_looks` but left the OTHER two
DB-derived factors — `n_concurrent_forward` (a racing sibling's ticks landing between the lock-free
read and the commit) and `family_seed_present` (a racing family reassignment) — as lock-free reads
the decision still bound on, i.e. the same defect one factor over. This mirrors the exact defect
classes already fixed for the research-promote holdout
(`reserve_holdout`/`finalize_holdout_reservation`, #161) and the drift-checked LORD++ FDR ledger
(`record_gate_with_fdr_and_maybe_promote`, #339/#324), and the fix is the same pattern.

**The fix — recompute the WHOLE multiplicity bundle + evaluate + insert are ONE `BEGIN IMMEDIATE`
critical section.** A new store method `record_forward_gate_with_multiplicity_and_maybe_promote`
replaces the current `run_forward_gate` two-path tail (`record_forward_pass_and_promote` /
`record_forward_gate_evaluation`). Modeled directly on `record_gate_with_fdr_and_maybe_promote`, it
is **top-level only** (raises if `conn.in_transaction`), takes the write lock up front via
`BEGIN IMMEDIATE`, and inside that one locked section:
  1. **[R3]** `inputs = recompute_multiplicity_inputs(conn)` — a single injected closure that
     re-derives ALL THREE DB-derived factors against the live, write-locked connection and returns
     `(n_prior_looks, n_concurrent_forward, family_seed_present)`: the lineage-component count SQL
     (which internally re-derives the CTE membership), the DISTINCT-strategy concurrent-overlap count
     over the fixed window `[window_lo, now_iso]`, and the widened family-seed existence check. The
     closure is over the strategy name/id + the fixed window bound + the precomputed horizon cutoff,
     mirroring how `level_fn` is injected into the FDR method; the store stays agnostic of the
     statistics. (The `window_lo`, `now_iso`, and horizon cutoff are pure functions of `now` and this
     strategy's own already-fixed first-admissible-tick `recorded_at` — none of them race — so they
     are precomputed at assembly and passed in as fixed params; only the COUNTS/existence they feed
     are re-run in-lock.)
  2. **[R3]** `decision = reevaluate(inputs)` — a pure closure that does
     `evaluate_forward_gate(dataclasses.replace(evidence, n_prior_looks=inputs.looks,
     n_concurrent_forward=inputs.n_concurrent, family_seed_present=inputs.seed_present), criteria)`.
     The lock-free decision computed earlier in `run_forward_gate` is provisional and is DISCARDED;
     ONLY this in-lock decision is persisted, acted on, AND returned to the caller (so the CLI payload
     and certificate summary, which read the returned/persisted decision, can never diverge from what
     promoted).
  3. **[R3]** Patch `decision_json` to the re-evaluated decision (exactly as the FDR method patches
     `raw_decision["passed"]` + the check payload before insert), set `gate_row["passed"]`, the
     `n_prior_looks`/`m`/`conf_floor`/`psr` audit fields, AND the persisted `n_concurrent_forward`
     COLUMN from the in-lock recount (`family_seed_present` rides in `decision_json`) — so the column,
     `decision_json`, the certificate summary, and the CLI payload are all sourced from the ONE in-lock
     recompute.
  4. INSERT the row with `created_at = now` (the threaded clock — see below), then, iff
     `decision.passed and rec.stage is Stage.PAPER`, the stage CAS + transition INSERT (the existing
     `_apply_transition_locked` born-consumed semantics, unchanged). A refresh at `FORWARD_TESTED`
     inserts a `consumable=False` certificate row with no stage change, as today.
  5. COMMIT (single `with self._conn` / explicit commit, `BaseException`-rollback like the FDR
     method).

**Why this serializes per lineage component (round-2 HIGH #2 — in-flight looks).** SQLite has a
single DB-wide writer: a second `paper promote` calling `BEGIN IMMEDIATE` **blocks** on the write
lock (subject to the `busy_timeout=5000` already configured, #164) until the first COMMITs. So the
loser's in-lock `recompute_multiplicity_inputs` runs AFTER the winner's row is committed and
therefore counts it — the two runs are strictly serialized and the second observes `L+1` (and any
newly-committed concurrent-sibling ticks / membership changes), not the stale reads. This
DB-wide serialization is COARSER than the per-component advisory lock the reviewer suggested, and
strictly dominates it (it serializes even unrelated components — a negligible-throughput cost here,
and the FDR/holdout methods already accept the same coarse write-lock serialization). In-flight
(uncommitted) writes from a third transaction are invisible under SQLite isolation, but that
transaction cannot have committed without having HELD the write lock, so it is strictly ordered
before or after this one — never concurrently visible-yet-uncounted. There is no interleaving in
which two runs both pass on stale multiplicity inputs on ANY of the three factors.

**One clock through assembly, count, and insert (round-2 clock-consistency item).** `now` is
threaded from `run_forward_gate` into (a) the horizon-cutoff computation, (b) the in-lock bundle
recompute, and (c) the inserted row's `created_at` — replacing the store's internal `_now()` for THIS
insert. So a
row written by a racing promote at `now_A <= now_B` lands on the deterministic side of run B's
horizon boundary (`created_at = now_A >= cutoff_B` iff genuinely within the horizon), and tests /
replays that inject a fixed `now` see a self-consistent horizon. `_insert_forward_gate_row_locked`
gains a `created_at` parameter (defaulting to `_now()` for any non-forward caller, though it has
none) threaded from the new method.

### 4. `algua/cli/paper_cmd.py` (CODEOWNERS-protected) — optional, but ALL-OR-NOTHING if included

The protected `FORWARD_FAMILY_ALPHA = 0.5` default already binds without any flag, so this whole
section may be **omitted** in the first cut. **But if a `--family-alpha` flag is added it MUST be
FULLY wired through the `#329` authenticate+guard context — a half-wired flag is a hard reviewer
reject, not "parity polish."** "Fully wired" means all three touchpoints, in the same shape as the
existing `--degradation-factor`/`--sharpe-floor`/... relaxation flags:
  1. **Auth criteria list.** `family_alpha` joins the list of relaxation parameters `paper_cmd.py`
     feeds into the `#329` authenticate-then-guard flow, so an agent passing a **looser** (higher)
     value than 0.5 is rejected pre-flight exactly like the other relaxations (the
     `guard_forward_relaxations` direction guard added in §3 is the enforcement; the CLI must ROUTE
     the flag through it, not bypass it).
  2. **`ForwardGateCriteria`.** The parsed flag populates `ForwardGateCriteria.family_alpha` (added
     in §2), which flows into `evaluate_forward_gate`.
  3. **Human signature challenge.** `family_alpha` is included in the relaxation set that the
     human-approval signature challenge enumerates/binds, so a human relaxation is signed over the
     same value the gate then uses (no unsigned side-channel).
Omitting any one of the three is worse than omitting the flag entirely (it would present a
relaxation knob that skips authentication). Covered by `tests/test_cli_paper.py -k family_alpha`
asserting agent-rejected / human-signed-and-applied / value-threaded-into-criteria.

### 5. Surfacing the multiplicity fields to a human (round-2 item — no silent JSON-only burial)

`n_prior_looks`, `m`, `conf_floor`, and `psr` ride in `decision_json` (no schema bump), but they
MUST be legible to the human who signs the go-live challenge and to an operator reading the CLI —
not buried. Concretely:

- **Helper accessor.** Add a small pure accessor
  `forward_multiplicity_summary(decision_json: str | dict) -> dict` (natural home:
  `algua/research/forward_gates.py` next to the check, or a `forward_promotion` helper) that reads
  the `forward_multiplicity` check payload out of a stored decision and returns
  `{m, n_prior_looks, n_concurrent_forward, conf_floor, psr, passed}` (or a documented empty/`None`
  for a legacy row with no such check). This is the single reader both the certificate summary and
  the CLI payload use, so the surfaced values can never drift from the stored ones.
- **Certificate summary.** `verify_forward_certificate`'s returned summary dict (the thing the human
  signs against, `forward_promotion.py:460-467`) gains the multiplicity block via the accessor
  (parsed from the selected row's `decision_json`), alongside the existing `n_concurrent_forward`
  column it already surfaces — so the live-wall challenge shows how much multiplicity correction the
  certifying run was under.
- **CLI payload.** The `paper promote` JSON payload includes the multiplicity block; `--summary`
  (#349) includes at least `m`, `conf_floor`, `psr`, `passed` (the decision-relevant scalars) so
  the context-rot-defended projection still tells an operator whether/why the correction bit.
- **Tests prove it surfaces.** `tests/test_forward_promotion.py` asserts the certificate summary
  carries the multiplicity block for a corrected (`m>1`) run; `tests/test_cli_paper.py` asserts the
  CLI payload (full and `--summary`) surfaces `m`/`conf_floor`/`psr`; an accessor unit test covers
  the legacy-row / missing-check path.

## Escape-hatch closure (round-1 findings #2, #6, and round-2 re-registration hatch)

The sequential look count is scoped to the strategy's **FAMILY + transitive ancestor families**
(the #222 family DAG), NOT to the identity hashes and NOT to a single `strategy_id`/name. This
closes three escalating escape hatches:

- **Identity-hash churn (r1 #2/#6).** Keying on hashes would let an agent reset `n_prior_looks` by
  editing a comment or a param (new `code_hash`). We never compare hashes in the count query, so
  the churn is inert — and the `dependency_hash IS NULL` NULL-safe hazard is **moot**.
- **Re-registration under a new name (r2).** Copying a peeked/failed strategy to a new registered
  name yields a new `strategy_id`, which a single-name scope would treat as a fresh slate. Lineage
  scoping defeats this because the platform's family classifier (#222) clusters by code-ancestry +
  factor-lineage + **return-correlation**: an economic clone correlates ~1.0 with its source and is
  assigned (MERGE) into the **same family**, or (PARENTAGE) into a child family joined by a parent
  edge to the incumbent — and we count the whole connected component, so the clone inherits the
  lineage's recent looks; the reset fails.
- **PARENTAGE sibling/descendant fan-out (r3).** An ancestors-only walk would catch a clone's
  *source* but not multiple PARENTAGE **sibling** children of one incumbent, nor descendants, whose
  looks could be staggered to dodge `n_concurrent_forward` too. Walking `family_parents` in BOTH
  directions counts the entire **weakly-connected lineage component** — ancestors, descendants, and
  siblings-via-a-common-ancestor are all in it — so every PARENTAGE-classified relative shares the
  component's recent-look budget regardless of run timing. This is the r3 fix.
- **Disconnected reassignment of the current strategy within the horizon (r3, [R3] seed fix).**
  Moving THIS strategy OUT of its component into a fresh disconnected family (old membership
  `removed_at`-set, new active membership elsewhere) would, under an active-only CTE seed, orphan the
  old component's recent looks. The round-3 seed is widened to
  `removed_at IS NULL OR removed_at >= <horizon_cutoff>`, so the walk still seeds from the old
  component and retains its within-horizon looks. This reassignment is itself only reachable via a
  human `--new-family` transition (an agent cannot mint a disconnected family — see next bullet), so
  it is a human-only residual that the seed widening closes anyway; a test asserts the agent path
  cannot exercise it.
- **Spinning up a fresh (disconnected) family to escape (the only remaining door).** The only way
  out of the connected component is a NOVEL classification — a family with no parent edge to the
  incumbent — and that is already **agent-fail-closed** in #222: NOVEL requires `--new-family` AND
  `--actor human` (an agent cannot mint a disconnected family). So an agent has no path to a
  clean-slate lineage; a human who deliberately opens one is the accountable actor. The read-only
  `research family-audit` detector (return-correlation-authoritative) is the backstop for a human
  who splits a thesis across genuinely disconnected families.

**No-seed behavior is fail-closed, not a fallback (finding 2 — contradiction removed).** If the
current strategy has NO resolvable family membership within the horizon (its widened CTE seed is
empty — only reachable via a human forced transition into this stage), the multiplicity check
**fails closed at `m > 1`** (`family_seed_present=False`, fail-closed rule 7) and stays inert at
`m == 1`. There is NO fallback to a `strategy_id`-only tally: an earlier draft described such a
fallback, which directly contradicted rule 7 and is DELETED. The `MAX(component_count,
strategy_id_count)` own-looks floor survives ONLY as a floor when a seed IS present (never below the
strategy's own looks); it is not a substitute for an absent seed. Trade-off stated plainly: with a
present seed, sibling strategies in one family share a recent-look budget within the horizon —
deliberately conservative, and it resets as looks age.

## Anti-scaling guarantees and the sequential-control trade-off (the #324 defense, explicit)

- **The horizon term is deliberately burst-rate limiting, and we claim nothing more.** Round-1
  finding #3 is correct on the mechanism: horizon-bounded counting taxes *burst* re-looks and lets
  looks age out after `FORWARD_RELOOK_HORIZON_SESSIONS`. We do **not** claim to control sequential
  false-pass probability over unbounded time — and this is not a gap we can close, it is a
  **forced trade-off**: controlling *lifetime* sequential false-pass requires an unboundedly growing
  penalty, which is exactly the lifetime-cumulative anti-scaling pathology #324 had to remove from
  the research gate (and which here would be *worse*, because the live wall MANDATES periodic
  re-certification, so an unbounded penalty would punish a strategy for complying with the
  freshness wall until the gate became unpassable). The horizon is the principled point on that
  trade-off, and it is aligned with `CERTIFICATE_FRESH_SESSIONS = 10`.
- **"Wait-out peeking" degrades to testing on genuinely fresh evidence.** To let looks age out of
  the horizon, an agent must stop re-running for `> FORWARD_RELOOK_HORIZON_SESSIONS` sessions —
  during which (a) the live certificate goes **stale** (`> 10` sessions) and forfeits live
  eligibility until a fresh pass, and (b) the growing window accrues a full horizon of **new**
  observations. Re-testing after a horizon of new data is not re-peeking the same fluctuation; it
  is a test on materially new evidence, which is legitimate. The horizon length and the certificate
  freshness bound are the same magnitude precisely so these two facts coincide.
- **Concurrent term is a snapshot, not cumulative.** `n_concurrent_forward` counts strategies
  currently overlapping the window; retire siblings and it drops. It cannot be driven up by
  historical throughput. Running `N` genuine concurrent forward tests *should* raise the bar
  ~`N`-fold (one of `N` passing by luck is ~`N`× likelier) — correct family-wise control that
  self-heals. It cannot be gamed to *loosen* anyone's bar (more siblings only tightens).
- **`m` is bounded, not lifetime-cumulative.** `n_prior_looks` is capped by the horizon (even
  though it now sums across a lineage component, only looks within `FORWARD_RELOOK_HORIZON_SESSIONS`
  count, and a component is a bounded set of families); `n_concurrent_forward` is capped by live
  capacity; their product is bounded and resets as looks age / siblings retire. This is the precise
  property #324 required — lineage-component scoping widens *which* looks count but does NOT make the
  count lifetime-cumulative.
- **Direction is tighten-only end to end.** `m >= 1`, `conf_floor` is monotone increasing in `m`
  and bounded in `[0.5, 1)`, higher-moment inputs are clamped to their conservative side
  (`min(skew,0)`, `max(kurt,3)`), the check is ANDed in, and every degenerate/undefined input fails
  closed. **There is no input an agent controls that relaxes the gate**, and no cheap identity
  churn resets the sequential count.

## CODEOWNERS / merge note (corrected from the task's stale operating rules)

The task's operating rules listed `forward_gates.py` and `forward_promotion.py` as **not**
protected. **This is stale.** The CODEOWNERS file in this worktree (`/CODEOWNERS`) now protects
BOTH:

```
/algua/research/forward_gates.py      @Lior-Nis   # forward-test gate criteria (#124)
/algua/registry/forward_promotion.py  @Lior-Nis   # forward-test evidence assembly + live certificate
/algua/registry/store.py              @Lior-Nis   # registry DB writes (in-lock accounting method)
```

`paper_cmd.py` is protected too. The round-2 concurrency fix ALSO adds an in-lock accounting method
to `store.py` (§3a) — likewise CODEOWNERS-protected. Therefore this PR **touches
CODEOWNERS-protected paths and MUST stay OPEN for human merge** — it may not auto-merge even on
green CI. This is correct: the change strengthens the last statistical wall before the live gate,
and the concurrency-critical accounting lives in the shared registry write path; a human should sign
off. The `dsr.py` helper is the only unprotected production file touched.

## Test plan

- `tests/test_forward_gates.py` (pure evaluator):
  - `m == 1` INERT: a case that passes today still passes; `forward_multiplicity` present,
    `passed=True`, `detail` names `m=1`, and NO PSR is computed (a degenerate-moment single-shot
    that passes today still passes — the exact-no-op property).
  - tightening: fixed evidence with `psr` just above some `conf_floor`; raising `n_prior_looks`
    and/or `n_concurrent_forward` so `m` grows and `conf_floor` exceeds `psr` flips `passed`->False.
  - multiplicative `m`: `n_prior_looks=1, n_concurrent_forward=1` -> `m=2`; `0,3` -> `m=3`;
    `1,3` -> `m=6` (asserts `m = (looks) * n_concurrent`).
  - conservative moments: a large POSITIVE realized skew does NOT raise `psr` vs `skew=0`
    (clamped by `min(skew,0)`); low kurtosis clamped up to 3.
  - fail-closed at `m>1`: `holdout_sharpe=None` -> failed with null threshold; `T<=1` -> failed;
    corrupt `n_concurrent_forward=0` -> **failed** (NOT floored); `family_alpha=1.5` or `0` or
    `nan` -> failed; `family_seed_present=False` at `m>1` -> **failed** (rule 7), but inert at `m==1`.
  - direction: more looks / more concurrency never turns a fail into a pass.
- `tests/test_forward_promotion.py` (assembly + in-lock accounting):
  - `n_prior_looks` counts prior rows across the strategy's **weakly-connected family lineage
    component** inside the horizon and excludes the current run; an identity/hash change does NOT
    reset it; a **clone under a new name in the same family (MERGE)** does NOT reset it; **PARENTAGE
    sibling fan-out** — two child families of one incumbent, with **staggered NON-overlapping** paper
    windows (so `n_concurrent_forward` would NOT catch them) — still share the component count (the
    r3 case); a genuinely **disconnected** family (no parent edge) is NOT counted (only reachable by
    a human `--new-family`); rows older than the horizon are excluded.
  - **[R3] Genuine-peek filter (corrected predicate):** a prior row with `holdout_sharpe IS NULL`
    (no qualified holdout) is NOT counted; a prior row with `n_forward_observations <
    min_forward_observations` — an empty/underpowered run whose `realized_sharpe` is a finite `0.0`
    (NOT null) — is NOT counted (this is the specific case round-2's `realized_sharpe IS NOT NULL`
    predicate wrongly counted); a prior powered row (`holdout_sharpe` present AND obs >= floor) that
    failed on hygiene IS counted.
  - **Historical membership (join AND [R3] seed):** a strategy reassigned OUT of the family
    (`removed_at` set) but whose look is within the horizon STILL counts (reassign-after-peek dodge
    defeated); a membership removed BEFORE the horizon does not; **[R3]** a DISCONNECTED reassignment
    of the CURRENT strategy within the horizon still seeds the walk from its old component (old-component
    looks retained) — and a test documents this reassignment is only reachable via a human
    `--new-family` transition (the agent path cannot produce a disconnected family).
  - **No family seed (finding 2 — fail-closed, NO fallback):** a strategy with no resolvable family
    membership within the horizon sets `family_seed_present=False` and the evaluator fails it closed at
    `m>1` (inert at `m==1`) — assert there is NO `strategy_id`-only fallback tally; with a seed
    present, the count is floored at the own `strategy_id` peek tally (`MAX`).
  - The five new `ForwardEvidence` fields are populated from `metrics_from_returns` + the query.
- `tests/test_forward_promotion.py` **([R3] all-inputs in-lock recompute):** the store method's
  returned decision is the IN-LOCK one, sourced from re-derived `n_prior_looks` AND
  `n_concurrent_forward` AND `family_seed_present` (not the assembly-time seeds); assert the persisted
  `n_concurrent_forward` COLUMN, `decision_json`, and the returned decision all agree; a case where a
  racing sibling's committed ticks raise the in-lock `n_concurrent_forward` above the assembly seed
  flips a lock-free pass to a persisted fail (and yields no promotion), and a racing family
  reassignment that flips `family_seed_present` to False in-lock likewise fails closed.
- `tests/test_forward_promotion.py` (concurrency — analogous to `tests/test_concurrency.py`'s
  `busy_timeout` registry coverage, #164): **two concurrent `paper promote` runs on the same lineage
  component do NOT both pass on a stale `L`.** Drive two sessions (separate connections,
  `busy_timeout` set) racing `record_forward_gate_with_multiplicity_and_maybe_promote`; assert they
  serialize — the second run's persisted `n_prior_looks` includes the first's committed row
  (`L_2 == L_1 + 1`), so a case tuned to pass at `L_1` but fail at `L_1+1` yields exactly ONE
  promotion, not two. Also assert the store method raises if called inside an open transaction
  (top-level-only contract).
- `tests/test_forward_promotion.py` (clock consistency): with a fixed injected `now`, the inserted
  row's `created_at == now`, and a sibling row stamped at exactly the horizon-cutoff boundary is
  counted deterministically (no `_now()`-vs-`now` boundary drift).
- `tests/test_forward_promotion.py` (surfacing): `verify_forward_certificate`'s summary carries the
  multiplicity block (`m`/`conf_floor`/`psr`/`n_prior_looks`) for an `m>1` certifying run, via the
  `forward_multiplicity_summary` accessor; the accessor returns the documented empty/`None` for a
  legacy row with no `forward_multiplicity` check.
- `tests/research/test_dsr_psr_threshold.py` (new; dsr unit tests live under `tests/research/`):
  `psr_above_threshold_annualized` — monotone in realized Sharpe, in `T`; `Phi(0) == 0.5` at
  `realized == threshold`; None on degenerate input.
- `guard_forward_relaxations`: agent passing `family_alpha > 0.5` is rejected; `<= 0.5` allowed;
  human bypasses.
- `tests/test_cli_paper.py -k family_alpha` (only if §4 is implemented): agent-rejected /
  human-signed-and-applied / value-threaded-into-`ForwardGateCriteria`; CLI payload (full and
  `--summary`) surfaces `m`/`conf_floor`/`psr`.

Fast per-task check during Implement:
`uv run ruff check . && uv run mypy algua && uv run lint-imports && uv run pytest -q
tests/test_forward_gates.py tests/test_forward_promotion.py tests/research/test_dsr_psr_threshold.py`
(add `tests/test_cli_paper.py` if the flag is included). The concurrency test may need the full
`tests/test_forward_promotion.py` module. Full gate at integration only.

## Task list

1. **dsr helper.** Add `psr_above_threshold_annualized` to `algua/research/dsr.py` + unit tests in
   a new `tests/research/test_dsr_psr_threshold.py`. Test file:
   `tests/research/test_dsr_psr_threshold.py`.
2. **ForwardEvidence + evaluator + accessor.** Add the FIVE `ForwardEvidence` fields
   (`n_concurrent_forward`, `n_prior_looks`, `realized_skew`, `realized_kurtosis`,
   `family_seed_present`), the `family_alpha` criterion, the two protected constants, the
   `forward_multiplicity` check (fail-closed rules 1-7 incl. rule 7 no-family-seed), and the pure
   `forward_multiplicity_summary(decision_json)` accessor in `algua/research/forward_gates.py`. Tests
   in `tests/test_forward_gates.py` (incl. the rule-7 and conservative-moment cases). Test file:
   `tests/test_forward_gates.py`.
3. **In-lock accounting store method (round-2 CRITICAL/HIGH fix, [R3] widened to ALL inputs).** Add
   `record_forward_gate_with_multiplicity_and_maybe_promote` to `algua/registry/store.py` (+ repo
   protocol), modeled on `record_gate_with_fdr_and_maybe_promote`: top-level-only, `BEGIN IMMEDIATE`,
   in-lock `recompute_multiplicity_inputs(conn) -> (n_prior_looks, n_concurrent_forward,
   family_seed_present)` (**[R3]** the injected closure re-derives ALL THREE DB-derived factors — the
   lineage-component count SQL, the DISTINCT-strategy concurrent-overlap count over the fixed window,
   and the widened family-seed existence check — not just the look count) + `reevaluate(inputs)` (via
   `dataclasses.replace` on all three fields) + `decision_json` patch + set the persisted
   `n_concurrent_forward` COLUMN from the in-lock recount + return the IN-LOCK decision as the method
   result + row INSERT with threaded `created_at=now` + pass-from-PAPER stage CAS. Thread `created_at`
   through `_insert_forward_gate_row_locked`. **`store.py` is CODEOWNERS-protected.** Concurrency +
   top-level-only + clock + all-inputs-in-lock (returned decision == persisted, column == decision_json)
   tests in `tests/test_forward_promotion.py`. Test file: `tests/test_forward_promotion.py`.
4. **Assembly + guard + wiring.** In `algua/registry/forward_promotion.py`: populate the five fields
   in `assemble_forward_evidence` as PROVISIONAL payload seeds — `family_seed_present` (via the **[R3]
   widened** seed `removed_at IS NULL OR removed_at >= cutoff`), the `n_concurrent_forward` overlap
   count, and the horizon-scoped, **bidirectional lineage-component, [R3] corrected-peek-predicate
   (`holdout_sharpe IS NOT NULL AND n_forward_observations >= min_forward_observations`),
   historical-membership (seed AND join)** `n_prior_looks` query with the `strategy_id`-only `MAX`
   floor; rewrite `run_forward_gate` to build the `recompute_multiplicity_inputs`/`reevaluate`
   closures (over the fixed window bound + horizon cutoff + `now`), call the new store method, and
   bind on/surface the RETURNED in-lock decision (dropping the old
   `record_forward_pass_and_promote`/`record_forward_gate_evaluation` two-path tail AND the lock-free
   `decision`); add `family_alpha` to `guard_forward_relaxations`; refresh the stale "not yet enforced"
   comment; surface the multiplicity block in `verify_forward_certificate`'s summary via the accessor.
   Tests in `tests/test_forward_promotion.py` (MERGE clone-reset-defeated, PARENTAGE staggered sibling
   fan-out, corrected-peek-filter incl. underpowered-`0.0` exclusion, historical-membership incl. [R3]
   disconnected-reassignment seed, no-seed fail-closed with NO fallback, all-inputs-in-lock, surfacing).
   Test file: `tests/test_forward_promotion.py`.
5. **(optional) CLI parity — ALL-OR-NOTHING.** If included, fully wire human-only `--family-alpha`
   into `paper promote` (`algua/cli/paper_cmd.py`): the `#329` auth criteria list, `ForwardGateCriteria`,
   AND the human signature challenge (§4) — no half-wiring — plus the multiplicity block in the CLI
   payload / `--summary`. Test: `tests/test_cli_paper.py -k family_alpha`.
6. **Integration.** Full gate `uv run pytest -q && uv run ruff check . && uv run mypy algua &&
   uv run lint-imports`; PR stays OPEN for human merge (CODEOWNERS: `store.py`, `forward_gates.py`,
   `forward_promotion.py`, `paper_cmd.py`).
