# Forward-gate realized-Sharpe lower confidence bound + optional-stopping / multiple-testing correction (#432, subsumes #431's evaluation gap) — design

Status: design (GATE-1 revision **round 4** — resolves the round-3d BLOCK on the *commit-time
concurrency* of the family alpha-spending ledger: the look counts were read lock-free and the decision
committed without re-validating them under the write lock, so two concurrent same-family evaluations
could both read and spend the same `alpha_j` slot. Round 4 moves the ledger read + `q_eff`/decision
re-validation + row insert/promotion into ONE `BEGIN IMMEDIATE` critical section on the store side,
pins the look index `j` to committed row order, and defines the lineage CTE over HISTORICAL
`family_members` rows so a reassignment cannot sever lineage and reset the budget)
Issues: **#432** (`[ds]` — "Forward-gate realized_sharpe is a point-estimate bar on as few as 63
observations with no confidence lower bound") **and #431** (`[ds]` — "Forward gate
(paper->forward_tested) has no multiple-testing / optional-stopping correction despite being
re-runnable, and n_concurrent_forward is recorded but never evaluated"). Round 3 consolidates the
two because they touch the same performance check on the same protected files and the round-2
GATE-1 review of #432 blocked *specifically* on the #431 gap.

**Revision history.**
- Round 1: Lo/Mertens *analytic IID* SE for the LCB. GATE-1 BLOCKED — an analytic IID SE is not
  serial-correlation robust and daily strategy returns are autocorrelated, so the α was overclaimed.
- Round 2: switched to the **stationary-bootstrap percentile LCB** (the codebase's own
  serial-dependence-robust instrument, the DSR path #221 Slice 2). GATE-1 BLOCKED again — the LCB
  fixes single-look noise but the forward gate is **re-runnable** and strategies forward-test
  **concurrently**, so a below-bar strategy still gets free re-rolls of a noisy tail (optional
  stopping) and the family-wise false-pass rate is uncontrolled. The unqualified "≈5% false-pass"
  framing was unearned.
- Round 3a: kept the robust LCB and folded a multiplicity correction into its quantile, but used a
  fragile **session-horizon** re-look reset (waitable), a loose `family_alpha = 0.5` budget (no
  tightening until m ≥ 11), and a **silently saturating** tail floor. Adversarial Codex re-review
  BLOCKED on all three plus certificate-binding.
- Round 3b: replaced the horizon with **evidence-window overlap** accounting, set
  `family_alpha = 0.05` (clean Bonferroni), and converted the tail floor to a hard fail-closed cap;
  fixed the seed reframing and certificate-display-vs-bound-row binding (the re-review confirmed
  those two landed). But it **exempted genuinely non-overlapping fresh windows**, so a *family* could
  still buy unlimited fresh looks by rolling disjoint 63-session windows — lifetime optional stopping
  across cohorts. BLOCKED.
- Round 3c: made the look count **cumulative and lifetime, scoped to the family lineage component**,
  so fresh disjoint windows also spend budget; a human-gated #222 reset; a fail-closed cap. Codex
  confirmed the three round-2 findings closed, but flagged (a) the `q_eff = α/m` rule is an
  **invalid alpha-spend** (per-look levels `α/1, α/2, α/3, …` sum to `α·H_m ≈ 0.17`, not `α`), and
  (b) the "deep-quantile fails safe" argument was overclaimed. BLOCKED on those.
- Round 3d: replaced `α/m` with a **valid `1/k²` alpha-spending schedule** — per-look
  spend `α·γ_j`, `γ_j = (6/π²)/j²`, so `Σ_j α·γ_j = α = 0.05` over the whole family look sequence
  (the spending-only analogue of the research gate's LORD++ `γ`-weights, #220/#324); Bonferroni over
  *same-family* concurrent siblings; and **fail closed when `q_eff` falls below what the bootstrap
  can resolve** (`Q_MIN_RESOLVABLE`). Adversarial Codex re-review confirmed the spend is now valid
  but BLOCKED on the **commit-time concurrency of the ledger**: `n_prior_lineage_looks` /
  `n_concurrent_forward_family` were read lock-free in `assemble_forward_evidence`, and the passing
  row was committed via `record_forward_pass_and_promote` (whose `BEGIN IMMEDIATE` only guards the
  row INSERT + stage CAS). So two concurrent same-family evaluations could both read `j = 1`, both
  compute `q_eff` at `α·γ_1`, and both commit at look 1 — spending the same `alpha_j` slot twice; and
  the lineage CTE walked only ACTIVE `family_members`, so a reassignment could sever the component and
  reset the budget.
- **Round 4 (this doc):** add the forward-lane analogue of the research funnel's serialized
  commit-time ledger (`record_gate_with_fdr_and_maybe_promote`). A new store method
  **`record_forward_gate_with_multiplicity_and_maybe_promote`** wraps the ledger read
  (`n_prior_lineage_looks` / `n_concurrent_forward_family`), a **re-validation CAS** of those inputs
  against the lock-free snapshot the LCB was computed against, the `q_eff`/decision re-computation, and
  the row insert (+ optional promotion) in ONE `BEGIN IMMEDIATE` critical section. **`j` is pinned to
  committed `forward_gate_evaluations` row order under the lock** (a `COUNT` of committed prior
  distinct-window lineage rows + 1), not a pre-lock snapshot, so the loser of a same-family race aborts
  (leaves no row) and re-runs at the deepened `j`. The lineage CTE is defined over **HISTORICAL**
  `family_members` rows (append-only `removed_at`, never DELETEd), so a reassignment can only ADD
  edges — lineage is monotone and the budget cannot be reset (option a; no fingerprint CAS or schema
  bump needed). The round-4 adversarial Codex re-review confirmed the concurrency/CAS mechanism sound
  and, after two corrections (the `family_parents`/`families` edges of the lineage graph are also
  DELETE-free — with `PRAGMA foreign_keys=ON` + no `ON DELETE CASCADE` blocking referenced-family
  deletes — and the monotonicity invariant is enforced by a mechanical static guard test, not a schema
  comment alone), **APPROVED** the round-4 fix.

## Problem (restated)

The forward gate (`paper -> forward_tested`, the last automated wall before a strategy can be
human-approved to `live`) passes its performance check when the **point estimate**
`realized_sharpe >= max(DEGRADATION_FACTOR * holdout_sharpe, SHARPE_FLOOR)`
(`algua/research/forward_gates.py:191-193`). Two independent statistical defects:

1. **Single-look noise (#432).** `realized_sharpe` is one draw of the annualized Sharpe over the
   ~63-session daily series (`forward_promotion.py:230`). The Sharpe t-stat is
   annualization-invariant: at n=63 a *true* annualized Sharpe of 1.0 gives
   `t = (1.0/√252)·√63 = 0.50` — under one SE from zero. A below-bar strategy passes on a favorable
   63-obs draw.
2. **Optional stopping + concurrency (#431).** The gate is **re-runnable** and many strategies
   forward-test **at the same time**. Re-running a below-bar strategy is a free re-roll of the same
   noisy tail, and the family-wise probability that *at least one* of K concurrent strategies clears
   on noise grows with K. `n_concurrent_forward` is already recorded
   (`forward_promotion.py:279-288`, `db.py:495`) but **never evaluated**.

Fix: certify a serial-correlation-robust **lower confidence bound (LCB)** of the realized Sharpe,
and **tighten that bound as the effective number of forward-gate draws grows** so re-runs and
concurrent siblings are not free re-rolls.

---

## (1) Method: stationary-bootstrap percentile LCB (serial-correlation robust)

**Choice (unchanged from round 2): a one-sided lower-percentile LCB from the Politis–Romano
stationary bootstrap of the daily return series**, reusing `algua/backtest/bootstrap.py` (the same
engine the DSR gate uses). New pure helper `stationary_bootstrap_sharpe_lcb`:

```
block_len = politis_white_block_length(returns, max_fraction)       # automatic, serial-dep aware
seed      = stable_bootstrap_seed(name, window_start, window_end, config_hash)  # deterministic
B_eff     = max(2000, ceil(MIN_TAIL_RESAMPLES / q))                 # >= MIN_TAIL_RESAMPLES below q
for b in 1..B_eff:                                                  # vectorised, ONE draw set
    r*  = stationary_bootstrap_resample(returns, block_len, rng)
    SR* = (mean(r*) / std(r*)) * sqrt(ANN)                          # annualized Sharpe of resample
sharpe_lcb = quantile(SR*, q)                                       # lower q-quantile of THIS set
```

Signature: `stationary_bootstrap_sharpe_lcb(returns, *, q, seed, block_len, min_tail=100) -> float | None`.
Returns `None` (→ gate fails closed) on degenerate input (n < 2, zero variance) or on a **domain
guard** `not (0 < q < 0.5)` — a quantile ≥ 0.5 would *loosen* the bar.

**Monotonicity is exact for the property we rely on.** For a *fixed* draw set the empirical quantile
is exactly non-increasing in `q` — so "smaller `q` ⇒ lower (more conservative) LCB" holds by
construction (tested with a fixed seed). Across different `m` the estimate is not compared to itself
(each evaluation has one `m`, one `q_eff`, one deterministic draw set); and `B_eff` only ever *grows*
as `q` shrinks, so a deeper quantile is always estimated with *more*, never fewer, tail draws
(≥ `MIN_TAIL_RESAMPLES = 100`). This answers the round-3a "Monte-Carlo monotonicity" objection.

**Why the stationary bootstrap:** (a) serial-correlation robustness (random-length blocks preserve
dependence; an IID SE understates the SE → a too-liberal wall, the round-1 BLOCK); (b) determinism
already solved (`stable_bootstrap_seed` = SHA-256 of identity + window → bit-identical re-runs);
(c) codebase consistency (the DSR gate ships this exact instrument, #221 Slice 2); (d) finite-sample
honesty (no Mertens plug-in, no normality).

**Known limitation (documented, §3).** Sharpe bootstrap coverage is *approximate* and mildly liberal
at small n. Acceptable because the wall is defense-in-depth. A studentized / BCa refinement is a
future tightening behind the same seam; a follow-up, not a blocker.

---

## (2) Gate condition: robust LCB whose binding quantile tightens with the effective draw count

The binding performance comparison in `evaluate_forward_gate` (the `else` branch at
`forward_gates.py:191-193`) becomes a **single robust check** whose quantile encodes both the
single-test protection (#432) and the multiple-testing / optional-stopping correction (#431):

```
bar    = max(criteria.degradation_factor * holdout_sharpe, criteria.sharpe_floor)
j      = n_prior_lineage_looks + 1                                 # this look's index in the family sequence
gamma  = (6/pi^2) / j^2                                            # 1/k^2 spending weight; SUM_j gamma_j = 1
alpha_j = FORWARD_FWER_ALPHA * gamma                              # this look's spend; SUM_j alpha_j = 0.05 (VALID)
q_eff  = alpha_j / n_concurrent_forward_family                    # Bonferroni over same-family concurrent siblings
FAIL-CLOSED if q_eff < Q_MIN_RESOLVABLE                           # deeper than the bootstrap resolves at n~63 -> human
PASS iff  realized_sharpe_lcb(q_eff)  >=  bar
```

This arithmetic runs **twice**: once lock-free on the *provisional* counts to size the (expensive)
bootstrap LCB, and once **in-lock** on the committed counts to bind the decision — `j` is pinned to
committed `forward_gate_evaluations` row order, and the provisional inputs are CAS-re-validated, inside
one `BEGIN IMMEDIATE` critical section (§2c) so two concurrent same-family looks can never both bind
the same `alpha_j` slot.

- `n_prior_lineage_looks` — a **cumulative, lifetime** count of every prior forward-gate evaluation
  in the strategy's family lineage component (§2b). Fresh disjoint windows spend budget too, so there
  is no "roll a new quarter for a free look" path.
- **The spending schedule is a *valid* alpha-spend (round-3c finding 1).** The `j`-th lifetime look
  in the family is tested at `α·γ_j` with `γ_j = (6/π²)/j²`, and `Σ_{j≥1} γ_j = 1`, so the per-look
  levels **sum to `FORWARD_FWER_ALPHA = 0.05`** over the entire (unbounded) family look sequence — a
  genuine family-wise-over-the-sequence control, the spending-only analogue of the research gate's
  LORD++ `γ`-weights (#220/#324). This replaces round-3c's invalid `α/m` rule (whose per-look levels
  summed to `α·H_m ≈ 0.17`). Look 1 tests at `α·6/π² ≈ 0.0304` (≈ the single-test #432 bar, slightly
  stricter); look 2 at `≈ 0.0076`; look 3 at `≈ 0.0034`.
- `FORWARD_FWER_ALPHA = 0.05` — the family-wise budget over the **whole** look sequence (not per
  look). **Fixed; NOT agent-tunable** (mirrors `DSR_ALPHA`); the human-only relaxation is *raising*
  it, gated by #329.
- `n_concurrent_forward_family` — distinct **same-lineage** strategies with paper ticks overlapping
  the window (the family-scoped restriction of the already-recorded `n_concurrent_forward`); it
  Bonferroni-divides the current look's spend across simultaneous siblings whose rows do not yet
  exist. Cross-family concurrency is deliberately out of scope (§3, #222/#228).
- `Q_MIN_RESOLVABLE = 0.005` — a **fail-closed floor**: when the schedule/concurrency drives `q_eff`
  below the shallowest quantile a 63–few-hundred-obs bootstrap can estimate with usable reliability,
  the gate **fails closed** and hands off to a human (consolidate the family, gather more evidence,
  or authorise a new lineage). At `Q_MIN_RESOLVABLE = 0.005`, `B_eff ≤ 100/0.005 = 20 000` (bounded,
  vectorised). This is the honest replacement for round-3c's overclaimed "deep-quantile fails safe":
  **we do NOT rely on deep-tail coverage** — bootstrap percentile LCBs are not guaranteed
  conservative at deep tails (with autocorrelated 63-obs data the lower tail can be *optimistic*, not
  only pessimistic), so the design keeps `q_eff` shallow and fails closed rather than trusting a
  quantile the data cannot support.

**Why one instrument, not a second analytic PSR.** An earlier draft proposed an analytic
Bailey/López-de-Prado PSR-above-threshold as a second multiplicity check. Rejected: the PSR assumes
IID returns, reintroducing the **exact serial-correlation non-robustness that BLOCKED round 1**, only
in the multiplicity dimension. Folding the correction into the bootstrap quantile keeps the whole gate
serial-correlation robust and needs no separate probability helper.

### (2a) Binding re-test policy — resolving optional stopping (finding 1): a cumulative family ledger + human-gated reset

The policy combines the review's options **(b)** (an alpha-spending ledger) and **(c)** (a
genuinely-new stream after each attempt requires human authority), computed from recorded facts —
not a waitable wall-clock horizon and not an overlap exemption:

- **Every prior look at a DISTINCT evidence window spends budget — fresh or overlapping.**
  `n_prior_lineage_looks` counts prior forward-gate evaluations in the strategy's family lineage
  component (§2b) at **distinct evidence windows** — rows are deduplicated by their `(identity,
  first_tick_ts, last_tick_ts)` window key, and evaluations that never reached a performance verdict
  (below the observation floor, or a degenerate/`None`-LCB series) are excluded (they consumed no
  real *look* at the performance question). It is otherwise **cumulative, never aged out, no overlap
  exemption**: rolling a new disjoint 63-session window is a distinct look and spends its `α·γ_j`
  slice of the budget (§2), deepening `q_eff`; once `q_eff` drops below `Q_MIN_RESOLVABLE` the gate
  **fails closed**. At the current floor a sole family gets **≈ two automated looks** (`q≈0.030`,
  `q≈0.0076`) before the third would fall below `Q_MIN_RESOLVABLE=0.005` and hand off to a human —
  the intended, deliberately-short automated budget. This is the forward-lane analogue of the
  research-gate LORD++ FDR ledger (#220/#324) — coarser (a spending-only `1/k²` schedule, not a
  wealth ledger), with per-look levels that **sum to `α` (a valid spend)**.
- **A same-`(identity, window)` re-run is a certificate *refresh* that does NOT spend budget.**
  Because looks are counted per distinct window, re-running at `forward_tested` on the *same*
  evidence window (the lifecycle's cheap live-wall-certificate refresh) does not increment `j` and
  does not tighten `q_eff` — the refresh recomputes the same LCB from the same
  `stable_bootstrap_seed`-keyed draws. (Its verdict can still change if the *evidence itself* changed
  since — new admissible ticks extend the window into a new distinct look; that is correct.) The
  certificate selection is newest-row-pass-or-fail, so a genuinely below-bar re-evaluation FAILS and
  invalidates any prior pass.
- **A genuinely new stream requires human authority (option c).** The only way to reset the family's
  accumulated looks is to create a **new lineage** — which #222 already makes **agent-fail-closed**
  (`--new-family` + `--actor human`). An agent cannot mint a fresh budget; a human who judges a
  strategy genuinely novel owns that (auditable, #222-clustered, #228-detected) decision and the DB.

**Scope boundary, stated honestly.** The ledger is **per family lineage component**, matching the
research funnel's own boundary. Cross-family lane-wide accounting is deliberately out of scope and is
handled by the *same* governance the research gate relies on: human-only NOVEL-family creation (#222)
and the advisory cross-family gaming detector (#228). A desk running many *unrelated* families is not
controlled by this α — as §3 states, live-wall safety is defense-in-depth, not a lane-wide calibrated
α.

### (2b) The draw count — `n_prior_lineage_looks` and `n_concurrent_forward_family` (finding 2 — *enforced*, over the HISTORICAL lineage)

Both counts are read **twice**: a lock-free *provisional* read in `assemble_forward_evidence` (to size
`q_eff` for the expensive bootstrap LCB, which must run outside the write lock — §2c) and an
**authoritative in-lock re-read** inside the serialized committer (§2c), which is the value that binds
the decision. The pure gate only does arithmetic on whichever counts it is handed.

- **`n_concurrent_forward_family`** — the family-scoped restriction of the already-computed
  `n_concurrent_forward` (distinct strategies with paper ticks overlapping the window): restricted to
  the **same lineage component** via the CTE below. Round 3 **enforces** it (the stale "recorded, not
  yet enforced" comment is replaced); it Bonferroni-divides the current look's spend. (Using the
  family-scoped count, not the raw all-strategies one, is deliberate: cross-family concurrency is the
  out-of-scope lane-wide dimension, §3.)
- **`n_prior_lineage_looks`** — the cumulative count above, scoped to the strategy's
  **weakly-connected family lineage component**: a recursive CTE over `family_members` /
  `family_parents` walked in BOTH edge directions (UNION, cycle-safe) joined through `strategies`,
  then a `COUNT` of prior `forward_gate_evaluations` rows (distinct-window deduped by `(identity,
  first_tick_ts, last_tick_ts)`, verdict-reaching rows only) for every strategy in that component,
  **excluding the current run's own about-to-be-inserted row**. `j = n_prior_lineage_looks + 1`.
  - **The CTE walks HISTORICAL `family_members`, not just active rows (finding a — the round-3d
    BLOCK).** `family_members` is **append-only**: a reassignment SETS `removed_at` on the old
    membership row and INSERTs a new active one — it **never DELETEs** (`db.py:565`, "APPEND-ONLY;
    removed_at SET never DELETE; breadth never decreases"; the two `ux_family_members_*` uniqueness
    indexes are *partial* on `removed_at IS NULL`, so they constrain only the active set — the base
    rows persist forever). The lineage CTE therefore joins `family_members` with **NO `removed_at IS
    NULL` filter** (all historical memberships count as edges).
  - **The whole lineage graph is DELETE-free, not just `family_members` (round-4 Codex finding).**
    The CTE also walks `family_parents` and dereferences `families`, so the proof must cover them too.
    Audited by grep, stated precisely: **no code path anywhere in `algua/` issues a `DELETE` against
    `family_members`, `family_parents`, or `families`.** The ONLY `UPDATE` on any of the three is the
    soft-retire `UPDATE family_members SET removed_at=? WHERE … removed_at IS NULL` (`store.py:1660`) —
    which the historical CTE deliberately IGNORES (it does not filter on `removed_at`, so a
    soft-retired row is still a lineage edge). `family_parents` is strictly INSERT-only via the single
    cycle-guarded `add_parent_edge` (`store.py:1708`) — no `UPDATE`, no `DELETE`; `families` rows are
    create-only. Moreover the store runs with **`PRAGMA foreign_keys=ON`** (`db.py:665`) and the
    `REFERENCES` clauses declare **no `ON DELETE CASCADE`**, so SQLite actively **BLOCKS** deleting a
    `families` row still referenced by a member/parent edge — FK enforcement is a *guard here, not a
    gap*. The lineage graph (`family_members ∪ family_parents ∪ families`) is therefore
    monotone-growing under all program behaviour.
  - **Proof the budget cannot be reset by a reassignment or a parentage change.** Because no edge is
    ever removed (only the `removed_at` soft-retire, which the CTE ignores) and the component is the
    weakly-connected closure over all historical edges, moving strategy S from family A to family B
    adds S↔B edges while the S↔A edges remain, and adding a *thesis* only ever ADDs `family_parents`
    edges — S's component after any such change is a **superset** of its component before. The count
    over a superset of strategies is monotone non-decreasing, so `n_prior_lineage_looks` can only grow,
    never reset. This is why option (a) suffices and option (b) (a lineage-graph fingerprint CAS) is
    not needed for correctness — a reassignment/parentage change that *raced* the eval is still caught
    by the in-lock count CAS (§2c), which aborts on any drift of the committed count.
  - **Mechanical future-change guard, not just a comment (round-4 Codex recommendation).** A schema
    comment alone is too weak for a load-bearing alpha-budget invariant, so a **static regression
    test** (AST/grep over `algua/`, in the same spirit as the #277 data-wall scanner) asserts the
    invariant the proof rests on: **zero `DELETE` against `family_members` / `family_parents` /
    `families`; zero `UPDATE` against `family_parents` / `families`; and the only `UPDATE
    family_members` is the `SET removed_at` soft-retire.** Any future edit that adds a delete or a
    hard membership rewrite fails this test in CI — the invariant is enforced, not merely documented.
    A `db.py` comment cross-references the test.
  - **Defensive fail-closed on a dangling edge.** Should a `family_parents` edge ever reference a
    `families` row the CTE cannot resolve (a hand-edited/corrupted DB — outside the program's own
    behaviour, and already blocked by the FK), the lineage read is treated as an integrity error and
    the gate **fails closed** (rule (v) below), never silently returning a smaller-component count of
    0. The only *sanctioned* budget reset remains a human-authored NOVEL family (§2a); a human
    hand-deleting lineage rows is the same human-owns-the-DB boundary.
  - **Escape-hatch closure (finding 5).** (i) Identity-agnostic (any identity's row counts), so a
    code/config re-hash does NOT reset it. (ii) A MERGE clone inherits the incumbent family (#222) →
    same lineage component → counted. (iii) A NOVEL new family is agent-fail-closed (#222), so an
    agent cannot reset the budget; a human forging a split is the human-owns-the-DB boundary that
    applies everywhere and is the deliberate reset in §2a. (iv) **Fallback** when the strategy has no
    family membership (never in any `family_members` row, active OR historical): the count over
    **this `strategy_id` alone** (still catches same-strategy re-runs). (v) If the lineage query
    itself errors, the gate **fails closed** (unavailable count, not zero).

**Fail-closed rules** (the same arithmetic runs in the pure evaluator for the provisional decision AND
in-lock for the binding decision, §2c; tighten-only, ANDed into `decision.passed`):
`q_eff < Q_MIN_RESOLVABLE` → fail (over-tested / under-resolved); `n_concurrent_forward_family < 1`
or `n_prior_lineage_looks < 0` → fail; `FORWARD_FWER_ALPHA` non-finite or ∉ `(0, 0.5)` →
fail; the computed `q_eff` non-finite or ∉ `(0, 0.5)` → fail; `holdout_sharpe`/criteria non-finite →
existing fail-closed branch unchanged; `realized_sharpe_lcb(q_eff) is None` or non-finite → fail
("bootstrap LCB unavailable / degenerate series"). A do-nothing (zero-vol) strategy fails: LCB `None`
or lower quantile ≤ 0 < bar. The `min_forward_vol`, drawdown, integrity, hygiene and staleness checks
are untouched; `passed` stays `all(c["passed"])`.

**Audit shape.** The binding check is named `realized_sharpe_lcb` (value = the LCB at `q_eff`,
`op=">="`, `threshold=bar`); its `detail`/evidence carries `q_eff`, `j` (look index), `gamma_j`,
`alpha_j`, `n_prior_lineage_looks`, `n_concurrent_forward_family`, `fwer_alpha`, `q_min_resolvable`,
`sharpe_lcb_seed`, `sharpe_lcb_block_len`, `sharpe_lcb_b` (= `B_eff`), `n_obs`, and the point
`realized_sharpe`. The `j`/`q_eff`/`alpha_j`/`gamma_j`/count fields written to the row are the
**in-lock committed** values (§2c), which may differ from the provisional lock-free snapshot.

### (2c) Serialized commit-time ledger — closing the two-siblings-spend-one-slot race (round-4 BLOCK)

The single-look LCB and the alpha-spending schedule are only sound if the `j`-th look truly spends
`α·γ_j` **once**. Round 3d computed `j` from a lock-free snapshot and committed the row through
`record_forward_pass_and_promote`, whose `BEGIN IMMEDIATE` guards only the INSERT + stage CAS — so two
concurrent same-family evaluations could both observe `n_prior_lineage_looks = 0`, both bind at
`q_eff = α·γ_1`, and both commit at look 1. Round 4 mirrors the research funnel's
`record_gate_with_fdr_and_maybe_promote` (`store.py:1264`), whose `BEGIN IMMEDIATE` already makes
"stream read → decision → INSERT → stage CAS" one atomic critical section precisely so "two concurrent
binding evaluations can't both read `t=0` and both write `fdr_test_index=1`" (`store.py:1309-1311`).

**New store method `record_forward_gate_with_multiplicity_and_maybe_promote(rec, *, gate_row,
snapshot, criteria_alpha, actor, reason)`** — TOP-LEVEL ONLY (asserts `not conn.in_transaction`,
mirroring the reference). `snapshot` carries the **provisional** `(n_prior_lineage_looks,
n_concurrent_forward_family, q_eff, sharpe_lcb)` the lock-free LCB was computed against. Inside one
`BEGIN IMMEDIATE … commit()/rollback()` on `BaseException`:

1. **Re-read the ledger under the lock.** Recompute `n_prior_lineage_looks` (the HISTORICAL-lineage
   distinct-window `COUNT` of committed `forward_gate_evaluations` rows, §2b, excluding this run) and
   `n_concurrent_forward_family` from the now-write-locked DB. This read reflects **all prior commits**
   (intentionally live, exactly like the reference's FDR-stream read).
2. **Pin `j` to committed row order.** `j := n_prior_lineage_looks_committed + 1`. Because the winner
   of a same-family race has already INSERTed its distinct-window row before releasing the write lock,
   the loser's in-lock re-read sees that row and computes `j = 2` — the look index is defined by
   committed insertion order, never by the pre-lock snapshot.
3. **Re-validation CAS of the LCB's inputs.** The expensive bootstrap LCB was computed lock-free
   against `snapshot`; committing it is sound **only if** the committed `(n_prior_lineage_looks,
   n_concurrent_forward_family)` still equal the snapshot's (so `q_eff` — and therefore the quantile
   the LCB was drawn at — is unchanged). Exact-equality CAS (mirroring `_cas_funnel`,
   `store.py:1208`): if either drifted (a racing sibling committed a look, or a reassignment enlarged
   the component, §2b), **roll back** — the row and any promotion vanish — and signal the orchestrator
   to recompute the LCB at the deepened `q_eff` and re-attempt (bounded retry; on exhausted retries or
   a `q_eff` that has fallen below `Q_MIN_RESOLVABLE`, **fail closed**). We do NOT recompute the
   bootstrap inside the write lock — a 2 000–20 000-resample bootstrap would hold the global write lock
   for seconds and serialize every writer; the CAS keeps the in-lock work to cheap `COUNT`s +
   arithmetic + the INSERT, exactly as the reference keeps only the FDR-stream read + `alpha_t`
   arithmetic in-lock.
4. **Recompute the decision in-lock and patch `decision_json`.** With `j` pinned and the CAS passed,
   recompute `gamma_j`, `alpha_j`, `q_eff`, the `q_eff < Q_MIN_RESOLVABLE` fail-closed check, and
   `passed = (sharpe_lcb >= bar) AND (all other checks)`; write the committed `j`/`q_eff`/`alpha_j`/
   counts into the row's `decision_json` and the `realized_sharpe_lcb` check (so the audit row reflects
   the value that bound, not the provisional snapshot).
5. **INSERT the row and (on pass from `PAPER`) promote — atomically.** Reuse
   `_insert_forward_gate_row_locked` for the row and `_apply_transition_locked` for the
   `paper -> forward_tested` CAS (both already lock-scoped), so the fail path, the certificate-refresh
   path (pass at `forward_tested`, no stage change), and the pass-and-promote path all commit their
   ledger-pinned row in this one critical section. As in the reference, a stage-CAS mismatch rolls the
   whole transaction back (the loser leaves **no** row) and its run is simply re-executed.

**Why this is the same guarantee the research funnel already ships.** The committed decision is a pure
function of the **committed** ledger state (CAS-verified), `j` is pinned to committed row order (INSERT
under the lock), and the whole read→decide→write→promote is one `BEGIN IMMEDIATE` section — so no two
same-family looks can bind the same `alpha_j` slot, the byte-for-byte translation of the reference's
"can't both write `fdr_test_index=1`" invariant into the forward lane.

---

## (3) Honest guarantee (findings 2 + 3) — what this wall does and does NOT promise

1. **A serial-correlation-robust, family-scoped, cumulative lower confidence bound** on the realized
   Sharpe, with a **valid `1/k²` alpha-spending schedule** whose per-look levels sum to
   `FORWARD_FWER_ALPHA = 0.05` over the whole family look sequence (look 1 ≈ 0.030), Bonferroni-divided
   across same-family concurrent siblings. It *targets* a family-wise ≈ 5% probability of passing a
   truly-at-bar strategy across all looks — an **approximate** control (Sharpe bootstrap coverage is
   mildly liberal at small n; Bonferroni over positively-dependent siblings is conservative; and the
   percentile is only usable while `q_eff ≥ Q_MIN_RESOLVABLE`), **NOT an exact finite-sample FWER**.
2. **A fail-closed under-resolution floor:** once the spend drives `q_eff` below `Q_MIN_RESOLVABLE`
   the gate fails closed (no "re-roll at a floor level" regime); the only budget reset is a
   human-authored new lineage (#222), whose novelty decision is itself auditable (recorded actor +
   #222 clustering + the #228 cross-family detector).

We **drop the unqualified "≈5% false-pass" framing of rounds 1–2.** Honest statement, scoped
precisely: *the forward gate is a serial-correlation-robust LCB with an approximate Bonferroni
family-wise 0.05 budget over the **cumulative** forward looks of a single family lineage (fresh or
overlapping) and its concurrent siblings, hard-capped by a fail-closed over-test limit with a
human-only reset. It is NOT a lane-wide (cross-family) or exactly-calibrated lifetime false-pass α*
— cross-family breadth is governed by the same #222/#228 machinery as the research funnel. **Live-wall
safety continues to rest on defense-in-depth** — the LCB is one AND-check among the vol floor,
drawdown cap, integrity, account-hygiene and staleness checks; the bar is holdout-derived; and
`forward_tested -> live` additionally requires a verified human signature over a fresh,
identity-matched certificate (`live_gate` + `transitions`). No single calibrated α backstops go-live.

**Seed / B reframing (finding 3).** The deterministic `stable_bootstrap_seed` and the `B_eff` sizing
provide **reproducibility / idempotence only** — a re-run over the same window+identity yields a
bit-identical LCB, so certificate refreshes are stable and auditable. They are **not a security
boundary and give no protection against optional stopping across window extensions**: the seed
*changes* when the window changes, precisely so an extended window earns a fresh, honest draw.
Protection against re-rolling is the cumulative family look-ledger and the `q_eff` tightening (§2),
not the seed.

---

## (4) Surfacing the LCB in the human-facing go-live certificate — with a display-vs-binding guarantee (findings 3 + 6)

`realized_sharpe_lcb` is now the **binding gate statistic**, so it must appear where the human reads
the certificate before signing. `verify_forward_certificate` selects the binding row
(`latest_forward_gate_row`), re-checks it is a fresh PASS, and returns the summary dict the CLI prints
as `forward_certificate` (`registry_cmd.py:131-138`) — and the CLI issues the go-live challenge
**from that same verification call, after it returns**. Round 3 extends that summary, read out of the
**exact bound row's** `decision_json`, with `realized_sharpe_lcb`, `sharpe_lcb_alpha` (= the binding
`q_eff`), `family_alpha` (= the effective `FORWARD_FWER_ALPHA` — surfaced prominently so a
human-relaxed budget is visible and certificates are not compared as if they shared the default),
`sharpe_lcb_b` (= `B_eff`), `sharpe_lcb_block_len`, `sharpe_lcb_seed`, and `j` /
`n_prior_lineage_looks` / `n_concurrent_forward_family`, alongside the existing `realized_sharpe`,
`holdout_sharpe`, `n_forward_observations`.

**Display cannot drift from the bound statistic (finding 6).** The summary is parsed from the *same*
`latest_forward_gate_row` the wall binds, in the *same* call that then mints the challenge, so the
LCB the human reads is definitionally the LCB of the row the gate enforced. The signed challenge bytes
intentionally bind **artifact identity + nonce** (the #124 anti-drift design) and the wall
independently guarantees that identity's newest row is a fresh PASS whose binding statistic is that
LCB — so a human cannot sign a strategy whose newest evaluation is not a fresh pass, and cannot be
shown a metric other than the enforced one. If the parsed `decision_json` is missing/malformed, the
verifier raises `TransitionError` (fail closed) rather than surfacing a fabricated number.
Cryptographically signing the metric bytes themselves is possible future hardening, but the
identity-binding + independent newest-row-pass check already prevents consenting to a non-passing
strategy; noted as a follow-up, not a blocker.

---

## (5) MIN_FORWARD_OBSERVATIONS — kept at 63; stated target power

**Keep `MIN_FORWARD_OBSERVATIONS = 63` as the hard floor.** The LCB makes the *effective* required
sample self-adjusting; α controls protection, n controls power (decoupled).

**Concrete target power (first family look, `j = 1`, sole sibling).** A reference strong strategy of
true annualized Sharpe `s = 2.0` reaches ≥ 50% pass power by ~1 trading year (252 obs) and ≥ 90% by
~3 years (756 obs), against the floor bar `b = 0.3` at the first-look level `q_eff = α·6/π² ≈ 0.030`.
We do **not** gate on n reaching a power target; the 63-obs floor is a hard minimum and the
idempotent re-runnable gate lets each strategy accumulate evidence until *its own* LCB clears. Under
later family looks / concurrent siblings the binding quantile is deeper (the `1/k²` spend), so more
evidence is required — the intended cost of breadth.

**Power table (Gaussian approximation, first look `q_eff ≈ 0.030`, `SE_ann ≈ √(ANN/n)`, `ANN=252`,
`b=0.3`, `z≈1.88`, `power = Φ((s-b)·√(n/252) − z)`).** The bootstrap LCB *tracks* these under near-IID
and is **wider → lower power / more conservative** under positive autocorrelation (intended). Values
are marginally below the old `q=0.05, z=1.645` figures (the first-look level is slightly stricter):

| true Sharpe s | n=63 (¼yr) | n=126 (½yr) | n=252 (1yr) | n=504 (2yr) | n=756 (3yr) |
|---|---|---|---|---|---|
| 1.0 | ~0.10 | ~0.12 | ~0.17 | ~0.26 | ~0.33 |
| 1.5 | ~0.15 | ~0.21 | ~0.33 | ~0.52 | ~0.67 |
| 2.0 | ~0.21 | ~0.33 | ~0.52 | ~0.78 | ~0.90 |

Required point-estimate Sharpe to pass the floor bar at the first look (`b + z·SE_ann`, `z≈1.88`):
~4.0 @ n=63, ~2.9 @ n=126, ~2.2 @ n=252, ~1.6 @ n=504, ~1.4 @ n=756. The human-only `--min-observations`,
`--sharpe-floor`, `--degradation-factor` and (optional) `--family-alpha` relaxations still exist.

---

## (6) Files touched — CODEOWNERS status

- `algua/backtest/bootstrap.py` — new pure `stationary_bootstrap_sharpe_lcb` helper (+ `q` domain
  guard + `B_eff` tail auto-scaling). **Not** CODEOWNERS-protected (pure-maths leaf).
- `algua/research/forward_gates.py` — new fixed constants `FORWARD_FWER_ALPHA = 0.05`,
  `Q_MIN_RESOLVABLE = 0.005`, `MIN_TAIL_RESAMPLES = 100`, and the `1/k²` weight `gamma(j) =
  (6/π²)/j²`; new `ForwardEvidence` fields `realized_sharpe_lcb: float | None`,
  `n_prior_lineage_looks: int`, `n_concurrent_forward_family: int`;
  `ForwardGateCriteria.family_alpha: float = FORWARD_FWER_ALPHA`; rewritten performance branch (§2)
  with the fail-closed rules. (`n_concurrent_forward_family` is carried on `ForwardEvidence`; the
  raw `AssembledEvidence.n_concurrent_forward` row field stays for the payload.)
- `algua/registry/store.py` — **new `record_forward_gate_with_multiplicity_and_maybe_promote`** (§2c):
  the `BEGIN IMMEDIATE` serialized committer that re-reads the ledger under the lock, pins `j` to
  committed row order, CAS-re-validates the LCB's inputs, recomputes the binding decision, and INSERTs
  the row (+ optional `_apply_transition_locked` promotion) atomically. Plus a private in-lock
  `_lineage_forward_look_count(strategy_id)` helper implementing the HISTORICAL-lineage distinct-window
  CTE (§2b) and its `n_concurrent_forward_family` sibling read; both reuse the existing
  `_insert_forward_gate_row_locked` / `_cas_*` primitives. Because the new committer is the SINGLE
  forward-gate recorder for every verdict-reaching row (fail, refresh, pass-and-promote — all must be
  ledger-pinned, §2c), the now-dead `record_forward_pass_and_promote` and `record_forward_gate_evaluation`
  production paths are **removed** (not left as unused alternates — the repo's no-dead-code rule), their
  `repository.py` Protocol entries dropped, and their existing tests migrated to the new committer.
  `_insert_forward_gate_row_locked` stays as the shared low-level INSERT the committer calls.
  **CODEOWNERS-policy-protected.**
- `algua/registry/forward_promotion.py` — compute the **provisional** counts + LCB at the provisional
  `q_eff` (lock-free), then call the new store committer with the `snapshot`; on its CAS-drift signal,
  recompute the LCB at the deepened `q_eff` and re-attempt (bounded retry, then fail closed). Replace
  the "recorded, not yet enforced" comment (line ~279); route BOTH the pass-and-promote and the
  fail / certificate-refresh paths through the committer (so every verdict-reaching row is ledger-pinned
  in one critical section); add `family_alpha` to `guard_forward_relaxations` (raising it is the
  human-only relaxation); extend `verify_forward_certificate`'s summary with the LCB fields from the
  bound row's `decision_json` (§4). No change at line 230.
- `algua/cli/paper_cmd.py` — (optional) human-only `--family-alpha` relaxation via the #329 flow.
- `algua/registry/db.py` — **comment-only** (NO `SCHEMA_VERSION` bump, §7): an explicit
  "lineage tables are DELETE-free; `family_parents`/`families` are INSERT-only; the only
  `family_members` UPDATE is the `removed_at` soft-retire; the forward-gate lineage budget (#432)
  depends on this — enforced by `<static guard test>`" note next to those `CREATE TABLE`s (§2b round-4
  proof). Not CODEOWNERS-protected as a comment, but bundled in the human-merge PR anyway.
- Tests: `tests/test_forward_gates.py`, `tests/test_forward_promotion.py`, a `store.py` concurrency
  test, a bootstrap unit test, and a **static lineage-invariant guard test** (§2b: no DELETE on the
  three lineage tables; no UPDATE on `family_parents`/`families`; only `UPDATE family_members SET
  removed_at`).

**Does it touch `algua/research/gates.py`? No.**

**CODEOWNERS.** The repo's `.github/CODEOWNERS` file gates only `/.github/`, `/.gitleaks.toml`,
`/.pip-audit-ignore.txt`. But `store.py`, `forward_gates.py`, `forward_promotion.py` and `paper_cmd.py`
are **policy-protected safety walls** (per the workflow operating rules — `store.py` is explicitly on
the CODEOWNERS-protected list), so the PR **stays OPEN for human merge** (no auto-merge) even with
green CI. `bootstrap.py` is unprotected. No other protected path (`lifecycle.py`, `engine.py`,
`gates.py`, `clustering.py`, `promotion.py`, `transitions.py`, `live_gate.py`, `approvers/`) is
modified.

---

## (7) SCHEMA_VERSION — no bump

**No `SCHEMA_VERSION` bump.** `db.py:SCHEMA_VERSION = 35` (committed HEAD; no competing bump in
flight). `n_concurrent_forward` is already a column; both `n_prior_lineage_looks` and
`n_concurrent_forward_family` are `COUNT`s over existing `forward_gate_evaluations` / `tick_snapshots`
rows through the family CTE (no new column needed); the LCB, `q_eff`, `j`, `alpha_j`,
`n_prior_lineage_looks`, `n_concurrent_forward_family` and bootstrap provenance ride inside the
existing `gate_evaluations.decision_json` blob; the certificate summary parses them back out.
Adding/renaming
checks and adding in-memory `ForwardEvidence` fields changes the JSON payload, not the table shape —
no migration. The new serialized committer (§2c) reads/writes only existing tables
(`forward_gate_evaluations`, `family_members`, `family_parents`, `strategies`, `tick_snapshots`) — no
new column or table. Preferred non-schema design.

---

## Test plan (additive)

- **`stationary_bootstrap_sharpe_lcb`:** deterministic given a fixed seed (two calls identical); LCB
  below the point Sharpe on a noisy series; wider (lower) LCB on a positively-autocorrelated series
  than its shuffled IID counterpart; a **smaller `q` yields a lower (more conservative) LCB on a fixed
  seed** (exact-monotonicity property); `B_eff` scales up as `q` shrinks (≥ `MIN_TAIL_RESAMPLES` below
  `q`); `q ∉ (0, 0.5)` / n<2 / zero-variance → `None`.
- **`evaluate_forward_gate`:** (i) at the first look (`n_prior_lineage_looks=0`, `n_concurrent_forward_family=1`,
  `q_eff≈0.030`), a series whose point estimate clears the bar but whose LCB does not →
  `realized_sharpe_lcb.passed == False` (the #432 regression); a high-Sharpe / large-n series clears.
  (ii) **valid spend / tighten-only:** the per-look levels are `α·(6/π²)/j²` and their partial sums
  stay ≤ `α` (assert `α·(6/π²)·Σ_{j≤J} 1/j² ≤ α`); raising `n_prior_lineage_looks` (larger `j`) or
  `n_concurrent_forward_family` deepens `q_eff` and only flips pass→fail; look-1 `q_eff≈0.030`.
  (iii) **under-resolution floor:** enough looks/siblings to drive `q_eff < Q_MIN_RESOLVABLE` → fail
  closed. (iv) fail-closed: LCB `None`, `n_concurrent_forward_family=0`, `n_prior_lineage_looks<0`,
  `family_alpha ∈ {0, 0.6, nan}`, computed `q_eff` non-finite, holdout None, non-finite criteria —
  all fail; existing branches intact; audit payload carries
  `q_eff`/`j`/`alpha_j`/`q_min_resolvable`/seed/block_len/`B_eff`/`n_obs`.
- **`assemble_forward_evidence`:** `realized_sharpe_lcb` populated deterministically and flows into
  the decision; `n_prior_lineage_looks` — cumulative lineage count excludes the current run and is
  **not** aged out; a fresh **non-overlapping** window still increments it (the round-3b hole
  regression: it must NOT reset to 0); identity/hash change does NOT reset; a MERGE clone under a new
  name does NOT reset (inherits the family); a disconnected human-created NOVEL family resets (starts
  its own component); no-family fallback to `strategy_id`; lineage query error → fail closed;
  degenerate window → LCB `None` → fail closed. `guard_forward_relaxations` rejects an agent raising
  `family_alpha`.
- **`verify_forward_certificate`:** the returned summary carries `realized_sharpe_lcb` +
  `q_eff`/`B_eff`/block_len/seed + `j`/`n_prior_lineage_looks`/`n_concurrent_forward_family` parsed
  from the bound row's `decision_json`; malformed/missing fields → `TransitionError` (no fabricated pass).
- **`record_forward_gate_with_multiplicity_and_maybe_promote` — serialized commit (round-4 finding,
  in `tests/test_forward_promotion.py` or a store test):**
  - **Concurrency-race regression (the core round-4 fix).** Two connections/processes (separate
    `sqlite3.Connection`s on the same DB file, `busy_timeout` set, mirroring the existing
    `record_gate_with_fdr` concurrency tests) race a same-family forward-gate evaluation from a common
    starting state (`n_prior_lineage_looks = 0`). Assert **exactly one** commits at `j = 1`
    (`decision_json.j == 1`, `q_eff ≈ α·γ_1`) and the other either (a) commits at `j = 2` after its
    in-lock re-read/retry (deeper `q_eff`, `alpha_j = α·γ_2`) or (b) fails closed — **never** a second
    row at `j = 1`. Assert the committed `forward_gate_evaluations` rows carry **distinct** `j` values
    (no two looks share an `alpha_j` slot), and that the loser leaves no orphaned row when it rolls
    back on stage-CAS/`q_eff`-floor failure.
  - **CAS-drift abort.** Simulate the LCB's provisional `snapshot` counts drifting from the committed
    counts (inject a sibling commit / a membership insert between the lock-free read and the in-lock
    re-read) → the committer rolls back and signals recompute; after retry the row commits at the
    corrected `j`. A snapshot that still matches commits without a retry.
  - **Family-reassignment-does-not-reset-budget (finding a), scoped over HISTORICAL memberships.**
    Seed a family A with `k` prior distinct-window forward-gate looks for strategy S, then reassign S
    to family B (SET `removed_at` on the A membership, INSERT an active B membership — the append-only
    reassignment). Assert the in-lock `_lineage_forward_look_count(S)` still returns `≥ k` (the
    historical A edge keeps S in A's component; the count is monotone, does NOT reset to 0), so the
    next look binds at `j = k + 1`, not `j = 1`. A control that walks only ACTIVE members must FAIL
    this test (guards against a regression back to the round-3d active-only CTE). Also: a reassignment
    into a family that *already* has looks yields a component-union count ≥ both (superset
    monotonicity).
- **Static lineage-invariant guard (round-4 finding a hardening):** an AST/grep scan over `algua/`
  asserts zero `DELETE` on `family_members`/`family_parents`/`families`, zero `UPDATE` on
  `family_parents`/`families`, and that the only `UPDATE family_members` is the `SET removed_at`
  soft-retire — so a future edit that could sever lineage (and reset the alpha budget) fails CI. It
  must **normalize Python SQL string literals** (collapse implicit string concatenation / whitespace,
  case-fold) before matching so a reformatted statement cannot evade the scan (the #277 scanner's AST
  approach, not a naive line grep).

## Quality gate

`uv run pytest -q tests/test_forward_gates.py tests/test_forward_promotion.py tests/test_store.py
<bootstrap test>` for the fast per-task loop (the concurrency-race test lives wherever the existing
`record_gate_with_fdr` concurrency tests do); full `pytest -q && ruff check . && mypy algua &&
lint-imports` at integration. `lint-imports` unaffected: `bootstrap.py` stays a pure leaf; `registry
-> backtest` is an existing allowed edge; `forward_gates.py` gains no new import; the new store method
lives in `store.py` alongside `record_gate_with_fdr_and_maybe_promote` (same layer).
