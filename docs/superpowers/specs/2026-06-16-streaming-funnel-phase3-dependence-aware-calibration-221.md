# Streaming-funnel multiple-testing: Phase 3 — dependence-aware calibration (issue #221)

**Status:** design — Slices 0–5 to build as tracked follow-up PRs; this doc triggers GATE-1 review.
**Date:** 2026-06-16
**Parent:** #211 / umbrella spec `2026-06-15-streaming-funnel-multiple-testing-issue-211-design.md`.
**Phase 1 reference:** `docs/superpowers/plans/2026-06-15-streaming-funnel-dsr-evidence-211.md`
  (merged PR #218, commit `d960655`, schema 24).
**Protected walls touched (eventual implementation):** `algua/research/gates.py`,
  `algua/registry/promotion.py` (both CODEOWNERS `@Lior-Nis`).

## Goal

Make the DSR calibration defensible enough to eventually **retire the haircut**.  The haircut and
the DSR both correct for the *same* best-of-N selection inflation — the haircut is the crude
unit-normal/asymptotic version; the DSR is the better version using measured trial dispersion and
non-normality.  Phase 1 kept both as an AND because two Phase-1 approximations left the DSR
layer unaudited:

1. **Raw-`N` upper bound.** DSR uses `N = effective_funnel_breadth` — the raw rolling-90-day
   combo count — as the trial count in the `SR*` benchmark.  When funnel strategies are correlated
   through the shared holdout, raw `N` overstates independent trials (conservative on `SR*`, safe)
   but is a crude proxy for true multiplicity exposure.
2. **Own-sweep `trial_sr_var` only.**  When a strategy's grid explores near-duplicate parameters
   (low Sharpe dispersion), `trial_sr_var → 0 ⇒ SR* → 0` and DSR collapses to plain PSR with no
   multiplicity penalty.  The haircut covers this gap; the dispersion floor in Phase 3(d) removes it
   without requiring new schema.

Phase 3 adds four components that together make the DSR layer self-sufficient and close both gaps,
enabling a named follow-up PR to safely retire the haircut.

## Invariants carried from Phase 1 (non-negotiable)

Every Phase 3 component must preserve all four Phase-1 disciplines before it can touch the protected
walls.

1. **Tighten-only.**  For every input,
   `new_pass == old_pass AND (NOT new_binding OR new_pass_subcheck)`.
   A new check is *appended to the AND-set when binding* and *omitted entirely otherwise* — the
   `dsr_evidence`/`pit_required` pattern.  Never append `passed=False` for the non-binding case
   (that blocks escape hatches).  Property-tested over a generated grid.

2. **Fail-closed.**  Every pure helper returns `None` on any degenerate or non-finite input.  The
   gate treats `None` as a failed bound check: never a pass, never NaN in the payload.  `to_dict`
   nulls non-finite floats via the existing `_f` helper.

3. **Unit discipline.**  Per-period vs annualized Sharpe is stated at every site.  System Sharpes
   are annualized (`SR_ann = SR_pp · √ANN`); variances scale by `ANN`; correlations are
   dimensionless.  Conversions happen once, inside `gates.py`, co-located with the math.

4. **Architecture boundary.**  `gates.py` stays pure-math reading pre-computed inputs.  All
   persistence, estimation, and bootstrapping live in unprotected `algua/backtest/*` and
   `algua/registry/*`.  `algua/contracts` and `algua/features` remain pure (no I/O).

## Discovered constraint: per-period return streams are currently discarded

`walk_forward` (`algua/backtest/walkforward.py`) computes `returns = pf.returns()` (a `pd.Series`
indexed by bar date), then collapses it to scalar metric dicts via `_segment_record →
metrics_from_returns`.  `WalkForwardResult` stores **only** `window_metrics`, `holdout_metrics`,
`stability` — the raw per-period vector is discarded when `walk_forward` returns.  `sweep()` goes
further: `_evaluate_combo` deliberately drops the per-combo holdout so it never leaves the process
(single-use discipline).

Components **(a)/(b)/(c)** all need persisted per-strategy OOS return streams (cross-strategy
correlation, bootstrap resampling, per-regime segmentation).  Component **(d)** needs none of that.
**Return-stream persistence is the heavy shared prerequisite for a/b/c; (d) is independent and
shippable first.**

## Component (d) — dispersion floor (independent, first, NO migration)

### Problem

The Phase-1 "DSR inputs" section documents this plainly: own-sweep `trial_sr_var` can be *small*
when the grid explores near-duplicate parameters (low Sharpe dispersion), which *shrinks* `SR*` and
makes the DSR layer lenient — in the limit `trial_sr_var → 0 ⇒ SR* → 0 ⇒ DSR collapses to plain
PSR with no multiplicity penalty`.  Phase 1 accepted this because the haircut remains a binding
AND-check and does not depend on `trial_sr_var`.  Phase 3(d) removes the gap by flooring the own
variance from below with the **funnel-wide cross-strategy** trial-Sharpe variance.

### Formula

Phase 1 already records a `(count nᵢ, mean μᵢ, sample-var sᵢ²)` triple per `search_trials` row
(schema 24 columns `trial_sharpe_count/mean/var_ann`), and `store.py:pooled_trial_sharpe_var`
pools a **single strategy's own** rows via the exact pooled-sample-variance formula:

```
M   = Σ(nᵢ·μᵢ) / Σnᵢ
SSE = Σ[ (nᵢ−1)·sᵢ²  +  nᵢ·(μᵢ − M)² ]
pooled_sample_var = SSE / (Σnᵢ − 1)     for Σnᵢ ≥ 2;  0.0 for Σnᵢ == 1
```

The funnel floor uses **per-strategy pooling first, then aggregates across strategies** — not raw
count-weighted pooling across all combos.  This prevents a family that runs many near-duplicate
combos from dominating the floor and gaming it downward:

```
Step 1 — pool each strategy's own rows:
  var_s = pooled_sample_var( search_trials rows for strategy s )      (same formula above)
  Exclude strategies where var_s is None (no finite rows or Σnᵢ < 1).

Step 2 — aggregate over strategies active in the rolling FUNNEL_WINDOW_DAYS window:
  Eligible: any strategy with at least one search_trials row whose trial date falls within
  the window.  Once eligible, ALL of that strategy's own rows (including rows outside the
  window boundary) are pooled in Step 1 to compute var_s.  The window SELECTS STRATEGIES;
  it does NOT slice rows — so a strategy is either fully in or fully out.
  funnel_floor_var_ann = mean( var_s )     for ≥ MIN_FUNNEL_FLOOR_STRATEGIES finite rows;
                         None              otherwise (fail-open)
```

`MIN_FUNNEL_FLOOR_STRATEGIES` is a protected constant (suggested: 5 — need at least a handful of
diverse strategies to form a meaningful floor).  Aggregating by mean of per-strategy variances
gives each strategy equal weight regardless of grid size, preventing combo-count domination by any
one family.

The floor applies as:

```
trial_sr_var_used = max(own_sweep_var, funnel_floor_var)     (both per-period post-/ANN conversion)
```

### Why tighten-only and migration-free

- `max(own, floor) ≥ own` always → `SR*` can only **rise** → DSR confidence can only **fall** →
  the check can only move PASS→FAIL.  Property-testable as
  `new_pass == old_pass AND dsr_pass_with_floor`.
- The `(count, mean, var)` triples exist since schema 24.  The new accessor is a pure read-pool
  over existing rows.  **No schema bump, no migration.**

### Gaming analysis (stated honestly)

A researcher who submits many near-duplicate strategies does NOT lower the floor: each strategy
gets one vote (its per-strategy pooled variance) regardless of how many combos it ran.  To lower
the floor, the adversary would need to register many distinct strategies, each with near-identical
Sharpes — which is exactly the behavior the broader gate already penalizes via the high raw `N`.

If the funnel floor is unavailable (fewer than `MIN_FUNNEL_FLOOR_STRATEGIES` finite strategies),
`trial_sr_var_used = own_sweep_var` — Phase-1 behavior.  The floor can only ever *help* (raise the
bar); its unavailability is conservative.  A persistently unavailable floor (early funnel, sparse
data) means the gate still relies on the haircut as the binding fallback.

**Optional Slice 0 hardening (file as follow-up if audit evidence warrants, not default):**
A single *family* (multiple strategies sharing the same author/lineage) could register many
strategies each with near-zero Sharpe dispersion.  Per-strategy pooling already gives each strategy
one vote regardless of combo count, but does not cap family-level vote weight.  A family-level
dedup (one-strategy-per-family, picking highest per-strategy var) can prevent correlated
near-duplicates from diluting the floor.  This is not included in the base Slice 0 design because
the per-strategy vote already handles the intra-family combo-count gaming; include it only if a
family-flood attack becomes evident in the audit trail.

### Fail-direction rule (consistent)

Rows with NULL/NaN/inf `trial_sharpe_*` stats for a given strategy are treated as that strategy
being excluded from the floor computation (its per-strategy pooled variance returns `None` and it
is dropped from Step 2).  The floor itself is fail-open: if Step 2 has fewer than
`MIN_FUNNEL_FLOOR_STRATEGIES` finite per-strategy variances, `funnel_floor_var` returns `None`,
and the gate falls back to Phase-1 behavior.

### Footprint

- **`algua/registry/store.py`** — new `funnel_trial_sharpe_var(window_days: int) → float | None`
  accessor (per-strategy pooling first, then mean); Protocol declaration in
  `algua/registry/repository.py`.
- **`algua/research/gates.py`** *(PROTECTED)* — `dsr_confidence` gains a `funnel_floor_var_pp:
  float | None` parameter; the `max(own, floor)` is applied before the `SR*` calculation.  New
  protected constants `MIN_FUNNEL_FLOOR_STRATEGIES`.  Reuse `FUNNEL_WINDOW_DAYS` as the window;
  no separate alias needed.
- **`algua/registry/promotion.py`** *(PROTECTED)* — `run_gate` calls
  `repo.funnel_trial_sharpe_var(FUNNEL_WINDOW_DAYS)` and passes the result to `evaluate_gate` /
  `dsr_confidence`.  Audit fields in `GateDecision`: `dsr_funnel_floor_var_ann` (null-coerced),
  `dsr_funnel_floor_n_strategies` (int, for auditability).

### Testing

- Unit test: `max(own, floor)` algebra — floor below own → unchanged; floor above own → raised;
  floor `None` → own unchanged.
- `funnel_trial_sharpe_var` accessor: per-strategy pooling first; a single family with many combos
  gets one vote, not count-weighted domination; fewer than `MIN_FUNNEL_FLOOR_STRATEGIES` strategies
  → `None`; all-NULL strategies → excluded from floor, not propagated.
- Tighten-only property test: for every `(own_var, floor_var)` pair, verify
  `new_pass == old_pass AND dsr_pass_with_floor`.
- Integration: a strategy with a single-combo own sweep (low dispersion, `own_var = 0.0`) in a
  funnel with real dispersion yields a higher `SR*` and tighter gate than in isolation.
- Anti-gaming: a funnel where one family contributes 100 combos and 4 other families contribute 1
  each produces a floor that is the mean of 5 per-strategy variances, not dominated by the large
  family's combo count.

---

## Return-stream persistence (prerequisite for a/b/c)

### Grain: per-strategy-holdout, NOT per-combo

Persisting per-combo OOS vectors would re-open the single-use-holdout best-of-N surface that
`sweep()` is built to prevent: a stored per-combo holdout vector is a selectable surface.  Persist
**exactly one OOS return vector per holdout burn** — the same event that already writes a
`holdout_evaluations` row.

The burn is already a transactional registry write (`reserve_holdout` → `finalize_holdout_reservation`
in `store.py`).  Co-locating the return vector in the same DB transaction means the vector and the
burn token commit or roll back atomically — no orphaned sidecar, no sidecar-without-burn.  Reject
parquet sidecars: they break burn atomicity; a holdout OOS vector is small (~63–500 floats × 8
bytes).

**Atomicity requirement:** the existing standalone `finalize_holdout_reservation` method must be
superseded (for the return-vector path) by a new composite method
`store.finalize_holdout_with_returns(holdout_id, strategy_id, returns, dates, holdout_start,
holdout_end)` that wraps both writes in a **single** `with self._conn:` block:

```python
with self._conn:  # one transaction — both commit or both roll back
    self._conn.execute(
        "UPDATE holdout_evaluations SET committed_at=?, config_hash=? "
        "WHERE id=? AND committed_at IS NULL", (now, config_hash, holdout_id)
    )
    if self._conn.changes() == 0:
        raise AlreadyCommitted(holdout_id)
    self._conn.execute(
        "INSERT INTO holdout_returns (...) VALUES (...)",
        (holdout_id, strategy_id, ...)
    )
```

If the UPDATE matches zero rows (already committed), the INSERT is never issued and the whole
transaction rolls back — leaving `committed_at` NULL and no `holdout_returns` row.

**Write-time validation contracts:**
- Assert `len(returns) == len(dates) == n_bars` before the transaction; mismatch raises immediately.
- Assert that `strategy_id` matches the `holdout_evaluations.strategy_id` for the given
  `holdout_id` (defensive redundancy check — the caller in `promotion.py` should always provide the
  same ID, but silent mismatch would corrupt the sibling-overlap index).

### Schema (schema bump 24 → 25, Slice 1 only)

```sql
CREATE TABLE IF NOT EXISTS holdout_returns (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    holdout_evaluation_id INTEGER NOT NULL REFERENCES holdout_evaluations(id),
    strategy_id           INTEGER NOT NULL REFERENCES strategies(id),
    holdout_start         TEXT    NOT NULL,   -- OOS interval identity (mirrors #192 / #205)
    holdout_end           TEXT    NOT NULL,
    n_bars                INTEGER NOT NULL,   -- length of stored vector; must equal holdout_metrics n_bars
    returns_blob          BLOB    NOT NULL,   -- float64 per-period OOS returns, np.asarray(..., float64).tobytes()
    bar_dates_blob        BLOB    NOT NULL,   -- ISO-8601 bar dates as UTF-8 newline-delimited text for date-aligned joins
    created_at            TEXT    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_holdout_returns_eval ON holdout_returns(holdout_evaluation_id);
CREATE INDEX IF NOT EXISTS ix_holdout_returns_strategy  ON holdout_returns(strategy_id);
CREATE INDEX IF NOT EXISTS ix_holdout_returns_interval  ON holdout_returns(holdout_start, holdout_end);
```

The `FK → holdout_evaluations(id)` ties each return vector to the burn that produced it.  The
`(holdout_start, holdout_end)` index is what (a) and (b) use to locate funnel siblings whose OOS
intervals overlap the strategy under promotion.  Correlation is only meaningful on overlapping
intervals; disjoint OOS windows carry no shared-regime signal.

`bar_dates_blob`: explicit bar dates (UTF-8 newline-delimited ISO strings) rather than
reconstruction from a calendar, because date-aligned inner-joins in (a)/(c) must survive halts,
gaps, and custom universes without a calendar dependency.

### Write path

Surface a new SENSITIVE field `holdout_returns: tuple[list[float], list[str]] | None` on
`WalkForwardResult` — the `(per-period returns, bar dates)` pair.  It inherits
`holdout_metrics`' withhold-from-operators handling; it is more sensitive, not less.  The write
happens in `promotion.py` `run_gate`, inside the same transaction that finalizes the holdout burn.
`gates.py` receives NO return vectors and does NO DB reads — the gate remains pure-math.

**`sweep()` must persist nothing.** The single-use discipline is preserved by persisting only at
the one burn point (`research promote`), never in the grid workers.

### Access-control design (required before Slice 1 ships)

The `holdout_returns` table is SENSITIVE in a stronger sense than `holdout_metrics` scalars: a
researcher who can read back their own burned return vector can identify which specific days their
strategy failed, then tune a subsequent strategy to exploit the same holdout interval.

SQLite provides no row-level access control; the defense is entirely application-layer:

1. **No CLI accessor.**  No existing or future `algua data ...` or `algua registry ...` command
   may expose the `returns_blob` or `bar_dates_blob` for any strategy.
2. **No "get my own vector" API on `StrategyRepository`.**  The store may ONLY expose a
   cross-strategy read: `overlapping_holdout_return_streams(strategy_id, holdout_start, holdout_end,
   window_days) → list[tuple[...]]` — which returns SIBLING vectors (other strategies' returns over
   a partially-overlapping interval), never the requesting strategy's own vector.  This is the only
   method that reads `returns_blob`.
3. **Promotion.py is the only writer and read coordinator.**  `run_gate` in the protected
   `promotion.py` writes the vector and passes sibling vectors to the estimators; it is the sole
   orchestrator.  The gate (`gates.py`) receives pre-computed scalar inputs, never a raw return
   vector.  The `strategy_id` parameter to `overlapping_holdout_return_streams` is trusted to come
   from `promotion.py` only — it is the caller's responsibility to pass the correct ID so the
   self-exclusion guard (sibling-only) works correctly.  No other code path may call this method.
4. **No export.**  `data inspect`, `data verify`, and any future export commands must explicitly
   exclude `holdout_returns` rows/columns.

The protection is correct within the single-owner, single-process model the system assumes.  If
agents were ever given direct SQLite file access, this boundary would need cryptographic hardening —
document that assumption.

### Retention

Permanent, mirroring `holdout_evaluations` and `gate_evaluations`.  Volume: funnel of 1000
strategies × 500 bars × 8 bytes ≈ 4 MB; trivially within SQLite.

### Testing

- Round-trip: vector and dates written at burn time, read back from `holdout_returns` row.
- Atomicity: if `finalize_holdout_with_returns` rolls back (e.g. the UPDATE matches zero rows
  because `committed_at` was already set), `committed_at` remains NULL **and** the
  `holdout_returns` row is absent — both writes are reverted atomically.
- Write is gated on the burn commit — a strategy with no `holdout_evaluations` row has no
  `holdout_returns` row.
- `sweep()` produces no `holdout_returns` rows in isolation.
- Access control: no `StrategyRepository` method returns a strategy's own `returns_blob` to the
  caller; only the sibling-read accessor exists.

---

## Component (a) — effective independent trials `N_eff` (shadow-only in Phase 3)

### Tighten-only constraint — CRITICAL architectural decision

**Component (a) is shadow-only in Phase 3.**  It computes `N_eff` and records it in the
`GateDecision` audit payload, but does **not** use it as the binding trial count in any gate check.
The binding gate continues to use raw `N` in the DSR benchmark (`SR*`), exactly as Phase 1.

**Why shadow-only:** DSR was already a binding AND-check in Phase 1.  If a Phase-1 evaluation
returned FAIL because `DSR(raw_N) = FAIL` (while the haircut passed), switching the binding DSR
to use `N_eff < N` could produce `DSR(N_eff) = PASS` — a FAIL→PASS transition that violates
tighten-only.  The claim that "the haircut still binds and covers this" is insufficient, because
the haircut and the DSR are independent checks: the haircut's binding status does not prevent the
DSR from moving FAIL→PASS when its `N` decreases.

**How N_eff becomes binding:** at Slice 5 (haircut retirement), disabling the haircut AND enabling
the binding `N_eff` in the DSR happen in the **same atomic CODEOWNERS-gated PR**.  The two changes
cancel: the haircut (which used raw `N`) is removed, and the DSR (which now uses `N_eff ≤ N`)
takes full responsibility.  The net effect is a pass-rate change whose direction must be validated
by the dominance audit before the PR can merge.

### Motivation

Raw `N = effective_funnel_breadth` overstates independent trials when funnel strategies are
correlated through the shared holdout (they are — they all trade against the same market regime).
The DSR benchmark `SR*` is monotone increasing in `N`; a more accurate (lower) `N_eff` lowers
`SR*`.  In shadow mode, this is recorded but does not affect gate outcomes.  In the binding
retirement slice, the lower `SR*` from `N_eff` replaces the haircut's penalty, giving a calibrated
rather than crude correction.

### Estimator: Kish average-pairwise-correlation form

```
N_eff = N / (1 + (N − 1)·ρ̄_lower)
```

- `ρ̄_lower` — a **conservative (lower-bound)** estimate of the mean off-diagonal pairwise Pearson
  correlation of funnel siblings' date-aligned overlapping-OOS return streams.  "Lower-bound" means
  uncertainty in ρ̄ pushes `N_eff` **up toward `N`** (conservative when uncertain), not down.
  Concretely: `ρ̄_lower = max(0, ρ̄_sample − k · SE(ρ̄))`, where `SE(ρ̄) = σ_ρ / √M` (σ_ρ the
  sample std-dev of off-diagonal pairwise correlations, M the number of pairs) and `k` is a
  protected constant `RHO_BAR_SHRINKAGE_K`.  Clamp `ρ̄_lower ∈ [0, 1]`.

  **Note:** `SE = σ_ρ/√M` treats pairwise correlations as independent, which understates
  uncertainty when many strategies share the same dates.  This means the lower bound may be
  tighter than warranted — i.e., `N_eff` may be lower (more lenient) than a proper block-bootstrap
  of the correlation matrix would give.  This is acceptable in shadow mode (no gate effect) and is
  flagged as **Q2.2** for GATE-1 to decide whether to use Fisher-z CIs or bootstrap-the-matrix
  before the binding slice ships.

- **Why Kish over alternatives:** monotone in ρ̄ (recovers raw `N` at ρ̄=0; recovers N_eff=1 at
  ρ̄=1), interpretable, no eigensolve, no clustering hyperparameters.  The eigenvalue participation
  ratio (`N_eff = (Σλ)²/Σλ²`) is more structure-aware but sensitive to noisy small-sample
  eigenvalues on short overlapping holdouts and harder to guard/audit.  The López de Prado ONC
  clustering method introduces hyperparameters and nondeterminism.  Both remain valid audit-only
  cross-checks.

### Guards (in shadow mode, all still apply for audit correctness)

1. **Cap at raw N:** `N_eff_int = max(1, min(N, round(N_eff)))`.
2. **Conservative ρ̄ shrinkage:** stated above — estimation noise pushes `N_eff` up.
3. **Fall-back to raw N in the shadow field:** if fewer than `MIN_N_EFF_SIBLINGS` overlapping-OOS
   siblings exist, or any pair has fewer than `MIN_CORR_OVERLAP_BARS` shared bars, or any
   correlation is non-finite, the estimator records `dsr_n_eff = None` in the audit payload
   (raw `N` used as always, since (a) is shadow-only).

Sibling set: funnel strategies whose `holdout_returns` interval overlaps the OOS interval of the
strategy under promotion, queried within the same rolling `FUNNEL_WINDOW_DAYS` window.  Inner-join
on `bar_dates` before computing pairwise correlations.

### Footprint

- **`algua/registry/store.py`** — `overlapping_holdout_return_streams(strategy_id, holdout_start,
  holdout_end, window_days) → list[tuple[list[float], list[str]]]` — returns date-aligned sibling
  vectors (NEVER the requesting strategy's own vector — see access-control design above).
- **`algua/research/gates.py`** *(PROTECTED)* — `estimate_n_eff(raw_n, siblings, ...)` helper
  that returns `int | None`; new protected constants `MIN_N_EFF_SIBLINGS`, `MIN_CORR_OVERLAP_BARS`,
  `RHO_BAR_SHRINKAGE_K`.  `GateDecision` gains audit fields `dsr_n_eff`, `dsr_rho_bar`,
  `dsr_n_siblings` — shadow-only, never fed into the binding `dsr_confidence` call.  The binding
  `dsr_confidence` continues to receive raw `N` as `n_trials`.  The shadow estimation site must
  include a guard comment:
  ```python
  # N_eff is shadow-only until HAIRCUT_RETIRED — never pass it as n_trials.
  ```
  This prevents an accidental wiring mistake in a future edit from silently changing gate behavior.
- **`algua/registry/promotion.py`** *(PROTECTED)* — queries siblings, calls the estimator, records
  `n_eff` in the audit payload; does NOT pass it as the binding trial count.

### Testing

- Shadow-only property: `dsr_evidence` (the binding check) is byte-for-byte identical to Phase 1;
  `dsr_n_eff` is a new audit field that does not affect `passed`.
- Estimator unit: `ρ̄=0 ⇒ N_eff=N`; `ρ̄=1 ⇒ N_eff=1`; shrinkage → N_eff closer to N than raw mean
  correlation; fewer than `MIN_N_EFF_SIBLINGS` siblings → audit field `None`.
- Tighten-only property test (Phase 3 with shadow (a)): `new_pass == old_pass AND new_binding_checks_pass`.
  Since (a) adds no binding check, this reduces to `new_pass == old_pass AND (b_pass AND c_pass AND d_pass)`.
- Access control: `overlapping_holdout_return_streams` never returns the requesting strategy's own
  vector (test with a singleton funnel where the strategy is its own sibling — must return empty).

---

## Component (b) — serial-dependence bootstrap of DSR confidence

### Scope (stated precisely)

The stationary bootstrap in component (b) addresses **intra-strategy autocorrelation** — it widens
the effective standard error of the DSR confidence estimate when a single strategy's OOS return
stream is serially correlated (momentum, mean-reversion, regime clustering within the holdout).
It does **not** calibrate the distribution of the selected maximum across correlated sibling
strategies — that is the shared-holdout dependence problem from the umbrella spec, which is
partially addressed by component (a) (effective-N) and component (c) (multi-regime).  A
cross-strategy block-bootstrap (resampling the joint sibling return matrix by date blocks) would
address the inter-strategy dependence more directly and is a candidate for a future phase.

### Design: stationary bootstrap (Politis–Romano 1994), AND cross-check

**Stationary over circular block.**  Circular block bootstrap fixes block length `L`, introducing
a wrap-around artifact and a tunable that biases the autocorrelation estimate.  The stationary
bootstrap draws geometrically-distributed block lengths (random, mean `1/p`), keeping the
resampled series stationary.  This is the standard choice for Sharpe-ratio inference under serial
dependence and requires one parameter (expected block length, set via the Politis–White automatic
block-length selector).

**AND cross-check, not a replacement.**  The bootstrap check `dsr_bootstrap` passes iff the
bootstrap-lower confidence ≥ `1 − DSR_ALPHA`, **alongside** (not instead of) the closed-form
`dsr_evidence`.  Replacing the closed-form would change Phase-1 behavior in both directions,
violating tighten-only.  An AND cross-check is monotone-tightening: a strategy must clear *both*
the closed-form and the bootstrap-robust bar.  When autocorrelation is benign, the bootstrap-lower
≈ closed-form and the cross-check is redundant (no harm).  When autocorrelation inflates the naive
Sharpe SE, the bootstrap catches it.

**What gets bootstrapped:** the DSR confidence itself.  Resample the OOS return vector
(block-stationary) `B = DSR_BOOTSTRAP_RESAMPLES` times; on each resample recompute `SR_obs`,
moments `γ₃, γ₄`, and the per-period DSR confidence against the same `SR*`.  Take the
`DSR_BOOTSTRAP_LOWER_PCT`th percentile (e.g. 5th) of the bootstrap distribution as
`bootstrap_lower_confidence`.  This estimates the sampling distribution of DSR confidence under
serial dependence and uses a lower percentile as a conservative cross-check — a lower percentile
means MORE resampled copies had lower confidence, indicating the closed-form overstated certainty.

**Deterministic seeding (the repo bans nondeterminism).**  The RNG is seeded from a stable hash
of `(strategy_name, holdout_start, holdout_end, config_hash)`.  Note: `strategy_name` is used
rather than `strategy_id` because auto-increment IDs are unstable across database deletions and
re-registrations; the name is the durable identity.  This is assembled in `promotion.py`
(unprotected precompute); the hash is deterministic from persisted DB fields.  The seed and `B` are
persisted in the `dsr_bootstrap` audit fields so the result is reproducible from the payload alone.
`gates.py` receives the already-computed `bootstrap_lower_confidence` as a pre-computed scalar
(the gate does no resampling — it remains pure-math).

**Cost:** `B` resamples × (Sharpe + 4 moments) on ≤500 floats is microseconds × `B`; `B = 2000`
is well under one second per promote.

### Footprint

- **`algua/backtest/bootstrap.py`** *(new, unprotected)* — `stationary_bootstrap_dsr(returns,
  dates, sr_star, dsr_alpha, b, seed, block_len_auto=True, block_len_override=None) →
  BootstrapResult(lower_confidence, seed_used, b_used, block_len)`.
- **`algua/research/gates.py`** *(PROTECTED)* — new `dsr_bootstrap_evidence` check appended to
  the AND-set when `bootstrap_lower_confidence` is not None; new protected constants
  `DSR_BOOTSTRAP_RESAMPLES`, `DSR_BOOTSTRAP_LOWER_PCT`; audit fields `dsr_bootstrap_lower`,
  `dsr_bootstrap_seed`, `dsr_bootstrap_b`, `dsr_bootstrap_block_len`.
- **`algua/registry/promotion.py`** *(PROTECTED)* — assembles the seed, calls the bootstrap
  helper, passes the scalar to `evaluate_gate`.

### Testing

- Bootstrap unit: on a white-noise return series, bootstrap-lower ≈ closed-form (no
  autocorrelation → no widening); on a strongly autocorrelated AR(1) series, bootstrap-lower <
  closed-form (widened effective SE → tighter gate); seed reproducibility (same inputs → same
  output).
- Tighten-only: `new_pass == old_pass AND (NOT bootstrap_binding OR bootstrap_pass)`.
- Block length: automatic selector produces a finite positive value; override path works.
- Degenerate: `T ≤ 1` or any non-finite input → `bootstrap_lower_confidence = None`.  If
  `dsr_evidence` is currently binding: append a FAILED `dsr_bootstrap` check (mirrors the
  fail-closed invariant — degenerate input on an active binding check is a failure, not a silent
  omission).  If `dsr_evidence` is non-binding: omit the bootstrap check entirely (consistent with
  the existing `dsr_evidence` non-binding pattern).
- Non-finite resamples: any resample yielding a non-finite Sharpe or moments is excluded from the
  percentile distribution; if fewer than `B / 2` resamples are finite, return `None` (degenerate
  path above applies).

---

## Component (c) — multi-regime robustness

### Motivation

A single aggregate holdout p-value (or DSR confidence) can pass because one benign regime carries
the average over a bad regime.  Phase 3(c) requires the strategy to clear a (deliberately relaxed)
per-regime bar in **each** sufficiently-long regime, composed as a tighten-only AND.  The aggregate
`holdout_sharpe` check from Phase 1 is **untouched**.

### Regime definition: market-volatility tertiles

Regimes are defined by **market-volatility tertiles** of the OOS interval, computed from a
*market/benchmark* return stream — NOT from the strategy's own returns (labeling by own returns
would be circular and gameable).  `N_REGIMES = 3` (low/medium/high vol), date-aligned to the
strategy's OOS vector via `bar_dates`.  Three regimes balances statistical power (each ≥ ~21 bars
at the `MIN_HOLDOUT_OBSERVATIONS = 63` floor) against meaningful regime diversity.

**Tie-breaking (deterministic):** if vol values are not distinct enough to produce three non-empty
tertiles, use rank-based assignment with ties broken by index position (earliest date wins the
lower tertile).  This is deterministic and reproducible.

**Constant-vol case:** if the vol series is nearly constant (all bars assigned to one tertile), no
second powered regime can survive → `< 2 surviving regimes` → fail closed (see below).  Fail
closed is the correct behavior here: a strategy with a nearly-constant-vol holdout cannot be
assessed for multi-regime robustness.

**Calendar-split fallback — EXPLICITLY NON-BINDING:** when no market-vol series is available (or
the vol series has insufficient overlap), the `regime_robustness` check is **omitted entirely**.
A calendar split provides temporal diversity, not market-state diversity; calendar-split evidence
does NOT count as multi-regime robustness.  The binding check requires a vol series.  This is
recorded as `regime_method = "unavailable"` and `regime_robustness_binding = false` in the audit
payload.  Omitting the check when vol is unavailable is NOT a binding fallback — the check simply
does not appear in the AND-set for that evaluation.  The benchmark vol series must be a
reproducible, PIT-compliant data source (same ingested universe bars available at promote time,
not a live pull — **Q4.2** must be resolved before Slice 4 ships).

### `regime_robustness` check

The check is only entered when a vol series is available (see calendar-split fallback above).

For each of the `N_REGIMES` regimes:
- If the regime has `< MIN_REGIME_OBSERVATIONS` bars: the regime is **underpowered** and dropped.
- If `< 2` regimes survive after dropping: `regime_robustness` is **appended as a FAILED check**
  (cannot establish multi-regime evidence — includes the constant-vol case above).  The check is
  NEVER omitted when vol is available; it is always appended, and <2 survivors = FAILED.
- Otherwise: require per-regime Sharpe ≥ `MIN_REGIME_SHARPE` (a deliberately lenient floor, lower
  than the aggregate `min_holdout_sharpe`).

`regime_robustness` passes iff every surviving regime clears `MIN_REGIME_SHARPE`.

**Tighten-only:** adding per-regime floors can only subtract passes from the aggregate gate, never
add — the aggregate `holdout_sharpe` check is byte-for-byte unchanged.  For any input when vol is
available, the new overall verdict is `old_pass AND regime_robustness_pass`.

**Short-regime leniency (stated explicitly):** dropping underpowered regimes avoids punishing a
strategy merely for a short holdout.  The gate still requires ≥ 2 surviving regimes, so it always
reflects at least two distinct market conditions when it binds.

### Footprint

- **`algua/registry/store.py`** — `market_returns_for_interval(holdout_start, holdout_end)` accessor
  pulling the benchmark vol series from existing ingested snapshots (see Q4.2 for source — must be
  a PIT-compliant snapshot, not a live pull).
- **`algua/research/gates.py`** *(PROTECTED)* — pure `regime_splits(returns, dates, market_returns,
  n_regimes) → list[RegimeSlice]` helper; `regime_robustness_check(slices, min_obs, min_sharpe)
  → RegimeRobustnessResult`; new check `regime_robustness` appended to the AND-set when the vol
  series is available (always appended; <2 survivors = FAILED rather than omitted); protected
  constants `N_REGIMES`, `MIN_REGIME_OBSERVATIONS`,
  `MIN_REGIME_SHARPE`; audit fields `regime_method`, `n_regimes_attempted`, `n_regimes_surviving`,
  `per_regime_sharpes` (list, null-coerced), `regime_robustness_binding`.
- **`algua/registry/promotion.py`** *(PROTECTED)* — fetches the market-vol series, passes it
  alongside the holdout OOS vector to `evaluate_gate`.

### Testing

- Regime split: vol-tertile labels correct on a synthetic return series with a known vol sequence;
  inner-join on dates handles gaps; tie-breaking is deterministic (test with repeated vol values).
- Constant-vol: single-tertile assignment → <2 survivors → fail closed.
- Vol unavailable: `regime_robustness` omitted entirely (not replaced by calendar split);
  `regime_method = "unavailable"`, `regime_robustness_binding = false` in payload.
- Per-regime Sharpe computation: uses `metrics_from_returns` on the regime sub-vector — consistent
  with Phase-1 moment computation.
- Short-regime policy: underpowered regime dropped; `< 2` survivors → FAILED check (not omitted —
  check is always appended when vol available); `≥ 2` → tighten-only AND on surviving regimes.
- Tighten-only: `new_pass == old_pass AND (NOT regime_binding OR regime_pass)` where
  `regime_binding = (vol series available)` and `regime_pass = False` when <2 survivors.
- Integration: a strategy that passes on the aggregate but fails in the high-vol regime is blocked;
  a strategy that passes in all regimes is unaffected.

---

## End-state (e) — retire the haircut (+ enable binding N_eff)

### What this slice does

This is an **explicit weakening move**, not a tighten-only slice.  It bundles two changes in a
single atomic CODEOWNERS-gated PR: (1) disable the haircut from the binding AND-set, and (2)
enable binding `N_eff` in the DSR benchmark.  **Both changes weaken the gate — they do NOT cancel
or offset each other in a provable sense:** removing the haircut reduces one constraint; enabling
the calibrated `N_eff ≤ N` reduces another.  The dominance audit (Q6.1) must validate that the net
pass-rate effect is acceptable before this PR merges.  Calling them "partially offsetting" or
"canceling" is misleading and must not appear in the audit rationale.

### Gate conditions

The haircut cannot be retired until:

1. **All of d/b/c/a (Slices 0–4) are live and have accumulated real-traffic evidence** — the
   DSR + bootstrap + regime AND-set is audited.
2. **Phase-2 FDR (LORD++) is live** — the online alpha-wealth ledger governs stream-level
   multiplicity.
3. **Dominance audit passes on DSR(raw_N) baseline** — over the persisted `gate_evaluations`
   decision trail (using the DSR baseline with raw `N`, NOT with `N_eff`, because (a) is
   shadow-only until this slice), no case of `haircut_fail AND dsr_full_pass` is observed.  The
   audit MUST use DSR(raw_N) because that is the binding baseline; auditing against DSR(N_eff)
   would measure the lenient version and understate the haircut's unique contribution.  A minimum
   evaluation window and minimum promote count must be **predeclared** (not decided post-hoc when
   data looks favorable) — see **Q6.1**.
4. **Calibration audit passes** — the bootstrap (b) nulls are validated; the shadow `N_eff` audit
   trail shows `N_eff < N` only when there is real correlation evidence among funnel siblings.

### Staging on the protected wall

1. **Shadow phase (in Slice 4 or earlier).** Keep the haircut binding.  Add non-binding audit field
   `haircut_would_have_blocked: bool` to every `decision_json` so the dominance audit can run on
   real traffic.
2. **Flagged cutover (Slice 5).** Introduce a protected named constant `HAIRCUT_RETIRED: bool =
   False` in `gates.py`.  When True, the haircut is dropped from the binding AND-set and the
   binding trial count for DSR switches from raw `N` to `N_eff`.  The haircut is still **computed
   and recorded** in the payload (reversible with one flag flip).  Flipping the flag is an explicit
   CODEOWNERS-gated edit — never an agent-tunable knob.
3. **Reversibility.** Because the haircut math stays in the code, reverting is a one-line flip.
   `effective_min_holdout_sharpe` keeps reporting the haircut value for continuity.

This is the **only** Phase-3 change that can weaken the gate.  It ships last, behind a flag,
with a documented dominance-audit gate and a one-flip rollback.

### Recommended filing

File Slice 5 as a **separate post-Phase-2 issue** (not a TDD plan in this spec) to enforce
the dependency on Phase 2 explicitly.  The dominance audit and calibration audit can only run after
all prior slices are live for enough traffic.

---

## PR slicing & dependency graph

Each slice is its own separately approved PR with its own TDD plan.  The graph:

```
Slice 0  (d) Dispersion floor
  ├─ Independent — NO persistence, NO migration.
  ├─ Per-strategy pooling first (anti-gaming), then mean across strategies.
  └─ Purely tightening. SHIP FIRST.

Slice 1  Return-stream persistence (schema 24→25)
  ├─ WalkForwardResult.holdout_returns (SENSITIVE) + holdout_returns table.
  ├─ Access-control design enforced (no "get my own vector" API).
  ├─ NO gate-behavior change — pure plumbing.
  └─ All of Slices 2/3/4 depend on this.
      │
      ├─ Slice 2  (b) Serial-dependence bootstrap (AND cross-check)
      │     ├─ Addresses intra-strategy autocorrelation (not cross-strategy dependence).
      │     ├─ Tightening-only AND check.
      │     └─ Independent of Slices 3 & 4.
      │
      ├─ Slice 3  (a) N_eff shadow recording
      │     ├─ SHADOW-ONLY: records dsr_n_eff in audit payload; no gate-behavior change.
      │     ├─ N_eff becomes binding only in Slice 5 (bundled with haircut disabling).
      │     └─ Independent of Slices 2 & 4.
      │
      └─ Slice 4  (c) Multi-regime robustness
            ├─ Vol-tertile AND floors; deterministic tie-breaking; constant-vol → fail closed.
            ├─ Tightening-only AND check.
            ├─ Independent of Slices 2 & 3.
            └─ BLOCKING PREDECLARATION before Slice 4 merges: DOMINANCE_AUDIT_MIN_PROMOTIONS,
               DOMINANCE_AUDIT_MIN_WINDOW_DAYS, and DOMINANCE_AUDIT_ZERO_HAIRCUT_EXCEPTIONS must
               be committed as CODEOWNERS-protected constants in gates.py before Slice 4 ships
               the haircut_would_have_blocked shadow field.  Post-hoc threshold selection (after
               seeing the data) invalidates the dominance audit.  See Q6.1.

Slice 5  (e) Haircut retirement + binding N_eff (ATOMIC)
  ├─ LAST — gates on Slices 0–4 live + Phase-2 FDR live + dominance audit (raw-N baseline).
  ├─ ONLY weakening change; ships as a bundle (haircut off + N_eff binding = net-calibrated).
  └─ Filed as a separate post-Phase-2 issue (dependency on Phase 2 explicitly enforced).
```

Slices 2/3/4 are **mutually independent** given Slice 1 and can be developed in parallel.
Slice 3 is shadow-only so it can merge before the haircut retires without any tighten-only risk.

---

## Named protected constants

| Constant | Slice | Location | Role |
|---|---|---|---|
| `FUNNEL_WINDOW_DAYS` | already exists | `gates.py` | reused as-is for funnel-floor window |
| `MIN_FUNNEL_FLOOR_STRATEGIES` | (d) | `gates.py` | min per-strategy variances for funnel floor |
| `MIN_CORR_OVERLAP_BARS` | (a) | `gates.py` | min shared OOS bars per pair to estimate ρ̄ |
| `MIN_N_EFF_SIBLINGS` | (a) | `gates.py` | min overlapping siblings to attempt N_eff |
| `RHO_BAR_SHRINKAGE_K` | (a) | `gates.py` | SE multiplier for conservative (lower-bound) ρ̄ |
| `DSR_BOOTSTRAP_RESAMPLES` (B) | (b) | `gates.py` | resample count |
| `DSR_BOOTSTRAP_LOWER_PCT` | (b) | `gates.py` | lower percentile of bootstrap confidence distribution |
| `N_REGIMES` | (c) | `gates.py` | regime bucket count (= 3) |
| `MIN_REGIME_OBSERVATIONS` | (c) | `gates.py` | per-regime power floor |
| `MIN_REGIME_SHARPE` | (c) | `gates.py` | relaxed per-regime Sharpe bar |
| `HAIRCUT_RETIRED` | (e) | `gates.py` | binding-membership toggle (also switches N to N_eff); default False |
| `DOMINANCE_AUDIT_MIN_PROMOTIONS` | (e) | `gates.py` | min promotions evaluated before retirement audit can pass; suggested 30; must be predeclared before Slice 4 ships |
| `DOMINANCE_AUDIT_MIN_WINDOW_DAYS` | (e) | `gates.py` | min calendar days of evaluation data before retirement audit; suggested 90; predeclared before Slice 4 |
| `DOMINANCE_AUDIT_ZERO_HAIRCUT_EXCEPTIONS` | (e) | `gates.py` | allowed `haircut_fail AND dsr_raw_N_pass` cases (suggested 0); predeclared before Slice 4 |

Schema bump: **24 → 25** in Slice 1 (`holdout_returns` table).  Slice 0 requires **no bump**.

## Critical files

- `algua/research/gates.py` (PROTECTED) — eventual home of all Phase-3 protected constants and
  gate checks; (d) floor, shadow (a) N_eff audit, (b) bootstrap cross-check, (c) regime AND
  floors, (e) retirement flag.
- `algua/registry/promotion.py` (PROTECTED) — burn orchestrator; assembles pre-computed Phase-3
  inputs and writes the OOS vector in the burn transaction.
- `algua/backtest/walkforward.py` — surfaces the SENSITIVE `holdout_returns` vector (Slice 1).
- `algua/registry/db.py` — schema 24→25 `holdout_returns` table (Slice 1 only; Slice 0 unchanged).
- `algua/registry/store.py` — funnel-wide per-strategy pooled-variance floor accessor (Slice 0);
  `holdout_returns` write + sibling-only cross-strategy read (Slices 1–4).
- `algua/backtest/sweep.py` — read-only reference; the `(count, mean, var)` triples feeding (d).
- `algua/backtest/bootstrap.py` — new unprotected serial-dependence bootstrap helper (Slice 2).

## Quality gate (per slice)

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

## GATE-1 decision list

Open questions to be resolved in the GATE-1 review round before per-slice TDD plans are written.

**Persistence (Q1.x)**
- **Q1.1** Confirm per-strategy-holdout grain (not per-combo, not per-window) for (a)/(b)/(c).
- **Q1.2** BLOB float64 + newline-delimited ISO date strings — confirm encoding/round-trip
  convention.
- **Q1.3** Confirm `bar_dates_blob` (explicit dates) rather than calendar reconstruction.
- **Q1.4** Is the access-control design (sibling-only read, no CLI accessor, no "get own vector"
  API) sufficient given the single-owner, single-process model?  If agent direct-DB access is ever
  added, what additional hardening is required?

**Component (a) — effective N (Q2.x)**
- **Q2.1** Shadow-only confirmed for Slice 3; binding switch happens at Slice 5 (retirement)?
- **Q2.2** ρ̄ SE calculation: `SE = σ_ρ/√M` treats pairwise correlations as independent (acceptable
  for shadow mode?) vs Fisher-z CI with effective sample size based on overlap length.  Pin which
  to use before Slice 5 where (a) becomes binding.
- **Q2.3** Sibling set = funnel strategies with overlapping OOS intervals in the rolling
  `FUNNEL_WINDOW_DAYS` window — confirm.
- **Q2.4** `MIN_N_EFF_SIBLINGS = 2` minimum?  Note: 2 is the minimum to compute any pairwise
  correlation; a larger minimum (e.g. 5) would give a more reliable estimate.

**Component (b) — bootstrap (Q3.x)**
- **Q3.1** Stationary bootstrap (Politis–Romano) confirmed over circular block?
- **Q3.2** Scope is intra-strategy serial dependence — confirmed, NOT a cross-strategy dependence
  fix?
- **Q3.3** Block length: automatic Politis–White selector with a floor/cap, audited in payload?
- **Q3.4** Pin `DSR_BOOTSTRAP_RESAMPLES` (suggested: 2000) and `DSR_BOOTSTRAP_LOWER_PCT` (suggested:
  0.05, i.e. 5th percentile).
- **Q3.5** Seed derivation: `hash((strategy_id, holdout_start, holdout_end, config_hash))` —
  confirm inputs are stable and available at promote time.

**Component (c) — multi-regime (Q4.x)**
- **Q4.1** Vol-tertile labeling by *market/benchmark* series (not strategy's own returns) confirmed?
  Calendar-split as fallback confirmed?
- **Q4.2** (**Blocking for Slice 4 TDD plan**) Benchmark vol series source: which universe?
  Suggested: equal-weighted cross-sectional daily return vol of the strategy's own `--universe`
  bars already ingested.  Must be a PIT-compliant reproducible snapshot (provenance fields to
  record in audit payload).
- **Q4.3** Short-regime policy: drop-underpowered-and-require-≥2-survivors confirmed?
- **Q4.4** Pin `N_REGIMES` (= 3), `MIN_REGIME_OBSERVATIONS` (suggested: 21), `MIN_REGIME_SHARPE`
  (suggested: 0.0 — require non-negative per-regime return, a lenient floor).

**Component (d) — dispersion floor (Q5.x)**
- **Q5.1** Per-strategy pooling first (anti-gaming), then mean of strategy-level variances —
  confirmed over raw count-weighted pooling?
- **Q5.2** `MIN_FUNNEL_FLOOR_STRATEGIES` value (suggested: 5)?
- **Q5.3** Floor confined to DSR `trial_sr_var` only (not applied to the haircut) — confirm.

**End-state / retirement (Q6.x)**
- **Q6.1** (**BLOCKING for Slice 4 — predeclare before shadow data collection begins**)  The
  dominance audit threshold constants (`DOMINANCE_AUDIT_MIN_PROMOTIONS`,
  `DOMINANCE_AUDIT_MIN_WINDOW_DAYS`, `DOMINANCE_AUDIT_ZERO_HAIRCUT_EXCEPTIONS`) must be committed
  as CODEOWNERS-protected constants in `gates.py` **before** Slice 4 ships the
  `haircut_would_have_blocked` shadow field that starts collecting live traffic.  Post-hoc selection
  of the threshold after seeing the data invalidates the audit.  Suggested values: 30 promotions,
  90 calendar days, zero `haircut_fail AND dsr_raw_N_pass` cases.  See Named constants table above.
- **Q6.2** Confirm Slice 5 filed as a separate post-Phase-2 issue (not a TDD plan in this spec)?

## Caveat (inherited from the umbrella spec)

FDR here governs **research-discovery quality** and, given shared-holdout dependence, is an
**operating target, not a proof**.  `candidate` is not capital; the live wall and forward-test
certificate remain the hard guards.
