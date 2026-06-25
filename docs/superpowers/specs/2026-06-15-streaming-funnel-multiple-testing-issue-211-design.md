# Streaming-funnel multiple-testing: two-layer gate (issue #211)

**Status:** design — Phase 1 to build; Phases 2–4 designed as tracked follow-ups.
**Date:** 2026-06-15
**Protected walls touched (Phase 1):** `algua/research/gates.py`, `algua/registry/promotion.py` (both CODEOWNERS `@Lior-Nis`).

## Problem

The funnel's goal is to stream **1000+ diverse-thesis** hypotheses through `research promote`
and trade only the most robust survivors over ~30 years. Today's gate uses a rolling-window
**deflated-Sharpe haircut** (`sharpe_haircut`, `effective_funnel_breadth` in
`algua/research/gates.py`): the holdout-Sharpe bar is raised by `√(2·ln N)·√ANN/√T` where `N`
is the rolling-90-day funnel-wide combo count and `T` the holdout length. Two scale problems:

1. **Order-dependence.** Because `N` grows over a rolling window, an identical strategy faces a
   higher bar at `N=500` than at `N=50` — last-tested pays most.
2. **Shared-holdout dependence.** Every hypothesis reuses the same holdout and market regimes, so
   the trials are correlated. Textbook multiplicity guarantees (FWER/FDR) do **not** hold cleanly.

Codex design consult (multi-model) confirmed the framing. **Key correction we adopt verbatim:**
the current haircut is a *deflated-Sharpe multiplicity heuristic, NOT formal FWER* — we do not
relabel it and do not rip it out. And because trials are correlated through a shared holdout,
**neither FWER nor FDR holds cleanly; nominal FDR is an operating target, not a guarantee.**
Benjamini–Yekutieli (valid under arbitrary dependence) is the wrong patch — brutally conservative,
ignores the structure.

## Scope decision

This is ~4–6 PRs across two protected walls. We write **one umbrella spec** (this document)
covering all four phases, **build Phase 1 now**, and file Phases 2–4 as tracked follow-up issues
straight from the "Deferred phases" section. `candidate` is not capital — downstream
paper / forward_tested / live gates remain — so the research-discovery gate may be conservative.

## Phase 1 — calibrated per-strategy DSR evidence (BUILD NOW)

### Goal

Add a calibrated per-strategy evidence layer — a PSR/DSR p-value (Bailey & López de Prado, 2014)
— as an **additional binding AND-check** alongside the existing haircut. It can only ever move a
PASS to FAIL, never the reverse (tighten-only), so it is safe to merge to a protected wall and
cannot weaken the gate. The existing haircut-deflated Sharpe check is left **exactly as is** and
serves as the conservative fallback.

### The statistic

**PSR (Probabilistic Sharpe Ratio)** — the *confidence* (a probability in `[0,1]`, **not** a
p-value) that the true Sharpe exceeds a benchmark `SR*`, adjusting for sample length and
non-normality of the return stream (Bailey & López de Prado 2012):

```
PSR(SR*) = Φ( (SR_obs − SR*) · √(T − 1) / √(1 − γ₃·SR_obs + ((γ₄ − 1)/4)·SR_obs²) )
```

- `SR_obs` — observed holdout Sharpe (**per-period**, i.e. the annualized holdout Sharpe ÷ √ANN).
- `T` — the number of **finite holdout returns actually used to estimate** `SR_obs`, `γ₃`, `γ₄`.
  This must be the *same* sample the moments are computed on — see the data-path note below; it is
  NOT independently re-derived from a bar count.
- `γ₃` — skewness of the holdout return stream.
- `γ₄` — **raw (Pearson) kurtosis** of the holdout return stream (`= 4` for a fat-tailed series;
  `= 3` for Gaussian). NOT excess kurtosis. The `(γ₄ − 1)/4` term assumes raw kurtosis — for a
  Gaussian series it must reduce to `(3−1)/4 = 0.5`, giving the Lo/Mertens SR-estimator variance
  `1 + SR²/2`. `scipy.stats.kurtosis(..., fisher=False)` (or `excess + 3`) — pinned by a test on a
  normal-like series expecting `γ₃≈0`, `γ₄≈3`.
- `√(T − 1)` — conservative small-sample form (matches the López de Prado reference
  implementations of PSR; the asymptotic form uses `√T`, a <1% difference at `T ≥ 30` and in the
  safe/stricter direction). Kept as-is; do not claim a specific equation number.
- `Φ` — standard-normal CDF (`scipy.stats.norm.cdf`).

**DSR (Deflated Sharpe Ratio)** — PSR where the benchmark `SR*` is the **expected maximum Sharpe
under N trials** rather than 0 (Bailey & López de Prado 2014, eq. 17):

```
SR* = √(trial_sr_var) · [ (1 − γ_E)·Z⁻¹(1 − 1/N) + γ_E·Z⁻¹(1 − 1/(N·e)) ]      for N > 1
SR* = 0                                                                          for N ≤ 1
```

- `γ_E` — the **Euler–Mascheroni constant** `0.5772156649015329` (a protected named constant
  `EULER_MASCHERONI` in `gates.py`). NOT `e⁻¹`: `e⁻¹≈0.368` systematically *understates* the
  expected max → too-low `SR*` → too-lenient gate. (Numerically, for N=100 the γ_E form gives
  E[max]≈2.53 vs the true 2.51; the `e⁻¹` form gives 2.46.)
- **Guard ordering (precise, to resolve the `N=1` vs `N<1` overlap):** evaluate in this order —
  `if N < 1: return None` (invalid breadth → fail closed); `elif N <= 1: SR* = 0` (collapse to PSR,
  before any `Z⁻¹`, since the formula yields `Z⁻¹(0) = −∞` at `N=1`); `else:` the formula. So
  `N=0` fails closed and is never silently turned into plain PSR.
- `N` — trial count = `effective_funnel_breadth(own_lifetime, windowed_total)` (the **same** `N`
  the haircut uses; a raw count).
- `trial_sr_var` — **variance of the per-combo trial Sharpe ratios** across the strategy's
  `backtest sweep`(s); the cross-trial dispersion Bailey–LdP use (see "DSR inputs" for how it is
  recorded and pooled).
- `Z⁻¹` — inverse standard-normal CDF (`scipy.stats.norm.ppf`).

**Unit discipline (critical, mirrors the haircut's `√ANN` handling).** The formulae are in
**per-period** Sharpe units; the system's Sharpes (both `SR_obs` and the sweep's per-combo Sharpes)
are **annualized** (`SR_ann = SR_per_period · √ANN`). Conversion happens at the point of use inside
`gates.py`: `SR_obs → SR_obs/√ANN`, `trial_sr_var → trial_sr_var/ANN` (variance scales by the
square). The DB column stores the variance of the **annualized** sweep Sharpes (name encodes it:
`trial_sharpe_var_ann`); the single `/ANN` conversion lives in the protected gate so all unit
handling is co-located with the math. **Fixed-`ANN` assumption:** `ANN` is the single global
constant (`algua.backtest._constants.ANN`) the haircut already uses; because both `SR_obs` and the
sweep Sharpes are annualized by the *same* constant, the `/ANN` conversion is internally consistent
even if a strategy was swept on a non-daily timeframe — the per-period units cancel. A future
per-timeframe `ANN` (intraday) would require persisting the sweep's annualization factor; out of
scope for Phase 1 (the same assumption the existing haircut already makes).

**Gate check:** `dsr_evidence` passes iff `dsr_confidence ≥ 1 − DSR_ALPHA`, protected constant
`DSR_ALPHA = 0.05` (≥95% confidence the true Sharpe beats the selection-inflated benchmark). The
returned quantity is named `dsr_confidence` (the probability), explicitly NOT `dsr_pvalue`, to
avoid the `≥`/`≤` inversion trap. Added as a new `GateSpec`-style check.

**Numerical guards inside `dsr_confidence(...)` (all → `None` = fail closed):** `T ≤ 1`; non-finite
or `< 0` `trial_sr_var`; the variance term `1 − γ₃·SR + ((γ₄−1)/4)·SR² ≤ 0` or non-finite
(unstable on short/pathological holdouts); any non-finite intermediate or result.

### DSR inputs (and the Phase-1 approximation, stated honestly)

The "effective number of independent trials" from return-stream correlation is **Phase 3**. Until
it exists, Phase 1 uses:
- `N` = raw funnel breadth. Taken alone this is the conservative direction (overstates independent
  trials → larger `SR*`).
- `trial_sr_var` = the variance of the strategy's **own** sweep Sharpes (per-period after `/ANN`),
  **pooled across all of the strategy's sweeps as the exact pooled SAMPLE variance** (`ddof=1`,
  matching how each row's `var` is computed) — NOT a naive count-weighted mean of per-sweep
  variances (that ignores between-sweep means and understates dispersion), and NOT the
  population-style `E[var]+Var[mean]` (divides by `N`, not `N−1`). From the `(count nᵢ, mean μᵢ,
  var sᵢ²)` triples (see Footprint):

  ```
  M   = Σ(nᵢ·μᵢ) / Σnᵢ
  SSE = Σ( (nᵢ−1)·sᵢ²  +  nᵢ·(μᵢ − M)² )
  pooled_sample_var = SSE / (Σnᵢ − 1)          for Σnᵢ ≥ 2;  0.0 for Σnᵢ == 1
  ```

**This pairing is NOT guaranteed conservative overall — stated plainly.** `N` is conservative, but
own-sweep `trial_sr_var` can be *small* when the grid explores near-duplicate parameters (low
Sharpe dispersion), which *shrinks* `SR*` and makes the DSR layer lenient — in the limit
`trial_sr_var → 0` ⇒ `SR* → 0` ⇒ DSR collapses to plain PSR with no multiplicity penalty. We
accept this for Phase 1 **because the existing haircut remains a binding AND-check and does not
depend on `trial_sr_var`** — so the low-dispersion gap is still covered by the haircut floor; the
DSR simply adds no extra protection there (it never *weakens*, by the tighten-only invariant).
A calibrated dispersion floor (e.g. funnel-wide cross-strategy trial variance) needs data we don't
have until Phase 3 and is **deferred there** rather than guessed now. The `(count, mean, var)`
triple is recorded precisely so Phase 3 can compute that floor without a migration. All of this is
documented at the computation site in `gates.py` and surfaced in the gate payload for audit.

### Footprint (five files; two protected)

1. **`algua/backtest/metrics.py`** *(unprotected)* — add `skewness` and **raw** `kurtosis`
   (`fisher=False`) to `metrics_from_returns`, computed on the **same** return series as `sharpe`.
   This puts the moments (and the consistent sample length) into `holdout_metrics` so the gate reads
   pre-computed values and `gates.py` stays pure-math. **The new keys are coerced to 0.0 when
   non-finite** — extend the degenerate guard beyond `len(r) == 0` to also cover `len(r) ≤ 1` and
   zero-variance series, where `scipy.stats.skew/kurtosis` return **NaN** that would otherwise leak
   into `holdout_metrics` and violate the "never NaN in the payload" discipline. These 0.0
   placeholders are never *consumed* by the PSR formula because `dsr_confidence`'s `T ≤ 1` guard and
   `MIN_HOLDOUT_OBSERVATIONS=63` floor reject such holdouts first. `holdout_metrics` is already
   SENSITIVE (withheld from operators) — the moments inherit that handling. The PSR `T` is the count
   of finite returns in that series (the segment length the moments were computed on), passed
   alongside the moments.

2. **`algua/backtest/sweep.py`** *(unprotected)* — `sweep()` computes, over its per-combo ranking
   Sharpes (annualized, in COMBO order, before ranking), the triple `(count, mean, var)` with
   sample variance `ddof=1` for count ≥ 2 and `var = 0.0` for count = 1, and returns it on the
   sweep result object. The holdout is still never recorded — only the ranking-Sharpe dispersion.

3. **`algua/registry/repository.py`** + **`algua/registry/db.py`** *(unprotected)* — schema bump
   **23 → 24** (`SCHEMA_VERSION` in `db.py`; marker only — migration is idempotent via
   introspection). Add three nullable columns to `search_trials`:
   `trial_sharpe_count INTEGER`, `trial_sharpe_mean REAL`, `trial_sharpe_var_ann REAL` (via
   `_add_missing_columns` in `migrate()`; no `user_version` gate). `record_search_trial` gains
   `trial_sharpe_count`, `trial_sharpe_mean`, `trial_sharpe_var_ann` parameters (update the
   `StrategyRepository` Protocol and all call sites — additive, no optional-default cruft on a
   single internal caller). New accessor pools the strategy's own `(count, mean, var)` triples via
   the exact pooled-sample-variance formula (see "DSR inputs") into one per-period dispersion; it
   returns `None` (→ agent fail-closed) **iff any row contributing to the strategy's own breadth
   lacks finite stats** (precise rule: the query selects the strategy's own measured sweep rows and
   fails if any has NULL/non-finite stats — NULL rows are never silently skipped). **Non-finite
   detection at the SQLite boundary:** treat `NULL`, `NaN`, and `±inf` uniformly as missing — after
   reading each value, `None`-or-`not math.isfinite(...)` → return `None`. **Known Phase-1 limitation
   (documented at the computation site):** overlapping combos across re-sweeps are double-counted in
   the pooled `count`/variance; acceptable because the haircut stays binding and Phase 3's
   cross-strategy dispersion floor supersedes it. (A sweep always yields ≥1 combo — grid validation
   enforces it — so `count = 0` is unreachable.)

4. **`algua/research/gates.py`** *(PROTECTED)* — pure `dsr_confidence(sr_obs_per_period, t, skew,
   raw_kurtosis, n_trials, trial_sr_var_perperiod) -> float | None` (with the `N≤1→SR*=0`,
   denominator, finiteness, and negative-variance guards above; `n_trials` is an integer count —
   defensive `int(...)` cast at the top). Protected constants `DSR_ALPHA = 0.05` and
   `EULER_MASCHERONI`. The `dsr_evidence` check is added to the binding check-list **only when
   binding** (see rules). New `GateDecision` fields recorded either way (nulled when not finite, like
   the existing payload): `dsr_confidence: float | None`, `dsr_binding: bool`,
   `dsr_skip_reason: str | None`, `dsr_sr_star: float | None`, `dsr_n_trials: int | None`,
   `dsr_trial_sr_var: float | None`, `dsr_t: int | None`, `dsr_skew: float | None`,
   `dsr_raw_kurtosis: float | None`.

5. **`algua/registry/promotion.py`** *(PROTECTED)* — `run_gate` reads `skew`/`raw_kurtosis`/`T`
   from `wf.holdout_metrics`, `N` from `effective_funnel_breadth`, and the pooled per-period
   dispersion from the new accessor; passes them into `evaluate_gate`; persists them (incl.
   `dsr_binding`, `breadth_source`, `dsr_skip_reason`) in the `gate_evaluations` row.

### Binding / fallback rules (advisory = OMITTED from the AND-set, not appended-as-False)

The binding rule is **actor-INDEPENDENT and keyed only on whether measured dispersion exists** — so
declared breadth can never be used to *dodge* a computable DSR check (the round-1 actor-based rule
had that bypass: a human with a real sweep could `--n-combos` past DSR).

- **DSR binds whenever measured dispersion is available.** Concretely: the strategy has ≥1 measured
  sweep row and the pooled-variance accessor returns a finite value → `dsr_evidence` is **added to
  the binding check-list** and contributes to `passed = all(checks)`, for agent AND human alike.
  The agent path requires measured breadth, so DSR always binds there. A human using `--n-combos`
  while a measured sweep also exists is **still bound** by DSR.
- **DSR is omitted ONLY when no measured dispersion exists** (a strategy with no sweep at all,
  reachable only on the human declared-breadth path). Then the `dsr_evidence` check is **omitted
  from the binding check-list** (NOT appended `passed=False`, which would wrongly block the human
  escape hatch). Payload records `dsr_binding=false`, `dsr_confidence=null`,
  `dsr_skip_reason="no_measured_dispersion"`. Mirrors how `pit_required` separates a relaxable
  concern from the hard AND-set.
- **Old pre-migration rows fail closed (not omitted).** If the strategy has measured sweep rows but
  any contributing row lacks finite `(count, mean, var)`, the accessor returns `None` →
  `dsr_confidence None` → the binding check **fails closed** with a re-sweep message. (Distinct from
  the no-sweep case above: a sweep happened but its stats predate the schema bump.)
- **No implicit DSR override.** Phase 1 has NO flag to relax a *failing* DSR check. Consistent with
  the gate philosophy, any future escape must be an explicit, audited human flag — deferred; not a
  side effect of declared breadth. (`candidate` is not capital and DSR is tighten-only, so a missing
  override is safe.)
- **Tighten-only invariant (precise).** For any input, the new overall verdict equals
  `old_pass AND (NOT dsr_binding OR dsr_pass)`. Existing checks and their thresholds are byte-for-byte
  unchanged; DSR can only ever subtract a pass when bound, and is absent otherwise.

### Edge cases (all fail-closed, mirroring the existing haircut)

- `T ≤ 1` (degenerate holdout): haircut already drives the Sharpe bar out of reach when `T ≤ 0`;
  `dsr_confidence` returns `None` for `T ≤ 1` → `dsr_evidence` fails closed (nulled in payload, never NaN).
- `N ≤ 1`: `SR* = 0` via explicit guard → DSR collapses to plain PSR against 0. Correct, not degenerate.
- `trial_sr_var = 0` (single-combo sweep, low-dispersion grid): `SR* = 0` → DSR = PSR. Accepted —
  the haircut remains the binding floor (see "DSR inputs"); DSR adds no extra protection here but
  never weakens.
- **dispersion missing (old `search_trials` rows) on the agent path:** **fail closed** with a
  re-sweep message (re-sweeping is cheap and `sweep()` drops the holdout, so it never burns it). **No
  grace fallback** — a NULL-tolerant advisory path would be dual-path cruft on a protected wall.
- Non-finite confidence / denominator ≤ 0 / negative variance: `None` → fail closed, nulled in payload.

### Testing

- Pure-function unit tests for `dsr_confidence`: pinned reference values using `γ_E` =
  Euler–Mascheroni and **raw** kurtosis; a normal-like series yields `γ₃≈0, γ₄≈3` and the variance
  term `1+SR²/2`; monotonic in N, T, SR_obs; `N≤1`→PSR-against-0 collapse (no `−∞`); `trial_sr_var=0`;
  `T≤1`→`None`; denominator≤0→`None`; negative/NaN variance→`None`.
- **Tighten-only property test (strong form):** over a generated grid of gate decisions, assert
  `new_pass == old_pass AND (not dsr_binding or dsr_pass)` — not merely "never flips FAIL→PASS".
- Pooling test: the exact pooled-sample-variance accessor across multiple sweeps with differing means
  exceeds the naive count-weighted within-sweep mean; **equal-sweep-means → pooled equals the naive
  within-sweep average** (between-sweep term zero); single-sweep matches that sweep's variance; any
  NULL/NaN/inf contributing value → `None`.
- Promotion integration: agent measured path binds; human declared path omits DSR (still promotable);
  missing-variance old row fails closed for an agent with the re-sweep message.
- `metrics.py`: skew/kurtosis present, raw-kurtosis convention pinned, empty segment → 0.0.
- `sweep()` triple-recording test (computed in combo order; single-combo → count 1, var 0.0).
- Schema-migration test (24, idempotent, NULL on pre-existing rows, no double-add).

### Dependencies

`gates.py` needs `Φ`/`Z⁻¹` (`scipy.stats.norm.cdf`/`ppf`). scipy is already importable in the env
(1.17.x, transitive); make it an **explicit** dependency in `pyproject.toml` since a protected module
now imports it directly. `lint-imports` boundaries are unaffected (`gates.py` is in `algua/research`,
which may import third-party libs; `contracts`/`features` purity is untouched).

### Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.

## Deferred phases (designed; file as follow-up issues)

**Phase 2 — online hierarchical FDR accounting.** **IMPLEMENTED** (#220, schema v26, 2026-06-16).
LORD++ alpha-wealth ledger merged into the `backtested → candidate` gate: `FDR_ALPHA=0.05`,
`W0=0.025`, γ-discount sequence normalized over 10 000 terms. The per-strategy DSR p-value
(`p = 1 − dsr_confidence`) is the LORD++ input; every measured-breadth gate evaluation is one
stream position; discoveries replenish budget, dry spells tighten it. FDR is a tighten-only
AND-check (declared/human breadth skips FDR entirely). The persistent stream lives in
`gate_evaluations` (5 new nullable FDR columns + partial unique index on `fdr_test_index`);
atomic write uses `BEGIN IMMEDIATE` (mirrors `reserve_holdout`). See plan doc
`2026-06-16-lord-plus-plus-fdr-ledger-211.md`. SAFFRON deferred to Phase 4 (#222).

**Phase 3 — dependence-aware calibration (load-bearing).** Estimate **effective independent
trials** from strategy return-stream correlation (replaces the raw-count `N` in the DSR benchmark);
block / stationary bootstrap to calibrate nulls under autocorrelation + shared regimes; require
**multi-regime robustness**, not a single aggregate holdout p-value. Also adds the **dispersion
floor** the Phase-1 own-sweep `trial_sr_var` lacks: a funnel-wide cross-strategy trial-Sharpe
variance, computable from the `(count, mean, var)` triples Phase 1 already records (no migration),
to remove the low-dispersion leniency noted in "DSR inputs".

**Phase 4 — hierarchical family budgets + anti-gaming (#222).** A GLOBAL alpha budget above
per-thesis-**family** budgets; family creation governed (not automatic); the global cap means
spawning families can't mint free alpha; empirical clustering by return-correlation / holdings /
**factor lineage (#140)** / **code ancestry**, with parentage tracking, so a "new" family that
behaves like an old one inherits its budget. Builds on #137 (bind funnel breadth to family-id),
#122 (family metadata), #161/#192/#193/#205 (holdout single-use + identity), #140 (factor lineage).
**Detailed spec:**
`docs/superpowers/specs/2026-06-16-hierarchical-family-budgets-anti-gaming-issue-222-design.md`.
TDD plan: `docs/superpowers/plans/2026-06-16-hierarchical-family-budgets-222.md`
**Status:** Stratum A (anti-gaming core) BUILT — canonical family registry + parentage DAG
(schema 25→26), pure clustering module (code-ancestry + factor-lineage + return-correlation axes),
governed family creation (agent NOVEL → fail-closed), family-scoped breadth (3-way
`effective_funnel_breadth`, tighten-only), anti-reset lifetime inheritance, CAS verification.
Stratum B (`FamilyBudgetLedger` Protocol + in-memory contract) shipped as interface-only — real
LORD++ binding deferred to Phase 2.

**End-state (retire the haircut).** The haircut and the DSR both correct for the **same** best-of-N
selection inflation — the haircut is the crude unit-normal/asymptotic version; the DSR is the better
version using measured trial dispersion + non-normality. Keeping both as an AND is **deliberately
conservative** and is the explicit Phase-1 transition state ("keep the haircut as a fallback until
calibration matures"). Once the DSR's calibration is audited under Phases 2–3, a named follow-up
**retires the haircut** so the system does not carry two redundant multiplicity penalties forever.

## Caveat (carried from the issue)

FDR here governs **research-discovery quality** and, given shared-holdout dependence, is an
**operating target, not a proof**. `candidate` is not capital; the live wall and forward-test
certificate remain the hard guards.
