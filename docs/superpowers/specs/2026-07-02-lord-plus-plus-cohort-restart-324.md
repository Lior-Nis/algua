# LORD++ FDR Ledger — Count-Triggered Cohort Restarts (Issue #324)

**Date:** 2026-07-02
**Issue:** #324 (refines #211/#220 Phase 2 LORD++ ledger)
**Schema:** 31 → 32
**Status:** design APPROVED (Codex GATE-1, 3 iterations)

## Problem
The LORD++ alpha-wealth ledger (#211/#220) is ONE lifetime-global stream: every MEASURED
`research promote` binding test (breadth measured + finite `dsr_confidence`) advances stream
position `t` — regardless of pass/fail. In a dry spell (no rejections) `alpha_t = gamma_t * W0`
and `gamma_t` decays, so `alpha_t -> 0` as `t` grows. At hundreds of failed attempts/day with a
realistic <1% pass rate the stream sits in a permanent dry spell and **every additional failed
exploration monotonically lowers everyone's future bar** — testing MORE makes the gate strictly
LESS passable. Anti-scaling by construction; the #1 binding constraint on throughput.

## Statistical analysis (GATE-1)
- **`alpha`-death is CORRECT** LORD++ behavior for a genuinely-null *lifetime* stream: any valid
  *lifetime* online-FDR procedure MUST drive the per-test level to 0 over an unbounded null
  stream, else it makes infinitely many false discoveries. So the anti-scaling is intrinsic to a
  *lifetime* target on a garbage-dominated funnel — not a tuning bug.
- **"Only bind passing rows" is FDR-INVALID** (issue rec #1): it hides non-rejections from the
  multiplicity process, pinning `alpha_t` near `alpha` forever. Covert loosening. REJECTED.
- **Fixed calendar epochs are insufficient**: at hundreds/day, `t` reaches 10k–45k *within* one
  90-day epoch and `alpha`-deaths again — merely reschedules death quarterly. REJECTED.
- **SAFFRON is insufficient** for THIS failure mode: SAFFRON indexes `gamma` by *non-candidate*
  count, so clear-null garbage (`p ≈ 1`) still advances the index and still `alpha`-deaths;
  SAFFRON only conserves wealth on *candidate* near-misses. REJECTED.
- **Theorem**: any valid procedure indexing decay by TEST COUNT must `alpha`-death over an
  unbounded mostly-null stream ⇒ the honest fix is to BOUND the count via a re-scoped target.

## Solution — count-triggered LORD-with-restarts (fixed-size cohorts)
Partition the binding-test stream into consecutive, non-overlapping **cohorts of exactly
`FDR_COHORT_SIZE = 64` binding tests**, assigned by **arrival order** (global binding ordinal
`k`, 1-based):

    cohort_index = (k - 1) // 64
    t            = (k - 1) %  64 + 1      # within-cohort position, 1..64

Each cohort runs an **independent LORD++ stream**: fresh `W0`, `t ∈ [1..64]`, `tau_j` = in-cohort
rejection positions. `lord_plus_plus_level` (pure math) is UNCHANGED. Reject iff `p_t <= alpha_t`.

**FDR is controlled PER COHORT of 64 binding tests at `ALPHA = 0.05`** — an explicit re-scoped
target (NOT lifetime FDR), documented in code + CLAUDE.md. Because `N` is bounded, the worst-case
dry-spell level is floored at `gamma_64 * W0 ≈ 4.6e-5`, **independent of throughput**: 1000
tests/day or 1/day yield identical within-cohort statistics. That is the fix — bounded `N` by
construction, decoupled from throughput (unlike calendar epochs).

### Why `N = 64` (power calibration, not aesthetics)
Real normalized-`gamma` floors (verified against `_LORD_GAMMA`): `alpha_1 = 0.00165`
(dsr ≥ 0.99835) — the FIRST test is already strict, inherent to `W0=0.025` + `gamma`
normalization (pre-existing, not a regression). `alpha_64 = 4.6e-5` (dsr ≥ 0.99995). `N=64` keeps
`alpha_N` within ~35× of `alpha_1` (same order as the pre-existing `alpha_1` strictness) and caps
decay; larger `N` (256 → 1.1e-5) approaches lifetime-like decay, smaller `N` → more independent
5% cohorts (weaker multiplicity control). 64 is the Codex-recommended balance. Protected constant.

### Cumulative exposure accounting (honesty — not a covert loosen)
Per-cohort control is strictly weaker than lifetime FDR: with `K` completed independent cohorts
each at `FDR ≤ ALPHA`, the honest cumulative expected-false-discovery upper bound is
`ALPHA * K` (NOT conditioned on cohorts-with-discoveries — that is post-selection and understates
exposure). The gate surfaces an audit-only `fdr_exposure` block (does NOT change pass/fail):
`fdr_cohort`, `fdr_test_index`, `fdr_cohorts_completed`, `fdr_binding_tests`, `fdr_discoveries`,
`fdr_expected_false_discoveries = ALPHA * fdr_cohorts_completed`. Documented: "FDR controlled per
cohort of 64 binding tests, NOT per lifetime; cumulative exposure grows ~ALPHA per completed cohort."

## Legacy migration (schema 31 → 32)
Add column `fdr_cohort INTEGER` (NULL on legacy). Replace the global unique index
`ix_gate_evaluations_fdr_index(fdr_test_index) WHERE fdr_binding=1` with
`ix_gate_evaluations_fdr_cohort_index(fdr_cohort, fdr_test_index) WHERE fdr_binding=1` (per-cohort
positions restart at 1, so the old global-unique index would false-conflict). One-time migration,
ordered by `id`: for each legacy binding row with global index `g`, set
`fdr_cohort=(g-1)//64` and REWRITE `fdr_test_index=(g-1)%64+1`.

- Stored `fdr_alpha_level` of legacy rows stays **frozen** — a historical record of the decision
  made at the time (computed under the old lifetime formula). The stream reader validates only
  index contiguity + `p`/`alpha` finiteness + `rejected ∈ {0,1}`; it NEVER recomputes alpha for
  past rows, so freezing is correct (recomputing would falsify the audit record). Fail-closed on
  invalid/missing/non-contiguous binding indices, non-finite `p`/`alpha`, or invalid `rejected`.
- **Seal the inherited partial cohort**: the FIRST new binding test after migration starts a
  FRESH cohort boundary (`t=1`) rather than continuing a legacy-populated partial cohort — legacy
  rows were decided under the old formula, so a mixed legacy+fresh cohort can't claim a clean
  cohort-LORD++ guarantee. Legacy cohorts are audit-only.

## Guarantee preservation
- Tighten-only AND-check preserved: `final_passed = provisional_passed AND (p <= alpha_t)`.
- Degenerate inputs / stream-integrity failures still fail closed (`alpha_t = 0.0` unreachable, or
  stream `None` → raise).
- Honestly controlled per-cohort (explicit documented target) — standard count-restart LORD++.

## Scope / conflict-avoidance
Touches `algua/research/gates.py` (CODEOWNERS-protected → PR OPEN for human merge),
`algua/registry/store.py`, `algua/registry/repository.py` (Protocol), `algua/registry/db.py`
(schema), `CLAUDE.md`. No overlap with #325 (engine.py). Merge `origin/main` before GATE-2.
