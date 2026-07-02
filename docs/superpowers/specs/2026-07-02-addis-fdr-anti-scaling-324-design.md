# Design v2 — Issue #324: LORD++ alpha-wealth ledger is anti-scaling

## Problem (verified)
The `backtested -> candidate` gate runs an online-FDR alpha-wealth ledger (LORD++, #211/#220) over
ONE lifetime-global stream. Every MEASURED attempt (breadth measured + finite dsr_confidence) is a
test at position t. p = 1 - dsr_confidence. Reject iff p_t <= alpha_t. In a dry spell alpha_t =
gamma_t*W0 -> 0 as t grows, so every additional FAILED exploration monotonically lowers everyone's
future bar toward 0. Testing MORE makes the gate strictly LESS passable -- anti-scaling.

## Rejected: issue "recommendation #1" (bind only PASSING rows) -- FDR-INVALID
Online-FDR must charge the scheduled alpha to every test that COULD have rejected. Excluding failed
tests is selection-on-outcome; the ledger never pays for the multiplicity it incurred, so true FDR
is uncontrolled (inflated). Loosening disguised as accounting. REJECTED. (Codex GATE-1 confirmed.)

## Rejected: SAFFRON -- does NOT actually fix conservative-null alpha-death
SAFFRON's constant-lambda gamma index is `t - tau_j - C_{j+}` where C counts CANDIDATES (p<=lambda),
so the index == 1 + (# NON-candidates since the event). A run of conservative nulls (p>lambda) STILL
advances the clock and STILL decays the next candidate's level. Verified against arXiv:1802.09098
Sec 2.3 with a concrete stream. SAFFRON avoids charging small-p CANDIDATES, not large-p nulls.
So SAFFRON does not fix #324. REJECTED.

## Chosen fix: ADDIS* (Tian & Ramdas, NeurIPS 2019, arXiv:1905.11465) -- adaptive DISCARDING
ADDIS adds a DISCARD threshold tau: a p-value with p > tau (a VERY conservative null -- exactly our
p ~= 1 garbage strategies) is DISCARDED. Discarded tests do NOT advance the gamma clock and do NOT
spend wealth. This is the published, FDR-valid fix for "conservative-null alpha-death". It reduces
to SAFFRON at tau=1 and controls FDR under the SAME assumption class #220 already used as its
operating target (independent nulls + conditional conservativeness + monotone predictable
alpha/lambda/(1-tau)). It DOMINATES SAFFRON in power in the conservative-null regime (our regime).

### EXACT constant-parameter recursion (verified vs arXiv:1905.11465 Algorithm 1 / Appendix J)
Symbols: alpha=target FDR; W0<=alpha initial wealth; gamma[0],gamma[1],... nonneg, nonincreasing,
sum 1; lambda=candidate threshold; tau=discard threshold, with lambda < tau; p_i=p-value.
Per test i: candidate C_i = 1{p_i <= lambda}; selected/not-discarded S-contrib = 1{p_i <= tau};
rejection R_i = 1{p_i <= alpha_i}. kappa_j = time of j-th rejection; kappa_j* = # selected up to
kappa_j; C_{j+}(t) = # candidates in (kappa_j, t) i.e. sum_{i=kappa_j+1}^{t-1} C_i; C_{0+} =
sum_{i=1}^{t-1} C_i; S_t = sum_{i<t} 1{p_i <= tau}.

  alpha_t = min( lambda, (tau - lambda) * (
       W0        * gamma[ S_t - C_{0+} ]
     + (alpha-W0)* gamma[ S_t - kappa_1* - C_{1+} ]
     + alpha * sum_{j>=2} gamma[ S_t - kappa_j* - C_{j+} ]
  ) )
(0-indexed gamma. If kappa_j doesn't exist its term is absent. Note the prefactor is (tau-lambda),
NOT (1-lambda).)

PROOF discards don't advance the clock (lambda=0.25, tau=0.5, no rejection):
 t=1 p=0.1 (candidate,selected); t=2..6 p=0.9 (DISCARDED); t=7 p=0.1 (candidate).
 At t=7: S_7 = # selected before 7 = 1 (only t=1; the five p=0.9 are discarded). C_{0+}=1.
 w0 gamma index = S_7 - C_{0+} = 1 - 1 = 0 -> gamma[0].
 Same index as if the 5 discards were absent. QED: adding conservative-null spam does NOT decay
 the next candidate's level. This is precisely the anti-scaling fix.

### Boundary conventions (paper-exact)
- candidate: p <= lambda ; selected/non-discarded: p <= tau ; discarded: p > tau.
- rejection (discovery): p <= alpha_t (keep <=, consistent with the ledger's existing p<=alpha_t).

### Constants (chosen)
- ADDIS_LAMBDA = 0.25, ADDIS_TAU = 0.5 (paper defaults, theta=lambda/tau=0.5).
- ALPHA=0.05, W0=0.025 unchanged. Constraint W0<=alpha holds; lambda(0.25) < tau(0.5) holds;
  and lambda(0.25) >= alpha(0.05) holds (Appendix J requires lambda in [alpha, tau)).
- gamma: KEEP the existing normalized LORD gamma sequence (nonneg, decreasing, sum~1) -- it
  satisfies ADDIS's gamma requirements. No need to switch to (j+1)^-1.6; reuse _LORD_GAMMA. This
  keeps the diff minimal AND is valid (ADDIS only requires nonneg/nonincreasing/sum<=1).

### Discovery / tighten-only preserved
final_passed = provisional_passed AND fdr_rejected. fdr_rejected can only REMOVE a pass. A
discarded row (p>tau) is a non-candidate: alpha_t <= lambda < tau < p, so p>alpha_t -> fdr_rejected
False -> auto-fails FDR, which is correct (dsr_confidence < 1-tau = 0.5 is not a discovery anyway).

## Implementation (contained)
1. gates.py (PROTECTED): NEW pure `addis_level(t_selected_state) -> float` replacing the
   lord_plus_plus_level call path. Because ADDIS needs the FULL selected/candidate/rejection
   history (not just rejection indices), the pure function takes the ordered list of (p_value)
   for all binding rows + the constants, and computes alpha_t for the NEXT test. Keep
   lord_plus_plus_level in place ONLY if referenced elsewhere; otherwise replace. Add ADDIS
   constants. Pure, unit-tested against hand-worked streams incl. the discard-proof above.
2. repository.py/store.py `fdr_stream_state`: the LORD stream state (t, discovery_indices) is
   INSUFFICIENT for ADDIS -- ADDIS needs each binding row's p_value (to recompute S/C/discard).
   Change the SELECT to also return fdr_p_value (already a stored column -- NO schema change) and
   have FdrStreamState carry the ordered p-values. store.py computes alpha_t via the injected
   level_fn given the full p-value history. Minimal store touch: SELECT column + NamedTuple field
   + pass p-history to level_fn. NO migration.
3. promotion.py: swap the injected partial from lord_plus_plus_level to addis_level with
   alpha/w0/lambda/tau. ~1 line.

### The level_fn signature change
Old: level_fn(t_next, discovery_indices) -> float.
New: level_fn(prior_p_values: Sequence[float]) -> float, where prior_p_values is the ordered list
of every prior binding row's p-value (chronological). addis_level derives S_t, C_{j+}, kappa*, and
computes alpha for the NEXT (t = len+1) test. The store still computes fdr_rejected = p_next <=
alpha_t and records t_next = len+1, fdr_rejected, alpha_t. discovery_indices are no longer needed as
input (derived internally), but fdr_test_index / fdr_rejected columns stay for audit.

### EPOCH / legacy-LORD (Codex GATE-1 BLOCKER -- resolved with a real epoch boundary)
Codex is RIGHT: ADDIS's past rejections kappa_j must be rejections ADDIS ITSELF produced
(recursively p_i <= alpha_i under ADDIS levels), NOT LORD++'s stored fdr_rejected flags. Mixing
corrupts the wealth recursion. So we introduce a hard ADDIS EPOCH.

Implementation (schema 31 -> 32, additive nullable column -- the established _add_missing_columns
pattern):
- NEW nullable column gate_evaluations.fdr_algo TEXT. Legacy LORD++ rows have fdr_algo NULL.
  ADDIS rows are stamped fdr_algo='addis_v1'.
- fdr_stream_state reads the ADDIS EPOCH ONLY: WHERE fdr_binding=1 AND fdr_algo='addis_v1'
  ORDER BY id, returning the ordered fdr_p_value list. Legacy LORD rows are EXCLUDED entirely.
- addis_level recomputes S/C/kappa AND rejections RECURSIVELY from that epoch's p-value history
  (a rejection at epoch-position i is defined as p_i <= alpha_i where alpha_i is itself the ADDIS
  level computed from p_1..p_{i-1}). It NEVER reads stored fdr_rejected. This is the only valid
  ADDIS state and is fully deterministic from the epoch p-value list.
- fdr_test_index stays the GLOBAL binding-row counter (t = total binding rows across BOTH epochs
  + 1) purely as an audit position, so the existing partial UNIQUE index on fdr_test_index never
  collides. ADDIS's math uses positions WITHIN the epoch list, independent of fdr_test_index.
- Because the epoch starts empty at deployment, the first ADDIS test sees an empty history ->
  alpha_1 = min(lambda, (tau-lambda)*W0*gamma[0]) -- a clean fresh ADDIS stream. No LORD leakage.
Reproducibility: lambda/tau/algo are protected constants; changing them changes recomputation --
same property LORD++ had. fdr_stream_state fail-closed validation extended: any epoch row with
NULL/non-finite fdr_p_value -> None (fail closed), contiguity check on the epoch's own ordering.

### Concurrency
Unchanged: fdr_stream_state runs inside the same BEGIN IMMEDIATE critical section; the p-value
history is read under the write lock, identical atomicity to today. Ordering: keep ORDER BY id AND
validate fdr_test_index contiguity (already done) -- add nothing.

## Scope / conflict-avoidance
- gates.py (protected -> PR OPEN for human merge): ADDIS pure math + constants.
- db.py: schema 31->32, one additive nullable column fdr_algo (established _add_missing_columns
  pattern, additive, does NOT touch #330/#334's promotion CAS or store schema logic).
- store.py: fdr_stream_state epoch filter + p-history; record_gate stamps fdr_algo='addis_v1'.
- repository.py: FdrStreamState carries ordered p-values; Protocol sig for level_fn.
- promotion.py: 1-line partial swap lord->addis.
- tests.
- Avoids engine.py (#325). The schema touch is additive-column only (noted in PR); does not
  overlap #330/#334's CAS/promotion-funnel logic.

## FDR validity summary (the honest answer)
- Still controls FDR at 0.05 under the same assumptions #220 accepted (ADDIS Theorem 1).
- NOT anti-scaling: conservative-null (p>tau) garbage is DISCARDED -> does not advance the gamma
  clock -> does not decay the bar for legitimate future candidates. Proven above.
- Not a loosening: the multiplicity of REAL candidates (p<=tau, near-significant) is STILL fully
  charged. We only stop charging alpha for hypotheses so far from significance they carry no
  evidence -- which is exactly what ADDIS proves is FDR-safe to discard.
