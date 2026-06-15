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

**PSR (Probabilistic Sharpe Ratio)** — confidence that the true Sharpe exceeds a benchmark,
adjusting for sample length and non-normality of the return stream:

```
PSR(SR*) = Φ( (SR_obs − SR*) · √(T − 1) / √(1 − γ₃·SR_obs + ((γ₄ − 1)/4)·SR_obs²) )
```

- `SR_obs` — observed holdout Sharpe (**per-period**, i.e. the annualized holdout Sharpe ÷ √ANN;
  the PSR formula is in per-period units — unit discipline mirrors the existing haircut's
  `√ANN` handling).
- `T` — holdout observation count (`wf.holdout_metrics["n_bars"]`).
- `γ₃`, `γ₄` — skewness and kurtosis of the **holdout** return stream (the non-normality
  adjustment). Computed from the holdout returns the walk-forward already produced.

**DSR (Deflated Sharpe Ratio)** — PSR where the benchmark `SR*` is the **expected maximum Sharpe
under N trials** rather than 0:

```
SR* = √(trial_sr_var) · [ (1 − e⁻¹)·Z⁻¹(1 − 1/N) + e⁻¹·Z⁻¹(1 − 1/(N·e)) ]
```

- `N` — trial count = `effective_funnel_breadth(own_lifetime, windowed_total)` (the **same** `N`
  the haircut uses; a raw count, hence a conservative upper bound on independent trials).
- `trial_sr_var` — **variance of the per-combo trial Sharpe ratios**, measured empirically from
  the strategy's own `backtest sweep`. This is exactly the cross-trial dispersion Bailey–LdP use.
- `Z⁻¹` — inverse standard-normal CDF.

**Unit discipline (critical, mirrors the haircut's `√ANN` handling).** The Bailey–LdP formulae are
in **per-period** Sharpe units, but the system's Sharpes (both `SR_obs` and the sweep's per-combo
Sharpes) are **annualized** (`SR_ann = SR_per_period · √ANN`). All three SR-bearing inputs must be
converted to per-period at the point of use, inside `gates.py`, next to the formula: `SR_obs →
SR_obs/√ANN`, and `trial_sr_var → trial_sr_var/ANN` (variance scales by the square). The sweep
**records the variance of its native annualized Sharpes**; the single `/ANN` conversion lives in the
protected gate so all unit handling is co-located with the math.

**Gate check:** `dsr_pvalue ≥ 1 − DSR_ALPHA` with a protected constant `DSR_ALPHA = 0.05`
(≥95% confidence the true Sharpe beats the selection-inflated benchmark). Added as a new
`GateSpec`-driven boolean check `dsr_evidence`.

### Why these inputs (and the Phase-1 approximation, stated honestly)

The "effective number of independent trials" from return-stream correlation is **Phase 3**. Until
it exists, Phase 1 uses:
- `N` = raw funnel breadth — overstates independent trials → **stricter** `SR*` → conservative.
- `trial_sr_var` from the strategy's **own** sweep, paired with the **funnel-wide** `N`. The
  dispersion is estimated from own-combo Sharpes while `N` counts funnel-wide trials — a known
  approximation. The own-sweep dispersion is a reasonable proxy for trial-Sharpe spread, and the
  funnel-wide `N` is the conservative count. Documented in `gates.py`, surfaced for review.

### Footprint (four files; two protected)

1. **`algua/backtest/sweep.py`** *(unprotected)* — `sweep()` computes the **variance of the
   per-combo ranking Sharpes** (the window/stability Sharpe it already computes per combo, in
   COMBO order, before ranking) and returns it on the sweep result object. The holdout is still
   never recorded — only the ranking-Sharpe dispersion. A single-combo sweep yields variance 0.

2. **`algua/registry/repository.py`** *(unprotected)* — schema bump 23 → 24. Add nullable column
   `trial_sharpe_var REAL` to `search_trials` (via the existing introspection +
   `_add_missing_columns` ALTER-TABLE migration; idempotent, no user_version gate). `record_search_trial`
   persists it. New accessor pools the strategy's own recorded `trial_sharpe_var` across its sweep
   rows (count-weighted pooled variance) for the DSR dispersion input.

3. **`algua/research/gates.py`** *(PROTECTED)* — pure `dsr_pvalue(sr_obs_per_period, t, skew,
   kurtosis, n_trials, trial_sr_var) -> float | None`; protected `DSR_ALPHA = 0.05`; new
   `dsr_evidence` check wired through the existing `GateDecision`/check-list machinery. DSR inputs
   and the resulting p-value recorded into the decision payload.

4. **`algua/registry/promotion.py`** *(PROTECTED)* — `run_gate` assembles DSR inputs (skew/kurtosis/T
   from the holdout return stream; `N` from `effective_funnel_breadth`; `trial_sr_var` from the
   pooled own-strategy accessor) and threads them into `evaluate_gate`; persists them in the
   `gate_evaluations` row alongside the existing breadth provenance.

### Binding / fallback rules

- **DSR binds only when its inputs are real.** `trial_sr_var` exists only from a *measured* sweep.
  The agent path **requires** measured breadth → DSR is always computable → **binding** for agents.
- **Human `--n-combos` declared-breadth path:** no measured variance exists. DSR is recorded
  **advisory** (computed where possible, not binding) — consistent with declared breadth already
  being a human-accepted relaxation. We never block the human escape hatch; we never let an agent
  skip DSR.
- **Tighten-only invariant.** DSR is an additional AND; it can only flip PASS→FAIL. The haircut
  check is unchanged.

### Edge cases (all fail-closed, mirroring the existing haircut)

- `T ≤ 0` (degenerate holdout): haircut already returns `inf` → gate fails closed; `dsr_pvalue`
  returns `None` → `dsr_evidence` fails closed (never recorded as NaN; nulled in payload like the
  other checks).
- `N = 1` (single pre-registered trial / single-combo sweep): no selection inflation → `SR* = 0`
  → DSR collapses to plain PSR against 0. Correct, not degenerate.
- `trial_sr_var = 0` (one-combo sweep): `SR* = 0` → DSR = PSR. Correct.
- **`trial_sr_var` missing (old `search_trials` rows from before the bump) on the agent path:**
  **fail closed** — `dsr_evidence` fails and the operator is told to re-run the sweep (cheap;
  `sweep()` drops the holdout so re-sweeping never burns it). **No grace fallback** — a NULL-tolerant
  advisory path would be dual-path cruft on a protected wall and buys nothing.
- Non-finite p-value (pathological return stream → NaN): check fails closed, nulled in payload.

### Testing

- Pure-function unit tests for `dsr_pvalue`: known Bailey–LdP reference values; monotonic in N, T,
  and SR_obs; correct skew/kurtosis direction; `N=1`→PSR-against-0 collapse; `trial_sr_var=0`;
  `T≤0`→`None`; NaN→`None`.
- **Tighten-only property test:** across a grid of inputs, `dsr_evidence` never flips a
  haircut-FAIL into a gate-PASS (DSR can only subtract passes).
- Promotion integration: agent measured path binds; human declared path advisory; missing-variance
  old row fails closed for an agent with the re-sweep message.
- `sweep()` variance-recording test (variance computed in combo order, single-combo → 0).
- Schema-migration test (24, idempotent, NULL on pre-existing rows).

### Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.

## Deferred phases (designed; file as follow-up issues)

**Phase 2 — online hierarchical FDR accounting.** LORD++ first (conservative, valid for
never-fixed-N streaming; plain BH needs a fixed N), SAFFRON later once p-value calibration is
auditable. Introduces a **persistent alpha-wealth ledger** (new state): discoveries replenish
budget, dry spells tighten it. **Builds on Phase 1's calibrated p-value** as the LORD++ input.

**Phase 3 — dependence-aware calibration (load-bearing).** Estimate **effective independent
trials** from strategy return-stream correlation (replaces the raw-count `N` in the DSR benchmark);
block / stationary bootstrap to calibrate nulls under autocorrelation + shared regimes; require
**multi-regime robustness**, not a single aggregate holdout p-value.

**Phase 4 — hierarchical family budgets + anti-gaming.** A GLOBAL alpha budget above per-thesis-
**family** budgets; family creation governed (not automatic); the global cap means spawning families
can't mint free alpha; empirical clustering by return-correlation / holdings / **factor lineage
(#140)** / **code ancestry**, with parentage tracking, so a "new" family that behaves like an old
one inherits its budget. Builds on #137 (bind funnel breadth to family-id), #122 (family metadata),
#161/#192/#193/#205 (holdout single-use + identity), #140 (factor lineage).

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
