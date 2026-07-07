# Forward-gate multiple-testing / optional-stopping correction (#431)

Date: 2026-07-07
Branch: `fix/forward-gate-mt-431`
Issue: #431 (lane:ds, severity:medium, north-star: safe_scale)

> This document describes what is ACTUALLY implemented on this branch: a simple, tighten-only
> multiple-testing Sharpe tax on the re-runnable forward gate. It deliberately makes only the
> claims the code backs up. An earlier revision of this doc described a much larger design (a PSR/DSR
> confidence floor, a `#222` family-lineage recursive CTE, an in-lock `BEGIN IMMEDIATE`
> recompute-and-insert store method, and several "Codex GATE-1 rounds"). None of that was built, so
> those claims have been removed. What is deferred is listed honestly in "Deferred / not in v1".

## Problem (grounded in the code)

`evaluate_forward_gate` (`algua/research/forward_gates.py`) is a **single-shot** performance +
integrity gate. Its performance check is the whole statistical story:

```
bar     = max(degradation_factor * holdout_sharpe, sharpe_floor)
passes <=> realized_sharpe >= bar          # a point-estimate comparison
```

There is **no** multiple-testing / optional-stopping term, even though the gate is re-runnable along
two independent axes:

1. **Sequential re-looks (optional stopping).** At `PAPER` an agent may call `paper promote` after
   every new tick session, each a fresh test of `realized_sharpe >= bar` on a **growing** window,
   stopping at the first pass. At `FORWARD_TESTED` a passing run "refreshes the certificate" with no
   stage change — and the live wall trusts the newest certificate for
   `CERTIFICATE_FRESH_SESSIONS = 10` sessions. Peeking-until-pass on one fixed hypothesis inflates
   the false-pass probability of the `paper -> forward_tested` edge — the last statistical gate
   before the (human-signed) live wall.

2. **Concurrent breadth (family-wise).** `n_concurrent_forward` — distinct strategies with paper
   ticks overlapping the window — is computed and persisted, but was **recorded, not enforced**: it
   never reached the evaluator. Running many forward tests at once and taking the one that passes is
   multiple testing, uncorrected.

## Design decision — a tighten-only Sharpe tax (issue option b, minimal)

Promote both signals into `ForwardEvidence`, combine them into an effective trial count, and **raise
the realized-Sharpe bar by an additive log penalty**:

```
effective_trials = max(1, n_prior_forward_looks + n_concurrent_forward)
penalty          = MT_SHARPE_PENALTY * ln(effective_trials)          # MT_SHARPE_PENALTY = 0.05
bar              = max(degradation_factor * holdout_sharpe, sharpe_floor) + penalty
passes          <=> realized_sharpe >= bar
```

Properties, all of which a reviewer can check directly against the code:

- **Exact no-op on the honest first solo look.** `n_prior_forward_looks == 0` and
  `n_concurrent_forward == 1` give `effective_trials == 1`, `ln(1) == 0`, `penalty == 0`. The bar is
  byte-for-byte the pre-#431 bar. **Zero regression** on the single-shot path.
- **Monotone tighten-only.** The penalty is `>= 0`, increases in both inputs, and is **added ON TOP**
  of the criteria-derived bar inside the SAME `realized_sharpe` check — it is not a
  `ForwardGateCriteria` knob, so a human's *tighter* criteria still gets the tax added and an agent
  can never subtract it. `MT_SHARPE_PENALTY` is a protected module constant, not agent-tunable.
- **Additive-log, not a confidence model.** This is a deliberately simple, conservative operating
  penalty — NOT a probabilistic-Sharpe / DSR / FDR guarantee. It makes no claim to bound a numeric
  family-wise error rate; it claims only to move the bar up monotonically as re-looks and concurrency
  grow, and to reduce to the status quo at one trial. `0.05` per natural-log trial is a calibration
  choice (e.g. 10 trials => +0.115 Sharpe, 20 trials => +0.150), tunable only by a human editing the
  protected constant.

The penalty rides in the existing `realized_sharpe` check payload as `effective_trials`,
`n_prior_forward_looks`, `n_concurrent_forward`, and `multiple_testing_penalty`, so the audit row and
CLI payload show exactly how much tax the run was under. No new check name, no `SCHEMA_VERSION` bump
(`n_concurrent_forward` is already a column; the rest ride in `decision_json`).

### Why not a full LORD++ ledger / PSR model for the forward gate (issue option a)

- **Statistical mismatch.** LORD++/FDR controls false *discoveries* across a stream of **distinct**
  hypotheses. Sequential re-runs of the *same* strategy+identity on a growing window are repeated
  **looks** at one hypothesis, not distinct hypotheses — FDR is the wrong tool for optional stopping.
  The forward gate also has no parameter *sweep*, so there is no trial-variance to build a DSR `SR*`
  from; the research gate's DSR machinery does not map.
- **Anti-scaling (the #324 lesson).** A lifetime-cumulative online level over forward evaluations
  would recreate exactly the pathology #324 fixed for the research gate: the live wall *mandates*
  periodic re-certification (certificate must be `<= 10` sessions old), so routine, required re-runs
  would ratchet a lifetime level toward 0 and eventually make the gate unpassable — punishing a
  strategy for *complying* with the freshness wall. The horizon bound below is what makes this tax
  immune to that.

## The two inputs

### `n_concurrent_forward` (breadth)

Already assembled: `SELECT COUNT(DISTINCT strategy) FROM tick_snapshots WHERE lane='paper' AND
recorded_at BETWEEN <window_start> AND <now>` — distinct strategies with any paper-lane ticks
overlapping the window (the strategy counts itself, so it is `>= 1` whenever there are observations).
It is a **platform-global** count, an intentional conservative superset of a family-scoped count: it
can only OVER-count concurrent pass-opportunities, never under-count. This value was already computed
and persisted; #431 simply threads it into the evaluator (removing the stale "recorded, not yet
enforced" note).

### `n_prior_forward_looks` (optional stopping) — horizon-bounded, identity-exact-match

```sql
SELECT COUNT(*) FROM forward_gate_evaluations
 WHERE strategy_id = ? AND code_hash = ? AND config_hash = ? AND dependency_hash = ?
   AND created_at >= ?          -- horizon cutoff
```

- **Horizon bound (the #324 anti-scaling fix — a real correctness fix in this pass).** Only looks
  whose `created_at` is on or after the session `FORWARD_RELOOK_HORIZON_SESSIONS` (= 10) trading
  sessions before `now` count. The cutoff is computed from the injected `SessionCalendar`
  (`session_on_or_before(now)` then `previous_session` walked 10 times) as a fixed ISO string, and
  compared lexically against `created_at` (the established pattern for the other `created_at >= ?`
  ledger reads). **Without this bound the count grew without limit**, so mandatory periodic
  certificate refreshes would accumulate looks forever and eventually push the bar out of reach —
  the exact #324 pathology. With it, the tax is burst-rate-limiting: clustered re-runs are taxed and
  looks age out after a horizon. The horizon is aligned with `CERTIFICATE_FRESH_SESSIONS = 10` so
  routine re-certification contributes at most ~1 in-horizon look. We claim nothing about controlling
  sequential false-pass over *unbounded* time — that is the forced trade-off #324 documents, and the
  horizon is the principled point on it.

- **Scope: identity-exact-match, a documented narrower v1 (NOT lineage-scoped).** The count keys on
  an EXACT `code + config + dependency` hash match under one `strategy_id`. This is a deliberate v1
  limitation, stated honestly:
  - It **does** tax the dominant optional-stopping pattern: an agent re-running `paper promote` on
    the SAME fixed artifact across sessions, hoping the growing window clears the bar.
  - It does **NOT** close the re-registration / code-churn escape hatch. Copying a peeked strategy to
    a new registered name (new `strategy_id`) or editing any byte of code (new `code_hash`) resets
    the count to 0. A code change resets it anyway by design, because it forces a fresh
    `research promote` pass first; a pure re-registration under a new name is the residual hole.
    Closing it would require walking the `#222` family DAG (return-correlation clustering assigns an
    economic clone into the same family/component); that lineage-scoped count is **deferred follow-up
    and is NOT implemented here.** This doc makes no escape-hatch-closure claim for v1.
  - A `None` `dependency_hash` (lockfile absent) matches nothing (`= NULL` never matches) and the
    holdout check already fails closed there, so the count is left 0.

- **Ordering.** `assemble_forward_evidence` runs BEFORE the new row is recorded, so the count is of
  PRIOR rows only and never includes the in-flight evaluation.

## Residual race (documented, not closed in v1)

The `n_prior_forward_looks` count is a plain read in `assemble_forward_evidence`; the row for the
current run is inserted later, in a separate write transaction. Two `paper promote` runs on the same
identity that interleave (read, read, insert, insert) can both observe the same look count `L` and
both pass on a stale tax. This residual is stated honestly and accepted for v1 because it is bounded
and tighten-only in the safe direction:

- It can only make the tax too **small** for that one race window — it can never spuriously FAIL an
  honest run.
- The worst-case under-count equals the number of truly-concurrent racing promotes **of a single
  identity**, a rare operator pattern (an agent drives one strategy's gate serially).
- It is self-healing: the next run counts both committed rows and re-applies the (now larger) tax.

Fully closing it means moving the count + evaluate + insert into one `BEGIN IMMEDIATE` critical
section (the pattern `record_gate_with_fdr_and_maybe_promote` / `reserve_holdout` use for the
research gate and holdout). That in-lock accounting store method is **deferred follow-up**; v1
accepts the documented residual rather than shipping an unbuilt method. The concurrent-breadth axis
(`n_concurrent_forward`) is likewise a lock-free snapshot with the same character.

## Changes

- **`algua/research/forward_gates.py`** (CODEOWNERS-protected):
  - New protected wall constants: `MT_SHARPE_PENALTY = 0.05` (per-natural-log-trial Sharpe tax) and
    `FORWARD_RELOOK_HORIZON_SESSIONS = 10` (trailing look-count horizon, aligned with
    `CERTIFICATE_FRESH_SESSIONS`). Both documented as tighten-only walls, not `ForwardGateCriteria`
    knobs.
  - `ForwardEvidence` gains `n_prior_forward_looks: int` and `n_concurrent_forward: int`.
  - `evaluate_forward_gate`: the performance branch computes
    `effective_trials = max(1, n_prior_forward_looks + n_concurrent_forward)`,
    `penalty = MT_SHARPE_PENALTY * ln(effective_trials)`, and raises the bar by `penalty`, attaching
    the four audit keys to the `realized_sharpe` check. The fail-closed branches (holdout `None` /
    non-finite, non-finite criteria) are unchanged and emit no penalty key.
- **`algua/registry/forward_promotion.py`** (CODEOWNERS-protected):
  - `SessionCalendar` protocol gains `previous_session`.
  - `assemble_forward_evidence`: computes the horizon cutoff and the horizon-bounded,
    identity-exact-match `n_prior_forward_looks` count (step 8b); threads both counts onto
    `ForwardEvidence` and `AssembledEvidence`. The stale "recorded, not yet enforced" concurrency
    note is superseded by the enforcement path. The scope + residual-race notes above live inline as
    comments.
- **`algua/cli/paper_cmd.py`** (CODEOWNERS-protected): the `paper promote` payload surfaces
  `n_prior_forward_looks` alongside the existing `n_concurrent_forward`.

## CODEOWNERS / merge note

`CODEOWNERS` protects `algua/research/forward_gates.py`, `algua/registry/forward_promotion.py`, and
`algua/cli/paper_cmd.py` (all touched). **This PR touches CODEOWNERS-protected paths and MUST stay
OPEN for human merge** — it may not auto-merge even on green CI. That is correct: the change
strengthens the last statistical wall before the live gate.

## Deferred / not in v1 (explicit)

- **Lineage-component scoping of `n_prior_forward_looks`** (walk the `#222` family DAG so a
  re-registration / clone shares the recent-look budget). v1 is identity-exact-match; the
  re-registration hatch is open.
- **In-lock recompute-and-insert** to close the read-then-insert race (and the concurrent-breadth
  snapshot race). v1 documents the residual and accepts it.
- **A probabilistic-Sharpe (PSR/DSR) confidence-floor formulation** instead of the additive-log
  Sharpe tax. v1 is the simpler tighten-only tax with no formal FWER claim.
- **A `--family-alpha` / `--mt-penalty` human relaxation flag.** The penalty is a protected constant;
  no CLI knob is wired.

## Test plan (implemented)

- `tests/test_forward_gates.py`:
  - `MT_SHARPE_PENALTY == 0.05` pinned.
  - clean first solo look (`0` prior, `1` concurrent) => `effective_trials == 1`, penalty `0.0`, bar
    unchanged.
  - prior looks and concurrency each raise the bar monotonically; a realized Sharpe just below the
    raised bar fails, just above passes.
  - the penalty is added even on top of an agent's TIGHTER criteria (not relaxable).
  - the holdout-`None` fail-closed branch is unaffected and emits no penalty key.
- `tests/test_forward_promotion.py`:
  - no prior evals => 0; matching-identity prior evals counted; other-identity (code/dep hash) not
    counted.
  - **horizon:** a look older than `FORWARD_RELOOK_HORIZON_SESSIONS` ages out; one within counts; the
    exact horizon-cutoff session boundary is inclusive (`created_at >= cutoff`).

Fast per-task check: `uv run ruff check . && uv run mypy algua && uv run lint-imports && uv run
pytest -q tests/test_forward_gates.py tests/test_forward_promotion.py`. Full gate at integration.
