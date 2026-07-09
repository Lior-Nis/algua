# Forward-gate realized-Sharpe lower confidence bound (#432) — design

Status: design — describes the change **as actually shipped** in this PR (an analytic,
non-normality-adjusted Sharpe lower-confidence-bound wall). Since #431 (multiple-testing Sharpe
tax) landed on `main`, this branch was merged with it: the shipped code **composes** #431's
`MT_SHARPE_PENALTY` into the shared performance bar, so BOTH the point-estimate check and the LCB
wall are held to the MT-taxed bar (see "Scope & explicitly deferred work" for what remains
deferred — the broader bootstrap LCB and the family alpha-spending ledger).

Issue: **#432** (`[ds]` — "Forward-gate `realized_sharpe` is a point-estimate bar on as few as 63
observations with no confidence lower bound").

> **Honesty note.** Earlier drafts of this doc described a stationary-bootstrap LCB plus a
> family-scoped alpha-spending ledger (a new serialized `store.py` committer, a HISTORICAL-lineage
> CTE, concurrency CAS, etc.). **That richer bootstrap/ledger machinery is still NOT built** and
> remains deferred. What IS built and shipped here is the analytic LCB in this diff, composed with
> #431's already-merged additive-log multiple-testing Sharpe tax (folded into the shared bar); the
> stationary-bootstrap LCB and the full alpha-spending ledger remain a genuinely future design doc.

## Problem

The forward gate (`paper -> forward_tested`, the last automated wall before a strategy can be
human-approved to `live`) passes its performance check on a **point estimate**:

    realized_sharpe >= bar,   bar = max(DEGRADATION_FACTOR * holdout_sharpe, SHARPE_FLOOR)

`realized_sharpe` is one draw of the annualized Sharpe over a daily series as short as
`MIN_FORWARD_OBSERVATIONS = 63` sessions (`forward_promotion.py` — `metrics_from_returns` over the
admissible tick equity curve). The Sharpe t-statistic is annualization-invariant: at n=63 an
*observed* annualized Sharpe of 1.0 is only `t = (1.0/√252)·√63 ≈ 0.50` — well under one standard
error from zero, let alone from the bar. A strategy whose true Sharpe is at or below the bar can
clear the point check on a single favorable short window.

## Fix (this PR): an analytic lower-confidence-bound wall against the SAME bar

Add one AND-check, `realized_sharpe_lcb`, next to the existing point check. It requires the
**one-sided lower confidence bound (LCB)** on the realized annualized Sharpe — at
`FORWARD_SHARPE_CONFIDENCE = 0.95` — to clear the **same** performance `bar` the point estimate is
held to. Both checks are ANDed into `decision.passed`.

The LCB uses the **Lo(2002)/Mertens non-normality Sharpe standard error** — the same variance term
`algua.research.gates` / `algua.research.dsr.dsr_confidence` use — kept self-contained (stdlib
`statistics.NormalDist`) so this pure wall does **not** import the scipy-backed DSR path:

    SR   = sharpe_ann / sqrt(ANN)                     # de-annualize to per-period (ANN=252)
    var  = 1 - skew*SR + ((rawKurt - 1)/4) * SR^2     # Lo/Mertens SE^2 numerator (skew/kurt adj.)
    SE   = sqrt(var / (T - 1))                        # per-period Sharpe standard error
    lcb  = (SR - z_confidence * SE) * sqrt(ANN)       # re-annualize the lower bound
    PASS iff  lcb >= bar

`z_confidence = NormalDist().inv_cdf(confidence)`. `skew` and `rawKurt` are the realized skewness
and **raw** Pearson kurtosis (~3 for a Gaussian series) of the forward return series, so a
fat-tailed / skewed series is penalized exactly as in the holdout DSR.

**Why the LCB is compared to the bar, not to zero (the core correctness point).** An intermediate
implementation computed a Probabilistic Sharpe Ratio against a `SR* = 0` benchmark, i.e. it only
checked that the true Sharpe is confidently **> 0**. That is redundant-at-best: the point check
already asserts the *level* (`realized_sharpe >= bar`), so the significance wall must ask the
*harder* question — is the true Sharpe confidently **>= the bar**. Testing `lcb >= bar` (equivalently
a PSR with `SR* = bar/sqrt(ANN)`) is that question. Concretely, at n=63 with a Gaussian series a
realized Sharpe of 3.4 has an LCB of ≈ 0.046 — it clears a zero benchmark but is nowhere near the
0.5 bar, so the corrected wall (rightly) fails it. This is the #432 regression case.

**Fail-closed on degenerate input.** `_forward_sharpe_lcb` returns `None` (→ the check fails
closed) when: `n_obs <= 1` (no `sqrt(T-1)`), any non-finite moment, a non-finite/≤0 variance term,
a non-finite/out-of-(0,1) confidence (`inv_cdf` blows up at the 0/1 edges), or a non-finite result.
When the **bar itself** is unavailable (no qualified holdout row, or non-finite criteria) the
performance question is undefinable, so `realized_sharpe_lcb` fails closed for the same reason the
point `realized_sharpe` check does. In every fail-closed branch the audit `value`/`threshold` are
scrubbed to `None` when non-finite (same rule `_metric_check` applies), keeping `decision_json`
JSON-clean.

**Power tradeoff (n controls power; α controls protection).** `MIN_FORWARD_OBSERVATIONS = 63` stays
a hard floor. Clearing even the 0.3 floor bar at the LCB needs an observed annual Sharpe of ~3.8 at
n=63; the sanctioned remedy for a marginal strategy is a **longer** forward window (the SE shrinks
with T — the gate is re-runnable and accumulates evidence), never a weaker bar. e.g. a realized
Sharpe of 2.0 fails the LCB at n=63 but clears it by ~n=1000.

## Threshold direction & authentication

`FORWARD_SHARPE_CONFIDENCE = 0.95` joins `ForwardGateCriteria` as `forward_sharpe_confidence`. It is
a **protected wall**: an agent may only **raise** it (stricter); lowering is human-only, enforced by
`guard_forward_relaxations` (higher-is-stricter set). `guard_forward_relaxations` additionally
**validates the value is a finite probability in (0, 1) for BOTH actors** and raises `ValueError`
otherwise — a `nan`/`inf`/`≤0`/`≥1` confidence would slip past the tighten-only comparison
(`nan < 0.95` is `False`) and blow up `inv_cdf`, so it fails closed at the guard boundary rather
than silently downstream. The confidence is included in the `ForwardGateCriteria` the #329
human-actor challenge binds (now 8 thresholds), and surfaces as a human-only
`--forward-sharpe-confidence` CLI flag on `paper promote`.

## Files touched — CODEOWNERS status

- `algua/research/forward_gates.py` — new constant `FORWARD_SHARPE_CONFIDENCE = 0.95`; new
  `ForwardEvidence` fields `realized_skew`, `realized_kurtosis`; new pure helper
  `_forward_sharpe_lcb(...)`; new `realized_sharpe_lcb` check (LCB vs bar) in
  `evaluate_forward_gate`, with the performance `bar` computed once and shared by the point and LCB
  checks; `ForwardGateCriteria.forward_sharpe_confidence`. **Policy-protected safety wall.**
- `algua/registry/forward_promotion.py` — `assemble_forward_evidence` populates `realized_skew` /
  `realized_kurtosis` from `metrics_from_returns`; `guard_forward_relaxations` gains the
  finite-and-in-(0,1) validation + the tighten-only entry for `forward_sharpe_confidence`.
  **Policy-protected safety wall.**
- `algua/cli/paper_cmd.py` — human-only `--forward-sharpe-confidence` relaxation via the #329 flow;
  the flag value is bound into the authenticated challenge. Not on the protected list, but bundled
  in the same human-merge PR.
- Tests: `tests/test_forward_gates.py` (LCB-vs-bar wall: the #432 regression at realized 3.4 /
  n=63; baseline pass; fail-closed on n≤1; more-observations rescue; agent-tightening allowed;
  threshold tracks the bar), `tests/test_forward_promotion.py` (guard tightening/relaxation for the
  confidence; `nan`/`inf`/out-of-range confidence rejected for both actors).

Because `forward_gates.py` and `forward_promotion.py` are policy-protected safety walls (per the
workflow operating rules), the PR **stays OPEN for human merge** (no auto-merge) even with green CI.
No other protected path is modified.

## SCHEMA_VERSION — no bump

No `SCHEMA_VERSION` bump. The LCB, its confidence, the realized moments and the `realized_sharpe_lcb`
check all ride inside the existing `gate_evaluations.decision_json` blob / in-memory
`ForwardEvidence`; no column or table changes.

## Scope & explicitly deferred work (NOT in this PR)

This PR fixes the **single-look** defect only. The following are **not implemented here** and are
deferred to a future design doc:

- **#431 — optional stopping & concurrency (NOW COMPOSED, not deferred).** Since #431 landed on
  `main`, this branch merged it: the horizon-bounded look count (`n_prior_forward_looks`) plus
  `n_concurrent_forward` drive `MT_SHARPE_PENALTY * ln(effective_trials)`, which is folded into the
  shared performance bar — so the point-estimate check AND this PR's LCB wall are BOTH taxed for
  optional stopping and breadth. What remains genuinely deferred is the richer version: a
  cross-look **alpha-spending ledger** and **family/lineage-scoped** look counting committed under
  the write lock (#431 ships an identity-exact, additive-log tax with a documented residual race).
- **Serial-correlation robustness.** The analytic Lo/Mertens SE assumes weak dependence; daily
  strategy returns are autocorrelated, so the LCB's coverage is approximate. A
  stationary-bootstrap LCB (reusing `algua/backtest/bootstrap.py`, the DSR gate's instrument) is a
  future tightening behind the same `realized_sharpe_lcb` seam — a follow-up, not a blocker.

## Honest guarantee

The wall adds a non-normality-adjusted, **single-look** lower confidence bound on the realized
forward Sharpe, tested against the holdout-derived performance bar (which, post-merge, already
carries #431's optional-stopping/concurrency Sharpe tax). It defends against a lucky short window
clearing the point bar. Beyond #431's additive-log tax it does **NOT** provide a full
alpha-spending / family-wise ledger, and its coverage is approximate under autocorrelation.
Live-wall safety
continues to rest on **defense-in-depth**: the LCB is one AND-check among the vol floor, drawdown
cap, integrity, account-hygiene and staleness checks; the bar is holdout-derived; and
`forward_tested -> live` additionally requires a verified human signature over a fresh,
identity-matched certificate.

## Quality gate

Fast per-task loop: `uv run ruff check . && uv run mypy algua && uv run lint-imports && uv run
pytest -q tests/test_forward_gates.py tests/test_forward_promotion.py`. Full
`pytest -q && ruff check . && mypy algua && lint-imports` at integration. `lint-imports` unaffected:
`forward_gates.py` gains no new import (stdlib `math`/`statistics` only); `forward_promotion.py`
gains a stdlib `math` import.
