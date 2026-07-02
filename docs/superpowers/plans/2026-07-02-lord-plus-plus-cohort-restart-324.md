# Plan — LORD++ cohort restarts (#324)

Spec: docs/superpowers/specs/2026-07-02-lord-plus-plus-cohort-restart-324.md
Quality gate each task: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

## Task 1 — gates.py: cohort constant + pure helper + docstrings (PURE)
- Add `FDR_COHORT_SIZE = 64` beside FDR_ALPHA/FDR_W0 with the protected-constant + re-scoped-target
  rationale in the comment.
- Add pure helper `fdr_cohort_position(k: int) -> tuple[int, int]` → (cohort_index, within_cohort_t)
  for a 1-based global binding ordinal k; fail-closed ValueError on k < 1.
  cohort = (k-1)//FDR_COHORT_SIZE; t = (k-1)%FDR_COHORT_SIZE + 1.
- Update module-level LORD++ docstring + `lord_plus_plus_level` docstring: FDR is now controlled
  PER COHORT of FDR_COHORT_SIZE binding tests (NOT lifetime); dry-spell alpha floored at
  gamma_{N}*W0; lord_plus_plus_level itself unchanged (per-cohort t/taus supplied by caller).
- Tests: fdr_cohort_position boundaries (k=1→(0,1); k=64→(0,64); k=65→(1,1); k<1 raises).

## Task 2 — repository.py: FdrStreamState + Protocol
- Extend FdrStreamState to carry cohort scoping + exposure: add fields
  `cohort_index: int`, `cohorts_completed: int`, `binding_tests: int`, `discoveries: int`.
  `t` and `discovery_indices` become CURRENT-COHORT-scoped (within-cohort position + in-cohort
  rejection positions). Update docstring.
- No Protocol signature change to fdr_stream_state() (still no args) — it computes the NEXT test's
  cohort from the binding-row count internally. record_gate_with_fdr_and_maybe_promote unchanged sig.

## Task 3 — store.py: cohort-scoped stream read + atomic write + migration wiring
- fdr_stream_state(): SELECT binding rows (fdr_cohort, fdr_test_index, fdr_p_value, fdr_alpha_level,
  fdr_rejected) ORDER BY id. Fail-closed validation as today PLUS: within the CURRENT cohort the
  fdr_test_index must be contiguous 1..t. Derive:
    binding_tests = len(rows); discoveries = total rejected==1
    next global ordinal k = binding_tests + 1; but SEAL legacy/partial: next cohort = if rows
    exist, (max stored fdr_cohort of the LAST row) → the next test joins the SAME cohort iff that
    cohort isn't full AND was produced under the new scheme; simplest deterministic rule:
    cohort_of_next = ceil-based from the count of rows already IN the highest cohort.
    IMPLEMENTATION: group rows by fdr_cohort; the "current" cohort = max(fdr_cohort); its rows give
    t_current + in-cohort taus; if that cohort is FULL (== FDR_COHORT_SIZE) the next test opens
    cohort+1 (t=1, no taus). cohorts_completed = number of FULL cohorts.
  Return FdrStreamState(t=<within-cohort position for NEXT test>, discovery_indices=<in-cohort
    taus for the cohort the next test lands in>, cohort_index=<that cohort>, cohorts_completed,
    binding_tests, discoveries).
- record_gate_with_fdr_and_maybe_promote(): inside BEGIN IMMEDIATE use the returned cohort_index +
  t for the INSERT (fdr_cohort=cohort_index, fdr_test_index=t). alpha_t = level_fn(t, in-cohort
  taus). Patch decision_json fdr_* AND a new fdr_exposure block:
    fdr_cohort, fdr_cohorts_completed, fdr_binding_tests(=binding_tests+1 after this insert or the
    pre-insert count — pick pre-insert count of PRIOR tests; document), fdr_discoveries,
    fdr_expected_false_discoveries = FDR_ALPHA * cohorts_completed.
  NOTE: FDR_ALPHA must be available to store.py WITHOUT importing algua.research (lint contract).
  Pass alpha via the injected level_fn's closure is NOT enough (need the scalar). Options: (a) pass
  `fdr_alpha: float` param into record_gate_with_fdr_and_maybe_promote from promotion.py (which
  already imports FDR_ALPHA) — CLEANEST, keeps the import boundary. Use (a).
- FdrGateOutcome: add cohort_index + exposure fields so promotion.py can surface them.

## Task 4 — db.py: schema 31→32 migration
- Add column fdr_cohort INTEGER to gate_evaluations (NULL legacy).
- DROP INDEX ix_gate_evaluations_fdr_index; CREATE UNIQUE INDEX ix_gate_evaluations_fdr_cohort_index
  ON gate_evaluations(fdr_cohort, fdr_test_index) WHERE fdr_binding=1.
- One-time backfill (ordered by id) over WHERE fdr_binding=1: for the g-th binding row (g 1-based),
  set fdr_cohort=(g-1)//64, fdr_test_index=(g-1)%64+1. Frozen fdr_alpha_level untouched.
  Use FDR_COHORT_SIZE — but db.py must not import algua.research. Hardcode 64 with a comment
  referencing gates.FDR_COHORT_SIZE (constants file already duplicates this pattern? verify). If a
  shared-constant import is cleaner and lint-clean, use it; else hardcode + comment + a test that
  asserts db's literal == gates.FDR_COHORT_SIZE.
- Bump SCHEMA_VERSION=32; PRAGMA user_version.

## Task 5 — promotion.py: pass fdr_alpha + surface exposure
- record_gate_with_fdr_and_maybe_promote(..., fdr_alpha=FDR_ALPHA).
- Fold fdr_exposure fields into GateDecision (new fields) + to_dict().

## Task 6 — gates.py GateDecision: exposure fields + to_dict
- Add fdr_cohort, fdr_cohorts_completed, fdr_binding_tests, fdr_discoveries,
  fdr_expected_false_discoveries fields (audit-only). Wire into to_dict().

## Task 7 — CLAUDE.md + tests
- CLAUDE.md research-promote paragraph: LORD++ now per-cohort (N=64), FDR per cohort not lifetime,
  cumulative exposure surfaced.
- Tests: cohort boundary in the real DB (65th binding test opens cohort 1, t=1); migration
  backfill correctness on a seeded legacy DB; concurrent binding evals in the same cohort get
  distinct serialized t (extend existing concurrency test); exposure fields present + correct;
  anti-scaling regression: a long run of failing binding tests keeps alpha_t bounded >= gamma_64*W0
  (never collapses toward 0 as in the old lifetime stream).

## Verify + PR
- Full quality gate green. git fetch origin && merge origin/main. Push branch alone. gh pr create.
- GATE-2 Codex on the diff. Post verdict comment. Leave OPEN (gates.py protected).
