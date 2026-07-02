# Plan — Issue #324: ADDIS replaces LORD++ (fix anti-scaling FDR ledger)

**Date:** 2026-07-02 · **Issue:** #324 (CRITICAL) · **Schema:** 31 → 32
**Design:** GATE-1 APPROVED (Codex) — ADDIS* (Tian & Ramdas 2019, arXiv:1905.11465), hard epoch.

## Goal
Replace the anti-scaling LORD++ lifetime-global alpha-wealth ledger with ADDIS*, whose adaptive
DISCARDING of very-conservative nulls (p > tau) means failed garbage explorations no longer decay
the promotion bar. FDR still controlled at 0.05. Fresh ADDIS epoch; legacy LORD rows excluded.

## The ADDIS* recursion (verified vs paper Algorithm 1 / Appendix J, raw-p scale)
Given ordered prior binding p-values p_1..p_{n} of the ADDIS epoch, the level for the NEXT test
(position n+1) is computed by replaying:
- candidate C_i = 1{p_i <= LAMBDA}; selected 1{p_i <= TAU}; discard p_i > TAU.
- rejection R_i = 1{p_i <= alpha_i} where alpha_i is itself the ADDIS level over p_1..p_{i-1}.
- alpha_t = min(LAMBDA, (TAU - LAMBDA) * (W0*gamma[S_t - C0plus] + (ALPHA-W0)*gamma[S_t - k1* - C1plus]
  + ALPHA * sum_{j>=2} gamma[S_t - kj* - Cjplus])), 0-indexed gamma; kappa terms absent until they exist.
Constants: ALPHA=0.05, W0=0.025, LAMBDA=0.25, TAU=0.5. gamma = existing _LORD_GAMMA (nonneg,
nonincreasing, sum~1 — valid for ADDIS).

## Tasks (TDD, one subagent per task, gate between)

### Task 1 — Pure ADDIS math in gates.py (PROTECTED)
- Add constants ADDIS_LAMBDA=0.25, ADDIS_TAU=0.5 with protected-constant docstrings.
- Add `addis_level(prior_p_values: Sequence[float], *, alpha, w0, lam, tau) -> float`:
  replays the epoch recursively (computes each prior alpha_i and R_i internally), then returns the
  level for the next test. Fail-closed: non-finite alpha/w0/lam/tau, lam>=tau, w0>alpha, lam<alpha,
  any non-finite prior p -> return 0.0 (only tightens). Reuse `_gamma(j)` helper (0-indexed here).
- REMOVE `lord_plus_plus_level` (no dual paths — memory: no backwards-compat cruft). Keep
  `_compute_lord_gamma`/`_LORD_GAMMA` (rename docstring to "gamma weights"; still used by ADDIS).
- Tests (test_research_gates.py): rewrite the LORD test block for ADDIS —
  * constants; gamma unchanged asserts kept.
  * alpha_1 (empty history) == min(LAMBDA, (TAU-LAMBDA)*W0*gamma[0]).
  * DISCARD-SPAM INVARIANCE (the headline): [0.1] + [0.9]*K + next gives the SAME level for the
    next candidate for K=0 and K=50 (conservative nulls don't decay the bar). GOLDEN.
  * (lambda,tau] band DOES advance the clock (p=0.4 spam lowers next level) — proves not a loosening.
  * first/second rejection replenish (a rejecting prefix raises the next level).
  * manual recursion check on a small hand-worked stream (values pinned).
  * fail-closed guards (non-finite p/params, lam>=tau).
  * recompute==recorded consistency property (alpha over prefix i-1 is deterministic).

### Task 2 — Schema 31->32: fdr_algo column (db.py)
- SCHEMA_VERSION 31 -> 32. `_add_missing_columns(conn,"gate_evaluations",{"fdr_algo":"TEXT"})`
  with a comment mirroring the v26 FDR-columns note (legacy LORD rows NULL, excluded from ADDIS
  epoch; ADDIS rows stamped 'addis_v1'). Add "fdr_algo" to the base _SCHEMA gate_evaluations DDL too
  (so fresh DBs have it). Test in test_registry_db.py: column exists, version==32.

### Task 3 — Epoch-scoped stream read + FdrStreamState (repository.py + store.py)
- FdrStreamState: replace `(t, discovery_indices)` with `(t_global: int, prior_p_values: list[float])`
  where t_global = COUNT of ALL binding rows (both epochs, for fdr_test_index continuity) and
  prior_p_values = ordered fdr_p_value of the ADDIS-EPOCH rows only. Update docstring.
- store.fdr_stream_state: SELECT fdr_p_value FROM gate_evaluations WHERE fdr_binding=1 AND
  fdr_algo='addis_v1' ORDER BY id (epoch p-history) + a separate COUNT(*) WHERE fdr_binding=1 (global).
  Fail-closed: any epoch row with NULL/non-finite fdr_p_value -> None. Drop the old
  discovery_indices/fdr_test_index-contiguity logic (ADDIS derives rejections itself; global
  fdr_test_index still validated for uniqueness by the existing partial index at write).
- repository.py Protocol: update fdr_stream_state + FdrStreamState docstrings; change level_fn type
  to `Callable[[list[float]], float]`.

### Task 4 — Wire ADDIS into the atomic write (store.py) + promotion.py
- record_gate_with_fdr_and_maybe_promote: level_fn type -> Callable[[list[float]], float].
  Inside the lock: `stream=fdr_stream_state(); t_next=stream.t_global+1;
  alpha_t=level_fn(stream.prior_p_values); fdr_rejected = p_value<=alpha_t`. t_next stays the GLOBAL
  audit index. Stamp the INSERT with fdr_algo='addis_v1' when fdr_binding (else NULL). Add fdr_algo
  to the INSERT column list + value. Add "fdr_algo":"addis_v1" to raw_decision when binding.
- promotion.py: level_fn = functools.partial(addis_level, alpha=FDR_ALPHA, w0=FDR_W0,
  lam=ADDIS_LAMBDA, tau=ADDIS_TAU). Import addis_level/ADDIS_LAMBDA/ADDIS_TAU; drop lord import.

### Task 5 — Update store/promotion tests + docs
- test_registry_store.py / test_promotion.py: update any fixtures asserting FdrStreamState shape,
  discovery_indices, or LORD alpha values; the concurrency test (two binding evals get distinct
  serialized fdr_test_index) must still pass (global counter). Add: a binding row is stamped
  fdr_algo='addis_v1'; legacy NULL-algo rows are excluded from the epoch history.
- CLAUDE.md: update the research-promote LORD++ description -> ADDIS (discarding), keep it terse.
- Design doc + this plan committed.

## Quality gate (between every task, FULL suite)
`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

## Conflict-avoidance
gates.py (protected -> PR OPEN), db.py (additive column), store.py (epoch SELECT + INSERT col),
repository.py (Protocol), promotion.py (partial swap), tests, CLAUDE.md. No engine.py (#325); the
schema touch is additive-only, disjoint from #330/#334's promotion-CAS logic.
