# Plan — Agent cold-start family #0 (#532)

Design: `docs/superpowers/specs/2026-07-27-agent-cold-start-family-532-design.md`
CODEOWNERS: `promotion.py` + `store.py` → PR stays OPEN for human merge.
No schema bump. Depends on #524 (must be merged first).

Per-task FAST check (during Implement):
`uv run ruff check . && uv run mypy algua && uv run lint-imports && uv run pytest -q <this task's tests>`
FULL gate ONLY at integration (Task 5).

## Task 0 — GATE-1 re-run (Codex, read-only)
Re-run the adversarial GATE-1 review on the round-2 design (this doc's spec). APPROVE gate before any
code. `codex exec -s read-only "<inline design>" < /dev/null`.

## Task 1 — `family_count()` accessor
- Add `family_count(self) -> int` to `StrategyRepository` Protocol (`algua/registry/repository.py`).
- Implement in `SqliteStrategyRepository` (`algua/registry/store.py`): `SELECT COUNT(*) FROM families`.
- Test (Finding 4.7): 0 on fresh DB; increments on create; counts a family whose members are all
  soft-deleted (table count, not active-member count). AS SHIPPED: `test_family_count_raw_table_count`
  in `tests/test_registry_store.py` (no dedicated module).
- FAST check: `-k family_count` + the store test.

## Task 2 — promotion.py Finding 1 (precise cold-start gate)
- In `_classify_and_assign_family`, NOVEL + `actor is Actor.AGENT`: replace `if not _has_any_family:
  raise` with `if not _has_any_family and repo.family_count() > 0: raise <no-active-member error>`;
  fall through to the existing `PendingNovelFamily` deferral for cold-start (count==0) AND normal.
- Keep `_has_any_family` (still distinguishes state (b)). Read `family_count()` inside the
  fp_before…fp_after window (covered by existing CAS).
- Tests: 4.1 (empty→mints #1 on PASS, store seam `test_cold_start_pass_mints_family_zero`), 4.3
  (families exist, zero active members → fail closed,
  `test_agent_novel_families_exist_but_no_active_member_fails_closed`), 4.6 (human `--new-family`
  cold-start unchanged, pre-existing), plus `test_agent_novel_cold_start_defers_on_empty_registry`.
- FAST check: `tests/registry/test_family_creation_guard.py` + `tests/test_registry_store.py`.

## Task 3 — promotion.py Finding 3a + store 3b/3c docs
- Add `if repo.agent_novel_mint_seed() <= 0: raise ValueError(...)` to `_revalidate_pending_novel`
  (pure read, pre-peek). Keep the under-lock guard in `_mint_agent_novel_family` unchanged.
- store.py: doc note near `check_agent_novel_mint_bounds` — rate cap is per connected registry DB
  (3b). Design already records the founder-gate-row `family_id=NULL` decision (3c) — no code change.
- Tests: 4.2 (gate FAIL → mints nothing, `test_cold_start_fail_mints_nothing`), 4.5 (seed==0 rejected
  at preflight before the holdout peek, `test_agent_novel_seed_zero_rejected_at_preflight_before_holdout`
  — a direct `_revalidate_pending_novel` call, since through full preflight the empty-`search_trials`
  case is shadowed by the breadth gate; see design §3 Finding 4.5).
- FAST check: `tests/registry/test_family_creation_guard.py` + `tests/test_registry_store.py`.

## Task 4 — concurrency test + migrate incidental existing tests
- Test 4.4: two-agent cold-start → loser `FamilyGraphDriftError`, exactly one family, no double-found
  — AS SHIPPED: `test_cold_start_concurrent_double_founding_prevented` in `tests/test_registry_store.py`
  drives it at the under-lock CAS seam directly (a concurrent founder mutates the graph between the
  pending-spec capture and commit).
- Migrate `tests/test_fundamentals_guards.py`, `tests/test_promotion_needs_fundamentals.py`,
  `tests/test_promotion_needs_news.py`: they asserted the removed `match="family registry is empty"`
  raise incidentally; re-point each to the guard it actually tests (§6 of the design).
- FAST check: the four affected test files.

## Task 5 — integration + PR
- FULL gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
- Scoped `git add` (never -A). Branch push alone; `gh pr create` separately.
- CODEOWNERS-protected (`promotion.py`/`store.py`) → PR stays OPEN for human merge (no auto-merge).
