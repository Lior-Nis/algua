# Plan — NOVEL family agent seed (#524)

Design: `docs/superpowers/specs/2026-07-18-novel-family-agent-seed-524-design.md` (**R9**)
CODEOWNERS-protected → PR stays OPEN for human merge. One schema bump (36→37, carrying columns +
append-only triggers + the `agent_mint_grants` ledger).

## Task 1 — Schema v37 (columns + append-only triggers + grants) + accessors + n_combos hardening (test-first)
- `db.py`: `SCHEMA_VERSION` 36→37. Fresh `_SCHEMA` + `migrate()` `_add_missing_columns`:
  - `families`: `seeded_prior_combos INTEGER NOT NULL DEFAULT 0 CHECK(seeded_prior_combos>=0)` (fresh only)
    AND **`founder_gate_id INTEGER REFERENCES gate_evaluations(id)`** (R9-M2 founder→gate audit link).
  - `family_members`: **`member_code_hash TEXT` + `member_factors_json TEXT`** (R9-H3 persisted member
    profile, materialised at assignment; immutable once set).
  - New **`agent_mint_grants`** append-only ledger (`granted_at`, `granted_by_actor` CHECK `='human'`,
    `grant_count` CHECK `>=1`, `reason`) — the human-replenished lifetime budget (R9-H2).
  - **Append-only TRIGGERS (R9-H1):** `BEFORE UPDATE`/`BEFORE DELETE` `RAISE(ABORT)` on `families`,
    `family_parents`, `family_events`, `backtest_returns`; `family_members` `BEFORE DELETE` ABORT +
    `BEFORE UPDATE` ABORT except the two one-way flips (`removed_at` NULL→ts; `member_code_hash`/
    `member_factors_json` NULL→value). Also add the type-safe+bounded
    `CHECK (typeof(n_combos)='integer' AND n_combos>=1 AND n_combos<=1000000000)` to the fresh
    `search_trials` DDL. Define `MAX_N_COMBOS = 1_000_000_000` beside the schema constants.
  - **Migration ORDER:** ALTER add columns → store-layer `_materialise_legacy_member_profiles()` (loads
    modules, NULL→value UPDATEs) → CREATE the append-only triggers LAST, so the one-time legacy backfill is
    not self-blocked. Fresh DBs skip the backfill (no legacy members).
- `store.py` `record_search_trial`: validate `type(n_combos) is int and 1 <= n_combos <= MAX_N_COMBOS`
  (fail-closed ValueError; `type() is int` NOT `isinstance` — excludes `bool`).
- `store.py` + `repository.py` (`SearchBreadthLedger`): `funnel_lifetime_search_combos() -> int` = the
  **WHERE-filtered** `SUM(n_combos)` over `typeof='integer' AND n_combos BETWEEN 1 AND MAX_N_COMBOS`
  (R9-M3 filter-before-sum; overflow-safe; agrees exactly with the §5.1 mint seed).
- `store.py` + `repository.py` (`FamilyRepository`): `family_graph_fingerprint() -> tuple[int,...]` = the
  monotone digest over `families` (COUNT,MAX id), `family_members` all-rows (COUNT,MAX id) AND active-only
  COUNT, `family_parents` (COUNT,MAX id), `family_events` (COUNT,MAX id), `backtest_returns` (COUNT,MAX id).
  With member profiles now persisted in immutable `family_members` rows, this single tuple covers the FULL
  classifier read-set (R9-H3). Pure SQL; boundary-clean.
- `store.py` `all_families_with_member_profiles`: read the **persisted** `member_code_hash`/
  `member_factors_json` columns (NOT live `compute_artifact_hashes`/`factors_used_by`) — the classifier's
  member-profile input is now DB state (R9-H3). Legacy rows fall back only until `_materialise_legacy_
  member_profiles()` has run.
- Tests: Task-1 slices of plan items 1, 2-partial, 9-partial, 18, 19, 22-partial — fresh-DB columns +
  CHECKs + grants table + all ten triggers + `user_version=37`; legacy-DB backfill (0/NULL) + profile
  materialisation-before-triggers + idempotent re-migrate; `funnel_lifetime_search_combos` filtered +
  `> windowed_search_combos(90)` + excludes a corrupt row; `record_search_trial` rejects non-int/`<1`/
  `>MAX_N_COMBOS`/`bool`; `family_graph_fingerprint` changes on family mint / member assign / member
  **removal** / parentage edge / `persist_backtest_returns`, stable otherwise; **trigger enforcement**
  (raw UPDATE/DELETE rejected; the two permitted `family_members` flips allowed); **guard test**
  (classifier-read tables ⊆ fingerprint tables; all triggers present).
- FAST check: `ruff && mypy && lint-imports && pytest -q tests/test_registry_db.py tests/test_family_registry.py -k "seed or lifetime or migrat or combos or fingerprint or trigger or profile or grant"`.

## Task 2 — Seed the family_lifetime SUM (deferred-creation arch) + ancestor dedup
- `store.py`: fold the per-family seed into `lifetime_combos_for_families` (SUM `seeded_prior_combos` over
  the ancestor closure, deduped by `set`; ADD to the real member-trial SUM). `family_lifetime_combos`
  picks it up transitively. NO standalone create method (create folds into the promote tx).
- Application-level `seed >= 0` guard on the `seeded_prior_combos` write + the reader treats a negative as
  corruption (migrated DBs lack the ALTER CHECK).
- Tests (plan items 6, 23, adversarial old-trials at the store layer): seed additive + monotone;
  `family_lifetime_combos == seed + members`; **multi-parent / diamond ancestor closure counts each seed
  EXACTLY ONCE** (R9-LOW dedupe).
- FAST check: `... pytest -q tests/test_family_registry.py -k "seed or lifetime or monoton or dedup or ancestor"`.

## Task 3 — Deferred pass-time create in atomic promote tx + budget + NOVEL wiring + grant CLI + CLAUDE.md
- `repository.py`: `PendingNovelFamily` NamedTuple (plain scalars: `slug_base`, actor, verdict, scores,
  clustering json, **`graph_fingerprint: tuple[int,...]`**, **`founder_code_hash`**, **`founder_factors_json`**
  — NO `profile_digest`, it folded into the fingerprint) + a `ClassifyResult(family_id | pending)` shape;
  optional `pending_novel_family` param on `record_gate_with_fdr_and_maybe_promote` (GateLedger). New
  exceptions `FamilyGraphDriftError` (carries a `{still_assigned, graph_fingerprint}` axis), `AgentMintCapError`,
  `AgentMintBudgetExhaustedError`. Constants in **CODEOWNERS-protected `store.py`** (NOT repository.py, NOT a
  flag/env — R9-M1): `AGENT_NOVEL_MINT_WINDOW_DAYS=90`, `AGENT_NOVEL_MINT_CAP=8`,
  `AGENT_NOVEL_MINT_LIFETIME_BUDGET=32`.
- `store.py` `record_gate_with_fdr_and_maybe_promote`: coerce `actor=Actor(actor)` at entry BEFORE the
  store-boundary guard; when `pending_novel_family` set, fail-closed unless `actor is Actor.AGENT` AND
  `pending.actor=='agent'` AND `pending.verdict=='novel'` (literal, no `SimVerdict` import). INSIDE the
  existing single `BEGIN IMMEDIATE`, in order:
  - **step 1 (R3-HIGH + R5-HIGH-1):** if pending, UNDER lock require `strategy_family(name) is None` AND
    `family_graph_fingerprint()==pending.graph_fingerprint` (one digest = whole read-set incl. persisted
    profiles) else rollback + `FamilyGraphDriftError`.
  - #339 CAS; FDR; INSERT gate row with EVALUATED breadth `family_id=NULL, family_lifetime_effective=0`
    — NEVER stamped afterward (R3-MED-1).
  - if `final_passed` AND pending: stage CAS, then
    **5a (R9-M3 filter-before-sum):** seed = `SUM(n_combos) WHERE typeof='integer' AND n_combos BETWEEN 1
    AND 1e9`; a diagnostic `(COUNT(*), well_typed_count)` for observability; require `seed>0` else rollback
    (a corrupt row is EXCLUDED, never a permanent DoS; only all-corrupt/empty fails closed);
    **5b (R5-HIGH-2 + R9-H2 + R9-M1 + R9-LOW UTC):** rate cap `COUNT(*) FROM families WHERE
    created_by_actor='agent' AND created_at>=:cutoff` vs `AGENT_NOVEL_MINT_CAP` (→ `AgentMintCapError`),
    parsing each counted `created_at` as canonical UTC / fail-closed otherwise; AND lifetime budget
    `consumed = COUNT(*) agent families` vs `AGENT_NOVEL_MINT_LIFETIME_BUDGET + SUM(agent_mint_grants.grant_count)`
    (→ `AgentMintBudgetExhaustedError`); both canonical `families`, not events;
    **5c (R6 uuid slug + R3-LOW raw INSERTs + R9-M2 founder_gate_id):** name `f"{slug_base}__{uuid4().hex}"`
    (full 32-hex; regenerate-retry on the rare UNIQUE clash); RAW locked INSERTs — families row
    (seed, actor='agent', created_by_strategy=name, **founder_gate_id=<step-4 gate row id>**),
    family_created event (strategy_name=founder), founding `family_members` row **with the founder's
    persisted `member_code_hash`/`member_factors_json` from the pending spec**, strategy_assigned event.
    Crash before commit rolls everything back.
- `promotion.py` `_classify_and_assign_family`: NOVEL+agent captures `fp_before` at the TOP, computes the
  verdict, captures `fp_after`, requires `fp_before==fp_after` else `FamilyGraphDriftError` (R6-CRITICAL);
  returns `ClassifyResult(family_id=None, pending=...)` with `graph_fingerprint=fp_after` +
  `founder_code_hash`/`founder_factors_json` (the candidate's own classified profile). **No `profile_digest`
  / no `_member_profile_digest`** (R9-H3 removed). MERGE/PARENTAGE assignment INSERTs now persist the
  joining member's profile columns. Human NOVEL + human create paths unchanged (still create-in-preflight,
  also persisting the founding member's profile).
- `promotion.py` `promotion_preflight`: after the pending spec returns and as the LAST step BEFORE the
  holdout peek, pre-check (1) rate cap AND lifetime budget (UTC fail-closed) and (2) still-unassigned +
  `family_graph_fingerprint()==pending.graph_fingerprint`. Both refuse BEFORE any holdout burn / gate row.
- `research_cmd.py` (CLI): **wrap the `on_peek` burn callback so it re-runs the pending-NOVEL revalidation
  (still-unassigned + fingerprint + cap + budget — all pure DB) in the SAME tx as
  `finalize_holdout_reservation`, raising BEFORE finalize on any drift** → reservation stays pending → the
  existing `except` `release_holdout_reservation` frees the window → **no holdout burned for drift caught
  at/before on_peek; the post-peek/pre-lock run_gate re-check is a NARROWED residual — documented and
  monitored (WARNING audit + release-on-failure), not silently closed** (R9-H4). CLI handles
  `FamilyGraphDriftError`/`AgentMintCapError`/
  `AgentMintBudgetExhaustedError` like `FunnelDriftError` (re-run preflight / surface the bound), logging
  strategy name + drift axis.
- `promotion.py` `run_gate`: pre-lock mirror — when pending, require `strategy_family(name) is None` AND
  `family_graph_fingerprint()==pending.graph_fingerprint`; keep the `expected_family_id` CAS otherwise;
  pass `pending_novel_family` to the atomic method; founder evaluates with family arm 0 / family_id None.
- **New human-only `registry grant-novel-mints --actor human --count N` CLI** appending an
  `agent_mint_grants` row (human-actor guarded; the `granted_by_actor='human'` CHECK backstops it).
- `family-audit`: add an `agent_novel_mints` block — `mints_in_window`/`window_cap`/`window_days`,
  `lifetime_consumed`/`lifetime_allowance`/`lifetime_remaining`, `search_trials_corruption_count`.
- `CLAUDE.md`: NOVEL bullet → seeded family AT PASS MOMENT, per-window cap AND human-replenished lifetime
  budget (both fail-closed, CODEOWNERS-constant, human-only to raise; top up via `registry
  grant-novel-mints`) + still-NOVEL graph re-check; human `--new-family`=fresh-zero. Add the
  `grant-novel-mints` command-surface line.
- Tests (plan items 2,3,4,5,7,8,9,10,11,12,13,14,15,16,16b,17,20,21): true-lifetime seed > windowed;
  founder pass no-op (arm 0); FAILED gate mints nothing + never an anchor; future-sibling durability across
  the 90d roll; concurrent NOVEL passes → fingerprint CAS routes related to MERGE/PARENTAGE else distinct;
  idempotent stage guard + uuid slug (no abort with reserved stems) + retry; **filter-before-sum: corrupt
  row EXCLUDED, mint still succeeds, no permanent DoS; all-corrupt fails closed**; pending-NOVEL fingerprint
  drift abort (returns-refresh + parentage + member-removal sub-cases + the R6-CRITICAL capture race);
  **persisted-profile: live source edit cannot flip a member / bumps no fingerprint (R9-H3)**; load-bearing
  rate cap fail-closed canonical `families` + boundary/roll-off/human-not-counted/event-missing; **lifetime
  budget fail-closed + human grant replenishes + agent cannot self-grant (R9-H2)**; **burn-time
  release-on-drift → no holdout burned for drift caught at/before on_peek; post-peek run_gate re-check is a
  narrowed, monitored residual (R9-H4)**; **founder_gate_id query both
  directions (R9-M2)**; **UTC fail-closed on the cap query (R9-LOW)**; **constants are CODEOWNERS store.py
  module constants, no flag/env (R9-M1 scanner)**; actor string-vs-enum coercion + pending-object
  validation; #339 drift still trips; family-audit advisory. End-to-end agent NOVEL promote now passes
  without lowering `n_funnel`.
- FAST check: `... pytest -q tests/test_cli_research_promote_pit.py tests/test_family_registry.py tests/test_cli_research.py tests/test_concurrency.py -k "novel or seed or family or concurrent or fingerprint or mint_cap or budget or grant or slug or actor or founder or burn or trigger"`.

## Task 4 — Integration + full gate
- Merge main, resolve, run the FULL gate:
  `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
- Open PR (stays OPEN — CODEOWNERS: store.py/promotion.py/clustering.py surface + db.py schema+triggers).
  Email the user (CODEOWNERS PR). Do NOT self-merge.

## GATE-1 status
- Codex read-only GATE-1 history: R0/R1/R2/R3/R4/R5/R6/R7 all **BLOCKED**; **R8 got an APPROVE but was
  then re-audited by THREE independent Codex passes (condensed summary; summary + governance-residual
  arguments; FULL verbatim body §1-9) — all three returned BLOCK** with new STRUCTURAL findings (not the
  previously-closed issues): they attacked what R8 left documentation-only / partially-addressed.
- **R9 closes the R8 re-audit BLOCK with mechanism, not prose** (8 findings, design §0):
  1. Append-only invariant → **DB triggers** on the five classifier-read tables + a source-scan guard test
     binding classifier reads to fingerprint components (Task 1).
  2. Founder budget rate-bounded-not-lifetime-finite + deferred → **human-replenished lifetime mint budget
     BUILT NOW** (`agent_mint_grants` ledger + CODEOWNERS constant, enforced under the lock; a
     non-statistical quota, does not offend #324) (Tasks 1+3).
  3. `profile_digest` pre-lock-only trust boundary → **member profiles DB-persisted at assignment**, read
     from DB, so the profile axis is inside the under-lock `graph_fingerprint` CAS; `profile_digest`
     REMOVED (Tasks 1+3).
  4. Holdout-burn-via-drift-race → **re-check at the `on_peek` burn instant releases the reservation for
     drift caught at/before the burn (no holdout burned); the post-peek/pre-lock run_gate re-check remains a
     NARROWED residual — release-on-failure + WARNING audit, documented + monitored, not fully closed** (Task 3).
  5. Cap/window constants' home unstated → **CODEOWNERS-protected `store.py` module constants, no CLI flag /
     env var** (Task 3 + a scanner test).
  6. Founder audit blind spot → **`families.founder_gate_id`** queryable both directions (Tasks 1+3).
  7. Overflow-guard ordering could permanently DoS → **WHERE-filtered seed SUM**; a corrupt row is excluded,
     never a permanent block (Tasks 1+3).
  8. Non-canonical-UTC `created_at` + additive-seed dedupe → **UTC fail-closed cap read** + explicit
     multi-parent/diamond dedupe tests (Tasks 2+3).
- **Re-run GATE-1 (Codex, read-only) on R9 before implementation.** Only after a clean APPROVE proceed to
  Tasks 1-4.
