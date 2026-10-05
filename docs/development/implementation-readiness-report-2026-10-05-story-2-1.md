---
stepsCompleted:
  - step-01-document-discovery
  - step-02-prd-analysis
  - step-03-epic-coverage-validation
  - step-04-ux-alignment
  - step-05-epic-quality-review
  - step-06-final-assessment
filesIncluded:
  - docs/development/stories/2-1-refuse-live-qualification-on-relaxed-gates.md
  - docs/development/specs/spec-story-2-1-unrelaxed-live-qualification/ (SPEC.md, live-qualification-contract.md, .decision-log.md)
  - docs/agent/story-delivery.md §2, AGENTS.md, CLAUDE.md
  - owner decisions #624 (comment of 2026-10-04, item 2) and #682 (comment of 2026-10-04)
  - sibling contract docs/development/specs/spec-story-2-2-paper-exit-drain/ (coordination only)
assessmentScope: Story 2.1
baseline: 047d6d0 (contracts commit; the code is identical to e875b6d, where the contract's line references were checked)
---

# Implementation Readiness Assessment Report

**Date:** 2026-10-05 · **Project:** Algua, Story 2.1 · **Assessor:** independent reviewer (Claude),
BMAD method, non-interactive defaults.

Code claims were checked at `047d6d0`. All experiments ran in scratch, and the repository was not
modified:

- The v49 prototype ran on a `cp` copy of `data/algua.db` and on fresh databases.
- A trigger battery ran through the repo's `connect()`.
- I built a scratch tree implementing contract §2–§6: vocabulary, DDL and classifier, recording
  threaded through both writer chains, the predicate at both call sites, and the raw-edge refusal.
- The full suite ran on that tree, on a refusal-only tree and on an unmodified baseline tree, as 12
  parallel chunks each. Git-dependent files were rerun in git-initialised copies.
- A real `ssh-keygen` go-live ceremony was driven through the CLI.
- `lint-imports` ran on the prototype.

**Discovery:** the inputs are as listed, with no duplicates and no UX surface. The only output change
is that the go-live challenge JSON gains four keys. The owner texts match the story: #624 item 2 says
signed relaxations stay available for exploration, a relaxed gate or certificate can never authorize
go-live, and agents get no waiver authority. #682 says the raw edges are removed for all actors. The
2026-10-05 sign-off covers three items: `agent_walls_waived` on human research rows, refusal of an
agent's NaN forward threshold at preflight, and `registry_cmd.py` joining CODEOWNERS.

## Requirements Traceability

| AC | Contract | Code seam (verified) | Status |
|---|---|---|---|
| 1 Closed vocabulary | §1.1–1.3 | new `registry/relaxations.py`; `GateCriteria` `research/gates.py:98-106`, every `GATE_SPECS` op `>=`/`>` (`:252-262`); forward guard `forward_promotion.py:47-70` | Covered. Direction rule verified, including NaN and `+inf` |
| 2 Every flag classified | §1.4, §10 | `research_cmd.py:32-95`; `promote_run.py:103-114`; `paper_cmd.py:1157-1198`; run-all keys `research_batch_cmd.py:80-85` | **Verified set-equal** (25 research keys, 11 paper keys). See m5 |
| 3 Recorded, immutable | §2, §4 | INSERTs `store/gate.py:85`, `:386`, `store/forward_gate.py:108`; `run_gate` `promotion.py:266`, `gate_row` `:416-450`; `promote_task` `:195-216`, `:291`; forward `gate_row` `forward_promotion.py:221-244` | Covered; the prototype records correctly. See m2 |
| 4 One-time classification | §3 | migrate tail `db/migrate.py:284-291` | **Verified on the production copy** (21 rows to `[]`). See **M1**, m3 |
| 5 One unbypassable predicate | §5 | `transitions.py:142-143`; `registry_cmd.py:216-217` | Cannot be skipped (verified). See m1 |
| 6 Refused before signing | §5 | issue `registry_cmd.py:218` (`live_gate.py:44-50`); verify `live_gate.py:132-159`; consume inside `apply_transition` | **Verified end to end** (probes B and C) |
| 7 No human override | §5, §10 | the same | Verified (probe C refuses a valid signature; probe A shows both sets) |
| 8 Exploration unchanged | §8 | `canonical_run_context` drops None (`human_actor.py:60-69`), so the §8 golden strings are consistent; intake eligibility `store/deployment.py:181-186` | Covered |
| 9 Protected and green | §9, §10 | CODEOWNERS; `tests/test_repo_hygiene.py:237`; ratchet `tests/test_module_size_ratchet.py:35,58-103` | Covered. See **M1**; `lint-imports` gives 29 kept, 0 broken |
| 10 No raw way in | §6, §7 | `transitions.py:36-113`; CLI caller `registry_cmd.py:249` | **Verified**: no production path breaks. See m6, m7 |

No clause contradicts AGENTS.md, CLAUDE.md, the 1.3d contract or the live wall. Frozen go-live still
ends at `refuse_frozen_deployment` (`transitions.py:138`), which runs before the predicate. Agents
gain nothing. Their only behavioural change is stricter: a NaN forward threshold is now refused at
preflight, as signed off.

## Code Feasibility Verification

| Claim | Verdict | Evidence |
|---|---|---|
| The §1.4 table equals the real parameter set | **Verified** | `typer.main.get_command(...)`: research click params ∪ `inspect.signature(promote_task)` equals the 25 table keys exactly. The 11 paper keys match too. `_ALLOWED_KEYS["promote"]` ⊆ `promote_task`. The merge-back seam passes only `universe/start/end/demo/snapshot/fundamentals_snapshot/news_snapshot/delistings/actor='agent'/attempt_token` (`paper_cmd.py:552-556`), all classified |
| Humans skip agent walls with no flag | Holds | `promotion.py:126` (reproducible source), `:152` (cost floor), `:168` (lookback), `:206` (gated universe). A human PARENTAGE verdict also auto-mints a child family **without** `--new-family` (`family_assignment.py:198-205`), which further supports `agent_walls_waived` |
| Forward guard refactor is byte-identical for finite inputs | Holds | `not (v >= d)` ≡ `v < d` for finite values, and removing the common `threshold:` prefix keeps the sort order. NaN confidence is still refused for every actor first (`forward_promotion.py:54-58`) |
| Every gate-row writer | **Verified** | Production SQL writers: INSERT `store/gate.py:85`, `:386`, `store/forward_gate.py:108`; UPDATE `consumed` (`store/base.py:89`, `:106`); UPDATE FDR relabel (`db/gate.py:135`, `:171`). No REPLACE or INSERT…SELECT. The only non-test caller of `record_gate_evaluation` is `scripts/seed_runs_dev.py:426`. `web/` reaches the registry only through the CLI |
| v49 DDL on the production copy (`cp`, WAL checkpointed, user_version 48) | **Verified** | The repo's `migrate()` runs, then the v49 block: 21 rows to `[]`, 0 unrecorded. A second run is skipped by the guard. All other columns are byte-identical. `foreign_key_check` is empty and `integrity_check` is ok. v48 `migrate()` over v49 objects runs and re-stamps user_version 48 (N6) |
| Fresh schema object count | **Verified** | 137 → 141 |
| Recorded trigger | **Verified** | Refused with the §2 messages: NULL, `''`, `null`, `[ ]`, `[] `, `{}`, `"x"`, `["zzz"]`, `[1]`, `[null]`, nested arrays, unsorted values, duplicates, and the other table's tokens. Accepted: `[]` and canonical sets. A v48 `record_gate_evaluation` is refused, so forward-only holds |
| Immutability | **Partly** | An UPDATE of the column, even with an unchanged value, is refused. `consumed` and FDR updates still work. **`INSERT OR REPLACE` rewrote a relaxed row to `[]` through the repo's `connect()`**, including the research row anchoring a deployment, despite `foreign_keys=ON` (m2) |
| Agent tightening records `[]`; prototype threading | Verified | Prototype CLI: agent `research promote --demo --min-holdout-sharpe 0.9` recorded `["demo_data"]`. Agent `--n-combos` is refused exactly as before |
| Predicate runs before challenge/ssh-keygen/consume (real `ssh-keygen`, CLI) | **Verified** | A clean world issues a challenge showing `deployment_id`, `research_gate_id` and both `[]`; it is signed, completed, consumed and goes live. B: a relaxed research row refuses issuance with `live_qualification_relaxed` (retryable false), 0 `live_challenges` rows and 0 `ssh-keygen` calls. C: a newer relaxed certificate after a clean issuance refuses a valid signature before `ssh-keygen`, and the challenge stays unconsumed. E: the legacy cohort is refused as `unrecorded` |
| An injected verifier cannot bypass the predicate | **Cannot skip, can steer** | D: a verifier (here the CLI seam) returning an older clean certificate id while the deployment's newest certificate is relaxed passed the predicate and **went live** (m1) |
| Raw-edge refusal breaks no production path | **Verified** | Refusal alone: 131 new failures in 14 test files, all scaffolding or expected-refusal tests already in the §11 files (most go through a few shared helpers, e.g. `test_cli_paper.py:78`). `test_cli_merge_back.py`, `test_drain_mergeback_queue.py`, paper intake and dust tests pass unchanged, in git-initialised copies. Production `transition_strategy` callers are only `registry_cmd.py:249` and `evaluation/backtest_run.py:105` (`→ backtested`) |
| Full prototype churn is fixture churn only | Verified | 554 further failures, all in expected categories: `record_gate_evaluation` missing `relaxations_json` (310, through shared helpers), `run_gate` missing `relaxations` (33), forward writers (37), raw INSERTs refused, schema-version pins, and go-live stubs. No production-path failure |
| Import contracts | **Verified** | `lint-imports` on the prototype: 29 kept, 0 broken. `db/relaxations.py` imports research only lazily (grimp still sees it; no contract forbids it) |
| Size pins | Mostly hold | `registry_cmd.py` line-neutral at 444/446. `transitions.py` 288 with refusal and predicate before the §6 deletions (about 235 after). `cli/errors.py` +2. **`db/migrate.py` measured 303 (M1)** |

## Findings

### BLOCKER

None. The relaxation vocabulary covers every option of `research promote`, `promote_task` (also
`research run-all` and the merge-back seam) and `paper promote`, and none is misclassified. The
entry-point inventory has no missing writer or go-live path. m6 is a hardening, not a missing path.

### MAJOR

**M1 — As drafted, the §2 `migrate()` block takes `db/migrate.py` past the ratchet floor.**
`migrate.py` is 291 lines and unpinned (FLOOR 300, `tests/test_module_size_ratchet.py:35`). §2
places the five-line loop inline, imports four names, and also requires that "the module docstring
of `migrate.py` names" the orderings. §9 calls the change "small", and states that no new module
reaches 300 lines and no pin is raised.

Measured: the inline block with ruff-sorted named imports (the file's own style, `:29`) gives 303
lines before any version comment or docstring line, and `test_no_new_god_file_appeared` fails. This
is the 1.3d m6 failure mode, so the contract as written leads to a red gate.

**Correction (replace the §2 `migrate()` paragraph and its code block):**
> `db/relaxations.py` exports `apply_relaxation_schema(conn: sqlite3.Connection) -> None`, which
> runs, in this order: `_add_missing_columns(conn, table, RELAXATIONS_COLUMN)` for each table in
> `RELAXATION_TABLES`, then `classify_unrecorded_gate_rows(conn)`, then each statement in
> `RELAXATION_STATEMENTS`. `migrate()` imports only that name and calls it once, after the v48
> tick-link loop and before the `user_version` stamp, under a one-line `# v49 (Story 2.1): ...`
> comment. The hard orderings (ALTER before classification and triggers; classification before the
> immutability triggers) are stated in `db/relaxations.py`'s module docstring; `migrate.py`'s
> docstring gains one sentence pointing there.

In §9, change the `migrate.py` row to: "one import, one call, one comment line, one docstring
sentence | 291 → at most 296 | stays unpinned (below 300)".

### MINOR

- **m1 — The predicate judges whichever certificate id the verifier returns.** §5 step 4 accepts any
  row of the strategy's active deployment. Probe D: an injected verifier returning an older clean id
  went live past a newer relaxed certificate. The default verifier never does this, because it
  selects the newest row (`live_certificate.py:83-89`), so production is unaffected. **Correction
  to §5 step 4:**
  > "Else read the newest forward row of the active deployment, `SELECT id, relaxations_json FROM
  > forward_gate_evaluations WHERE strategy_id=? AND deployment_id=? ORDER BY id DESC LIMIT 1`. If
  > there is no row or its `id != cid`, the result is `("unrecorded",)`. Otherwise decode it with
  > `FORWARD_VOCABULARY`; NULL or undecodable reads as `("unrecorded",)`."

  Add to §10 Predicate: "certificate id not the deployment's newest forward row". Add to the
  mutation list: "drop the newest-row condition". The default-path verdict is unchanged.
- **m2 — Immutability and canonical form have two holes.** First, `INSERT OR REPLACE` with an
  existing `id` passes the recorded trigger, and the implicit delete fires nothing, so a relaxed set
  can be rewritten. This was verified on both tables, including a deployment-anchoring research row
  under `foreign_keys=ON`. Second, `["demo_data"]` is accepted by the trigger, because SQLite's
  `json()` keeps the escape, but `decode_relaxations` rejects it; that direction fails closed.
  **Correction:** make this the first statement of both `_relaxations_recorded` triggers:
  ```sql
  SELECT RAISE(ABORT, '<table> rows are append-only')
  WHERE EXISTS (SELECT 1 FROM <table> WHERE id = NEW.id);
  ```
  Also append `AND instr(NEW.relaxations_json, char(92)) = 0` to the canonical-array test, since no
  token contains a backslash. Verified in scratch: both REPLACE forms are refused, and plain and
  explicit-new-id INSERTs are still accepted. The object count stays 141. Add both to the §10
  Schema tests and the mutation list. §5's "rows that cannot change" then holds under the repo's
  own connection; this is the 1.3d m1 precedent.
- **m3 — Roll-forward sequencing is missing.** The drainer runs every 30 min
  (`algua-mergeback-drain.timer:8`). If a v48 promote is still running when a v49 process migrates,
  `on_peek` has already burned the single-use holdout (`promote_run.py:255-273`), and its INSERT is
  then refused. That strategy can be re-gated only with a human `--allow-holdout-reuse`, which v49
  records as a relaxation, so the strategy can never be live-eligible. **Add to §2's forward-only
  paragraph and to the `deploy/systemd/README.md` note:**
  > "Roll-forward: stop `algua-mergeback-drain.timer` and `algua-research.timer` and wait for
  > running units to exit before deploying v49; run one registry command (e.g. `algua registry
  > list`) to migrate; then restart the timers."
- **m4 — The §11 churn list is incomplete.**
  - Add the schema-version literal pins: `test_registry_db.py:894,918,983,1036,1041`,
    `test_frozen_evidence_schema.py:616`, `registry/test_runs_schema.py:38,39,66`,
    `registry/test_holdout_returns.py:17`, `registry/test_novel_family_seed_524.py:60` and
    `test_family_registry.py:58,69`. All were found by the prototype run.
  - Add the research `gate_row` builder `_make_gate_row` (`test_registry_store.py:1212`), which
    feeds 18 direct `record_gate_with_fdr_and_maybe_promote` calls.
  - Correct "25 sites" to 26 test call sites plus `scripts/seed_runs_dev.py:426`.
  - Correct the run-all keys reference `research_batch_cmd.py:76-81` to `:80-85` (§7 W1b).
- **m5 — Clarify the AC2 mapping in §1.4.** Add after the table: "The `(no option)`
  `min_holdout_observations` row documents a token only; it is not a key of
  `RESEARCH_PROMOTE_INPUTS`, whose keys equal the click parameters ∪ `promote_task` parameters (25,
  verified at baseline)." Otherwise §10's set-equality test cannot pass with that row as a key.
- **m6 — The structural test misses SQL stage writers.** §10 pins the callers of
  `.apply_transition(`/`._apply_transition_locked(`, but stages are also written by raw SQL:
  `mergeback_intake.py:238` (`idea → backtested`), `store/base.py:125` (the CAS) and `db/core.py:75`
  (the `shortlisted` rename). **Append to the §10 AC10 structural test:**
  > "and that the only statements in `algua/` writing `strategies.stage` are those three"

  so a future direct write into `candidate`/`forward_tested` fails the test.
- **m7 — Coordinate with Story 2.2.** 2.2 also rewrites `transition_strategy` (`exit_guard` becomes
  `exit_guard_selector` behind the operator lock), deletes `_live_exit_guard` from `registry_cmd.py`
  (lowering the 446 pin) and edits CODEOWNERS and `INTEGRITY_CRITICAL_MODULES` (2.2 contract
  §2.2–2.3, §6). **Add to §9:**
  > "Whichever of 2.1/2.2 merges second rebases. §6's refusal stays directly after the intake
  > refusal. `verify_live_qualification` stays in `_validate_live_gate`, i.e. before 2.2's lock and
  > drain, so a refused go-live cancels nothing. The issuance edit stays line-neutral against the
  > then-current pin."

  The combined `transitions.py` comes to about 248 lines (277 − 42 + 13).

### NOTE

- **N1 — No production go-live is reachable until Story 2.3.** Intake admits only frozen
  descriptors (`store/deployment.py:146-150`), and frozen go-live is refused first. In production
  the predicate therefore only refuses the legacy cohort: one tenant remains,
  `liquidity_stable_quality_momentum` at `paper`, kill-switched. Otherwise it is exercised on test
  working-tree deployments, and probe A shows that real path succeeds when clean. Story 2.3 AC1–2
  already requires the predicate and binds both row ids into the signed payload.
- **N2 — Trade-time re-verification does not re-check certificates or relaxations.**
  `verify_live_authorization` (`live_gate.py:166-214`) re-verifies only the signature over the
  recomputed identity. That is adequate: both sets and `research_gate_id` are trigger-immutable,
  `paper promote` refuses stage `live` (`forward_promotion.py:175`), and any return to live repeats
  the wall. 2.3 AC5 will pin the bound rows transitively. Production has no live strategy.
- **N3 — The agent-row classification rule assumes today's agent walls applied.** That holds for
  production: the oldest of the 21 rows is from 2026-09-12, after #205, #325, #345 and #559/v39.
  There are no forward rows, so the #432 assumption is moot.
- **N4 — `--windows` and `--holdout-frac` are agent-settable non-relaxations.** They change what the
  advisory window checks and the binding floor measure, and have no monotone strict direction. The
  decision log records this; the owner may revisit it like `--demo`. The authoritative merge-back
  path never sets them.
- **N5 — `promote_run.py` may fall below 300 after the carve** (about 299 by count). In that case
  delete its pin, as §9 already allows.
- **N6 — v48 `migrate()` on a v49 DB re-stamps user_version 48** (measured). This is consistent with
  the rollback note: rows written by v48 code stay NULL and read as `unrecorded` after roll-forward,
  because the classification guard sees the immutability trigger.

## Risk to Existing Behaviour

- **Research and paper exploration:** unchanged, apart from the signed-off NaN delta. Signed bytes
  are untouched (`canonical_run_context` and `build_actor_challenge` are not edited).
- **Autonomous funnel:** merge-back, drainer and intake tests pass under the raw-edge refusal. Rows
  the drainer writes before deploy classify as `[]`, or `["demo_data"]` under `MERGEBACK_DEMO`. The
  deploy window itself needs m3.
- **Production migration v48 → v49:** an additive ALTER (the CHECK passes on NULL), a one-time
  21-row UPDATE and four triggers. It is forward-only, and the rollback runbook is in §2.
- **Live lane:** untouched. No live strategies exist.

## Scope: one cycle or split

One cycle is feasible: the prototype implemented §2–§6 in about 150 production lines, and the churn
is mechanical and concentrated in shared helpers. Use three slices with strict file ownership:

- **(A) Recording.** Vocabulary, `db/relaxations.py` (with M1), migrate/constants, store writers and
  Protocols, `promotion`/`promote_run` and the `gate_fail_capture` carve, `forward_promotion`, plus
  the writer and version fixture churn.
- **(B) Predicate.** `live_qualification.py`, `_validate_live_gate`, the issuance line, `errors.py`
  and the envelope doc, plus the go-live test churn.
- **(C) Raw-edge refusal.** §6, the finder deletions, the raw-edge test churn and the docs.

B and C both edit `transitions.py`, so give it one owner or sequence them (and see m7). Splitting
AC10 into its own story is possible but not needed.

## Summary and Verdict

**Overall readiness: READY WITH CONDITIONS.** There are no blockers. The vocabulary covers every real
option, and the inventory is complete. The schema, the classification (21 production rows to `[]`)
and the predicate placement were verified by execution, including a real `ssh-keygen` ceremony. The
raw-edge removal breaks no production path.

Apply the M1 text correction before development. Fold in m1–m7. m1–m3 are small DDL, SQL and
runbook additions that close real holes and should not be deferred. No further full review is
required after these are applied. Readiness never authorizes deployment, trading or capital use.
