---
stepsCompleted:
  - step-01-document-discovery
  - step-02-prd-analysis
  - step-03-epic-coverage-validation
  - step-04-ux-alignment
  - step-05-epic-quality-review
  - step-06-final-assessment
filesIncluded:
  - scratchpad/1-3d-spec/ (draft SPEC, frozen-evidence-contract, decision log)
  - docs/development/stories/1-3d-bind-operational-evidence-and-qualification.md
  - docs/development/specs/spec-story-1-3c-frozen-paper-execution/ and story 1-3c (review findings)
  - docs/development/stories/1-3-*.md, sprint-change-proposal-2026-09-25.md, epics.md
  - docs/superpowers/specs/2026-09-22-artifact-freeze-design.md, AGENTS.md
assessmentScope: Story 1.3d
baseline: ae7e3ff (main after PR #680; 1.3d-relevant code identical to df0662e, where probes ran)
---

# Implementation Readiness Assessment Report

**Date:** 2026-09-30 · **Project:** Algua — Story 1.3d · **Assessor:** independent reviewer (Claude),
BMAD method, non-interactive defaults. Code claims checked at `ae7e3ff`. The experiments ran in a
scratch copy: the §2 DDL was applied to a DB built by the repo's `migrate()`, a trade-tick/run-all
transaction probe was run, 17 tick, promotion and live test files were run on v48, and a replay
byte-determinism probe was run.

**Discovery:** the inputs are as listed, with no duplicates. UX does not apply (no new command; one
JSON key is added). The story traces FR5–FR6, FR9–FR10, FR12 and NFR1–NFR8. For FR12, the epics
coverage map assigns Epic 1 only the deployment/tick linkage (epics.md:154), so deferring
order-intent linkage is in scope as drafted.

## Requirements Traceability

| AC | Contract | Status | AC | Contract | Status |
|---|---|---|---|---|---|
| 1 append-preserving attempts | §1, §2 | covered (m1, m2, m4) | 6 promotion trusts deployment | §5 | covered, **M1** |
| 2 no authority copies | §2 | covered | 7 restart and replay | §6 | covered (m5, N2) |
| 3 atomic linkage | §2 triggers, §3 | covered, verified | 8 classification | §3, §7 | covered (m4) |
| 4 exact epoch inclusion | §4 | covered, **M2** (m3) | 9 matrix | §7 | covered |
| 5 content verifiable | §5 fresh verifier | covered | 10 retention/authority | SPEC, §2, §5 | covered (m8) |

No clause contradicts the parent story, the 1.3c contract, AGENTS.md §3 or the live wall. Go-live
still ends in `frozen_live_unsupported` (transitions.py:138, 198-218). Agents gain only
`paper → forward_tested` through `paper promote`, which CLAUDE.md already allows. The contract
matches 1.3c's decisions to use no lease, no GC and no `snapshot_id` in the v1 wire.

## Code Feasibility Verification

| Claim | Verdict | Evidence |
|---|---|---|
| v48 DDL on a populated v47 DB; `ALTER … ADD COLUMN … REFERENCES` under `foreign_keys=ON` | Verified | Applied twice (idempotent), `foreign_key_check` clean. Allowed because the default is NULL; precedent migrate.py:258-260 |
| CHECKs, `UNIQUE(request_id,phase)`, FKs, append-only aborts | Verified | Both/neither result and failure, diagnostic on success, 31-char id, `a` with parent, 256 KiB+1 multibyte: all abort. UPDATE and DELETE abort |
| Phase-b trigger | Verified | Aborts after a failed A, an `early_no_decision` A, or a mismatched binding, snapshot, request or deployment. A failure B after a good A is allowed |
| Tick trigger and partial unique index | Verified | Frozen tick with no, failed, phase-a, wrong-deployment, wrong-snapshot or dangling link: aborts. Working-tree tick with a link, or a NULL-deployment (legacy) tick with one: aborts. Duplicate link: unique violation |
| `IS` comparisons | Hold | Behave as NULL-safe equality, so NULL=NULL passes (m2) |
| Existing non-frozen writers unaffected | Verified | `record_tick_snapshot` legacy INSERT and INSERT…SELECT (tick_snapshots.py:38-71), raw fixture INSERTs, `UPDATE … recorded_at`. On v48, 563 of 574 tests in 17 tick, promotion and live files pass; all 11 failures are expected (m7) |
| REPLACE bypass | Partly | Blocked through `connect()` (recursive_triggers, connection.py:19-27). A raw connection replaced a failed Phase A row (m1) |
| Attempts judged after `_invoke` returns | Holds | Canonical compare, `_checked_decision` and breach-by-rerun: frozen_dispatch.py:126-131, 148-157, 239-262. Supervisor-settled paths never reach `_invoke`: :124-125, 146-147, 161-193 |
| No open transaction at recording time | Verified | Probe: `conn.in_transaction` is False at entry, at both launches and at the tick write, on both the trade-tick and run-all paths |
| Port built per tick with snapshot and window in scope | Holds | paper_cmd.py:609-620, 623-626, 683. Both callers pass the snapshot (:846, :1100) |
| A tick is written only after a Phase B child | Holds | `peak_equity` guard (paper_cmd.py:743); an early no-decision has peak None (planner_early.py:42-55) |
| Import contracts | Hold | `contracts` stays pure; registry must not import live (pyproject); the port receives a callable |
| Admissibility insertion point and protocol check | Hold | forward_evidence.py:57-58, 109-133, 219-233; the wire stamp is parsed at artifact_manifest.py:79-92 |
| Promotion chokepoint "called first" | **Fails** | M1 |
| "`no_decision` excludes late no-decisions" | **Fails** | M2 |
| Bars re-read and Arrow bytes deterministic | Verified | Store-backed read plus `encode_bars` in 4 processes (PYTHONHASHSEED 0/1/12345/random) gave an identical SHA-256 with pyarrow 23.0.1. Rows are sorted at data/schema.py:103 |
| `request_json` is enough to relaunch | Holds | It carries identities, recorded config, `now`, positions, gate universe, captured values and the bars reference (frozen_wire.py:199-250) |
| Size pins | Tight | frozen_dispatch 296 (floor 300); paper_cmd 1358/1409; forward_evidence 401/429; migrate 282 (floor 300); promote_run 348/348 |

## Findings

### BLOCKER

None.

### MAJOR

**M1 — "`paper promote` calls it first" changes working-tree and legacy promotion.** Today's order
is: the ledger-only frozen refusal (paper_cmd.py:1202), then `authenticate_actor`, which hashes the
checkout only for a human (human_actor.py:222-224), then preflight (:1222), then `run_forward_gate`,
which computes identity and calls `require_tick_deployment` (forward_promotion.py:130-139).
`require_tick_deployment` raises for a strategy outside the legacy cohort that has no deployment, and
returns None for a legacy one (store/deployment.py:98-108). Running the working-tree branch of the
chokepoint first therefore reorders refusals.

This was measured in scratch with a stub chokepoint placed first. test_cli_paper.py:1593 (agent
relaxation refused) now fails with "forward promotion requires one active deployment epoch", and
:1619 (wrong stage) fails with "not in the fixed legacy cohort". The contract also runs the
chokepoint twice (in `paper promote` and again in `run_forward_gate`), which means two ~1 GB
environment verifications.

**Fix:** keep the ledger-only frozen check in today's slot. Only for a frozen deployment, call
`promotion_identity` there, so fresh verification happens before authentication. Working-tree and
legacy strategies keep today's order and identity sites. Resolve once, and pass
`(deployment, identity)` to `run_forward_gate`, which keeps a cheap frozen re-check. Give
`authenticate_actor` a lazy `identity_for: Callable[[], ArtifactIdentity]`, called only for a human,
so agents are not hashed early. Research promote passes `lambda: compute_artifact_hashes(name)`;
this edit must be line-neutral because promote_run.py is at its 348-line pin.

**M2 — §4's "(`no_decision` still excludes late no-decisions)" is false.** A late warming
no-decision carries Phase A's decision timestamp: `_late_state(captured, phase_a.decision_ts)`
(planner_late.py:180), then `LateNoDecision("warming", state)` (:216-217), then `_tick_result`
(live_loop.py:230-232, 331-332), then a tick written with that `decision_ts`
(paper_cmd.py:743-747). `no_decision` only tests for NULL (forward_evidence.py:125-126). Such a tick
always holds positions, because a flat warm-up already ends in Phase A (planner_early.py:180-181).
Working-tree ticks of this kind count today.

The implementer must then either add a frozen-only exclusion, breaking "filters apply unchanged" and
parity, or write a §7 test that cannot pass. **Fix:** delete the parenthetical. State that a linked
`late_no_decision` tick counts exactly like the equivalent working-tree tick, which is why the link
admits both result kinds.

### MINOR

- **m1** Append-only fails under a raw connection. `INSERT OR REPLACE` via `sqlite3.connect`
  without the pragma replaced a failed Phase A row, so a failed attempt could be rewritten into a
  success. This matches the #524 residual, and there are no raw connections in production. Close it
  with one trigger, verified to block both REPLACE forms with recursive_triggers OFF:
  `BEFORE INSERT … WHEN EXISTS (SELECT 1 FROM frozen_invocations WHERE id=NEW.id OR
  (request_id=NEW.request_id AND phase=NEW.phase)) … RAISE(ABORT, …)`.
- **m2** Tighten the DDL: `snapshot_id NOT NULL` (every caller has one; replay needs it); a
  non-NULL `phase_a_binding` on `snapshot_required` rows; and CHECK vocabularies for `result_kind`
  (5 kinds) and `failure_code` (the ten §8 codes), both of which currently accept typos.
- **m3** A linked tick's `snapshot_id` is still updatable (verified), and §4 does not re-check it.
  Add `snapshot_id` and `strategy_id` to the trigger's `UPDATE OF` list, or have the §4 join require
  `i.snapshot_id IS t.snapshot_id`.
- **m4** Attempt edges:
  - Systemic exceptions (`OSError` from launch, SQLite errors, `KeyboardInterrupt`) leave no row,
    like a crash; CAP-1 should say so.
  - The no-open-transaction check must raise explicitly. A bare `assert` is stripped by `-O`; see
    the repo rule at live_loop.py:266-272.
  - Phase A is recorded as a success before `closed_bars` can still raise `frozen_result_invalid`
    (frozen_dispatch.py:133-141). That failure then appears only in setup_error and the audit log;
    say so.
  - Define `result_sha256` over the accepted stdout bytes (canonical form plus `\n`,
    frozen_wire_result.py:177-189). The re-encoded result is not the same bytes, because `_risk`
    sanitizes `detail` (frozen_dispatch.py:283-286).
- **m5** Replay inputs are unpinned.
  - Symbols: specify `sorted(set(gate_universe) | {s: q≠0 in early_positions})` from `request_json`
    (live_loop.py:276-279).
  - Window: store `bars_start`/`bars_end` as ISO-8601 UTC of the exact values passed to `get_bars`.
  - Acceptance: use the logical `bars_digest` in `request_json` (which the child enforces). Treat
    `bars_sha256` equality as a same-supervisor-environment assertion, pinned by an `encode_bars`
    golden-bytes test. Arrow bytes follow the supervisor's pyarrow, and AC7 says dependency changes
    must not alter evidence identity.
- **m6** Placement:
  - The recording hooks alone take frozen_dispatch.py past 300 lines. Move existing code (for
    example `_checked_decision`, or the failure plumbing) into `frozen_attempt.py`; do not add a pin.
    `cli/errors.py:118` imports `FrozenTenantFailure`, so update that import if it moves.
  - Keep the tick triggers as constants in `db/frozen_evidence.py`, called after the v47 block of
    `migrate()`. Inlining them, as the family precedent does, would take migrate.py past the floor.
- **m7** Name the churn:
  - docs/architecture.md:28-30 still names `frozen_qualification_pending`. The spec only lists the
    error registry and the envelope doc.
  - Tests that stamp unlinked frozen ticks: test_deployments.py:447-458,
    test_frozen_runtime.py:265-275, and eight 1.3c e2e tests in test_frozen_paper_cli.py.
  - Schema fingerprint: test_registry_db.py:112 (128 → 136 objects, measured).
  - The filter tuple at test_forward_promotion.py:45, and test_frozen_paper_cli.py:309 (old code).
- **m8** v48 is forward-only. Measured: 1.3c code on a v48 DB aborts `run-all` with
  `{"code":"internal"}` at the first frozen tenant, after that tenant's orders are sent. Later
  tenants never tick and the session marker is never written, so the job retries every 20 minutes.
  State this in the contract and the runbook: before any rollback, retire frozen deployments or drop
  `tick_snapshots_frozen_link`.

### NOTE

- **N1** Storage is about 2 rows of 10–20 KiB per frozen tenant per session (the timer is
  session-gated, algua-paper.timer). That is roughly 10 MB per tenant-year, so indefinite retention
  is affordable.
- **N2** Replay is exact only for strategies whose output does not depend on string-hash order. The
  child runs with `-I`, which ignores `PYTHONHASHSEED`, and wire v1 fixes the environment. The
  fixture strategies satisfy this; record it as a residual.
- **N3** Removing `frozen_qualification_pending` is otherwise contained: artifact_errors.py:43,
  cli/errors.py:47,66 (`TransitionError` → `wrong_stage`, :69), forward_promotion.py:71-80,129,
  transitions.py:198-218, paper_cmd.py:95,1202, test_frozen_promotion_refusal.py, and
  cli-error-envelope.md:87.

## Risk to Existing Behaviour

- **Legacy tenants** (the 18 in production, `deployment_id` NULL): the tick trigger is a no-op for
  them (verified). `_frozen_planner` returns None for them, so nothing is recorded. Their only change
  is one optional keyword argument.
- **Working-tree promotion:** unchanged once M1 is applied. As drafted, the refusal order changes.
- **Live lane:** untouched. `test_cli_live` and `test_live_loop` pass on v48.
- **Production migration v47→v48:** `ADD COLUMN` does not rewrite the table. The partial index is
  built over all-NULL rows. Triggers act only on future writes. 1.3c-era frozen ticks stay unlinked
  and never count, so their clock restarts at deploy (by design). Rollback is m8.

## Summary and Verdict

**Overall readiness: READY WITH CONDITIONS.** No blockers. The protected schema does what it claims
under the repo's connection helper: 49 of 51 probes behaved as intended, and the 2 exceptions are m1
and m3. The recording seam fits the real `FrozenPlanner` with no open transaction, and replay inputs
are byte-deterministic.

Two MAJOR text amendments are needed before development: M1 (promotion ordering, one resolution,
and a lazy identity for `authenticate_actor`) and M2 (correct the late no-decision statement).
Fold m1–m8 in as convenient. No further full review is required after these are applied. Readiness
never authorizes deployment, trading or capital use.

## Resolution (2026-09-30)

M1, M2 and m1–m8 were applied to the normative companion
(`specs/spec-story-1-3d-frozen-evidence-and-qualification/frozen-evidence-contract.md`) and recorded
in its decision log (§1 attempt edges; §2 DDL, triggers, forward-only note; §3 placement and the
transaction check; §4 admissibility; §5 promotion order; §6 replay; §7 churn). NOTEs N1–N3 are
recorded in §2, §6 and §7.

**Updated verdict: READY.**
