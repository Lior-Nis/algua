---
baseline_commit: ae7e3ff
---

# Story 1.3d: Bind operational evidence and qualification

Status: done

Prepared: 2026-09-25. Readiness: 2026-09-30. Baseline: Story 1.3c merge `ae7e3ff` (PR #680).
Epic: 1. Parent: Story 1.3. Requirements: FR5–FR6, FR9–FR10, FR12 and NFR1–NFR8.
Depends on: Stories 1.3a–1.3c reviewed and merged.

## Story

As Algua's operator,
I want frozen planner invocations and their resulting observations bound to the exact verified
deployment that produced them,
so that paper evidence can justify qualification without trusting the ambient checkout or incomplete
runtime attempts.

## Scope and authority

This story adds permanent deployment-bound invocation evidence, teaches forward evaluation and paper
promotion to require it for frozen deployments, and removes the temporary
`frozen_qualification_pending` block only after all checks pass.

It does not migrate existing working-tree/legacy tenants, garbage-collect content, change the live
ceremony, authorize capital, narrow code hashing or grant agents new authority. Live remains bound to
the existing authenticated human wall until its separately reviewed deployment-signing story.

## Normative contract

The [Story 1.3d machine contract](../specs/spec-story-1-3d-frozen-evidence-and-qualification/SPEC.md)
and its [field-level companion](../specs/spec-story-1-3d-frozen-evidence-and-qualification/frozen-evidence-contract.md)
are normative; the companion's §2 (exact DDL and triggers) is this story's protected schema review.
Implementers and reviewers must read both, and the Story 1.3c contract they build on.

## Acceptance criteria

1. **Append-preserving invocation evidence.** Every Phase A and Phase B attempt records deployment,
   artifact, environment, protocol, strategy and request identities; snapshot/captured-input
   identity; exact input-byte digest; Phase A binding; validated result digest or stable failure
   code; start/end timestamps; and bounded exit/signal/timeout/truncation metadata. Records cannot be
   updated into success or deleted.
2. **No authority-bearing copies.** Invocation evidence does not copy raw bars, credentials, raw
   stderr, registry/broker handles or account secrets. Immutable snapshots/artifacts remain the
   recovery source. Persisted diagnostics obey Story 1.3c's bound and sanitation rule.
3. **Atomic successful linkage.** A successful frozen tick is durably linked to the validated final
   invocation and its deployment. Failure between invocation recording and tick persistence cannot
   create an admissible orphan tick or mutate a failed attempt into success. Failed/partial attempts
   never write a successful tick.
4. **Exact epoch inclusion.** Forward evidence for a frozen deployment includes a tick only when its
   active deployment epoch, invocation linkage, request/result bindings and supported protocol are
   complete and consistent. It never mixes deployments, back-credits pre-activation rows or reuses
   evidence from a retired/abandoned epoch.
5. **Content remains verifiable.** Evaluation and promotion require the stored manifest, bundle and
   environment to verify against their recorded identities. Missing, corrupt, replaced,
   permission-drifted or unsupported content fails closed with a stable code and is never rebuilt
   from mutable `HEAD`.
6. **Promotion trusts the deployment.** `paper promote` verifies the active frozen deployment and
   its admissible observations from immutable records/content. It does not recompute identity from
   the ambient checkout. The temporary `frozen_qualification_pending` block is removed only for a
   deployment satisfying the normal forward gates plus these frozen-evidence checks.
7. **Restart and replay.** After supervisor restart and without a Git checkout, the same stored
   deployment resolves, verifies and deterministically replays a recorded request. Unrelated
   working-tree or dependency changes cannot alter its evidence identity or stop its clock.
8. **Failure classification remains correct.** Tenant artifact/protocol/invocation failures remain
   isolated before tenant effects; shared authority, SQLite, global halt, account reconciliation and
   book-risk failures remain systemic. `trade-tick`/`run-all` JSON and exit semantics remain stable.
9. **Complete operational matrix.** Tests cover incomplete A/B attempts, timeout/termination,
   malformed results, missing/corrupt/replaced content, permission drift, valid sibling continuation,
   systemic failure, replay determinism, evidence inclusion/exclusion, and promotion refusal/success.
10. **Retention and authority preservation.** Bundles, environments, manifests, attempts and linked
    evidence remain retained indefinitely in Phase 1. No GC, live signing change, migration or new
    permission is introduced. Protected review and the full repository gate pass.

## Tasks / subtasks

- [x] Design append-preserving invocation/link records and obtain protected schema review (AC1–AC3).
- [x] Record both phases and atomically bind successful final results to ticks (AC1–AC3).
- [x] Require complete frozen linkage in forward-evidence queries/evaluation (AC4).
- [x] Add immutable bundle/environment verification to evaluation and promotion (AC5–AC6).
- [x] Remove the temporary promotion block only behind the complete verifier (AC6).
- [x] Add restart/replay and mutable-checkout independence tests (AC7).
- [x] Exercise the full tenant/systemic failure and evidence matrix (AC8–AC9).
- [x] Independently review evidence authority and live-wall preservation (AC10).

### Review Findings

Independent review round 1 (2026-10-01): Codex was out of quota, so the story-delivery fallback ran
BMAD Blind Hunter + Edge Case Hunter over three bounded file groups (schema, store and evidence
filter; attempt recording and CLI wiring; promotion, human actor and transitions), and an
Acceptance Auditor over the whole story. The schema reviewer also migrated a copy of the production
registry (v47, 1,025 legacy ticks) to v48: clean, idempotent, all rows preserved, no measurable tick
write cost. No reviewer found a path to `forward_tested` without the forward gate on linked evidence
and freshly verified content. Every finding was patched, dismissed with a reason or deferred.

- [x] [Review][Patch] `run_forward_gate` compared the identity only with its own deployment row, so
  a direct caller could evaluate unverified content; `PromotionIdentity` is now an opaque value only
  the chokepoint can mint and the gate's only input [algua/registry/forward_promotion.py]
- [x] [Review][Patch] The later working-tree identity resolution in `paper promote` re-read the
  ledger and could verify a newly frozen epoch after authentication; it now refuses an epoch change
  [algua/registry/forward_promotion.py, algua/cli/paper_cmd.py]
- [x] [Review][Patch] A frozen human-actor challenge did not bind the deployment epoch; it now binds
  `deployment_id` and `manifest_digest` (working-tree and legacy challenge bytes unchanged)
  [algua/registry/forward_promotion.py, algua/cli/paper_cmd.py]
- [x] [Review][Patch] `result_kind` was not tied to `phase`; the v48 table CHECK and `FrozenAttempt`
  now allow only the kinds each phase can produce [algua/registry/db/frozen_evidence.py,
  algua/contracts/frozen_evidence.py]
- [x] [Review][Patch] The code's failure-code and result-kind vocabularies are tied to the DDL CHECK
  lists by a test, and §2's DDL is pinned to the code [tests/test_frozen_evidence_schema.py]
- [x] [Review][Patch] Test gaps: evidence snapshot equals the snapshot the provider served
  (`--snapshot` and `--refresh`); stored rows hold no credentials, account id or bar rows; a
  recorder SQLite error aborts `trade-tick` and `run-all` as `db_unavailable`
  [tests/test_frozen_paper_cli.py, tests/test_frozen_paper_isolation.py, tests/_frozen_paper_world.py]
- [x] [Review][Patch] Contract wording: `diagnostic` and `stderr_truncated` meanings, the `_late`
  encoding refusal, the §5 chokepoint; Story 1.3c's `frozen_qualification_pending` marked superseded;
  the architecture map; the v48 forward-only rollback note [deploy/systemd/README.md]
- [x] [Review][Dismiss] Explicit id −1, REPLACE/DELETE of a tick row: unreachable from any code path
- [x] [Review][Dismiss] Replay tests mock environment verification: real verification is covered by
  the promotion tests and the opt-in real-environment test
- [x] [Review][Dismiss] A dependency-change test: tripwires prove frozen paths never compute the
  checkout identity
- [x] [Review][Defer] Raw `registry transition --to forward_tested --actor human` is unauthenticated
  and skips the forward gate for non-frozen strategies — pre-existing (#682)
- [x] [Review][Defer] Tests write fixture strategy modules into the source tree (Story 1.3c residual);
  concurrent runs in one worktree can collide on them

## Dev Agent Record

### Implementation Plan

Contract first (PR #681), then disjoint slices implemented test-first by Claude subagents and
committed by the coordinator: the `FrozenAttempt` value; attempt recording in the supervisor port;
schema v48 and the evidence store; deterministic replay; CLI recording and tick linkage; the
admissibility filter and the promotion chokepoint. Then one bounded independent review round and
fixes.

### Completion Notes

- Every frozen planner attempt the supervisor decides to run a child for is one append-only
  `frozen_invocations` row, written after full judgement and committed immediately; supervisor-settled
  phases and systemic faults record nothing.
- A frozen tick exists only with its link to the successful Phase B attempt of the same deployment
  and snapshot (one INSERT, trigger-enforced, link immutable); forward evidence for a frozen epoch
  admits only linked ticks (`invocation_unlinked`).
- `paper promote` qualifies a frozen deployment from its recorded descriptor after fresh bundle and
  environment verification, never from the checkout; the raw edge refuses frozen deployments and
  go-live stays refused until Epic 2.
- Recorded attempts replay byte-exact after a restart with the checkout module edited or deleted,
  without Git or uv.

### File List

Production: algua/contracts/frozen_evidence.py; algua/live/{frozen_attempt,frozen_dispatch}.py;
algua/registry/db/{frozen_evidence,migrate,schema,constants}.py; algua/registry/store/frozen_evidence.py;
algua/execution/tick_snapshots.py; algua/registry/{forward_evidence,forward_promotion,frozen_runtime,
human_actor,promote_run,transitions,artifact_errors}.py; algua/cli/{paper_cmd,errors}.py.
Protection and docs: CODEOWNERS, pyproject.toml (import contract), tests/test_repo_hygiene.py,
docs/architecture.md, docs/contracts/cli-error-envelope.md, deploy/systemd/README.md, the Story 1.3d
spec folder and the Story 1.3c spec supersession notes. Tests: tests/test_frozen_attempt.py,
tests/test_frozen_evidence_{schema,store}.py, tests/test_forward_evidence_frozen.py,
tests/test_frozen_promotion.py, tests/test_frozen_replay.py, tests/_frozen_replay.py,
tests/_frozen_evidence_helpers.py, tests/contracts/test_frozen_evidence.py and updates to the frozen
dispatch, paper CLI, isolation, promotion-refusal, forward-promotion, human-actor and registry tests.

### Change Log

- 2026-10-01: Contract and readiness (PR #681); implemented frozen evidence and qualification; review
  round 1 applied; moved to review.
- 2026-10-01: Merged (PR #683, `14621bf`); schema v48 applied in production; done.

## Development notes

- Gate enforcement must trust recomputed/verified identities, not audit prose or runtime diagnostics.
- Prefer append-only attempt and linkage records over an updateable status row. If atomicity requires
  a link table, protect it from update/delete and enforce one successful final invocation per tick.
- Keep forward evaluation scoped to one explicit deployment ID as Story 1.2 established.
- Promotion must use the same artifact that traded; no rebuild or re-freeze occurs at promotion.
- Do not broaden this story into controlled fleet migration or the later signed live ceremony.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

## References

- [Parent Story 1.3](1-3-materialize-and-execute-frozen-planner-artifacts.md)
- [Approved Sprint Change Proposal](../sprint-change-proposal-2026-09-25.md)
- [Artifact-freeze design](../../superpowers/specs/2026-09-22-artifact-freeze-design.md)

