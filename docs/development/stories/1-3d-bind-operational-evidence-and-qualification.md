---
baseline_commit: dc2a222ef811dc1c3a835d656a411de58423796c
---

# Story 1.3d: Bind operational evidence and qualification

Status: backlog

Prepared: 2026-09-25. Baseline: Story 1.3 readiness baseline `dc2a222` (PR #671).
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

- [ ] Design append-preserving invocation/link records and obtain protected schema review (AC1–AC3).
- [ ] Record both phases and atomically bind successful final results to ticks (AC1–AC3).
- [ ] Require complete frozen linkage in forward-evidence queries/evaluation (AC4).
- [ ] Add immutable bundle/environment verification to evaluation and promotion (AC5–AC6).
- [ ] Remove the temporary promotion block only behind the complete verifier (AC6).
- [ ] Add restart/replay and mutable-checkout independence tests (AC7).
- [ ] Exercise the full tenant/systemic failure and evidence matrix (AC8–AC9).
- [ ] Independently review evidence authority and live-wall preservation (AC10).

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

