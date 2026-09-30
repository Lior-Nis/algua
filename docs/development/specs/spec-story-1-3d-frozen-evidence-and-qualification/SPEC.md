---
id: SPEC-story-1-3d-frozen-evidence-and-qualification
companions:
  - frozen-evidence-contract.md
  - ../../stories/1-3d-bind-operational-evidence-and-qualification.md
  - ../spec-story-1-3c-frozen-paper-execution/SPEC.md
sources:
  - ../../stories/1-3-materialize-and-execute-frozen-planner-artifacts.md
  - ../../sprint-change-proposal-2026-09-25.md
  - ../../../superpowers/specs/2026-09-22-artifact-freeze-design.md
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete contract for
> what to build, test and validate. Source documents remain narrative and traceability evidence;
> they do not override this contract.

# Story 1.3d Frozen Evidence and Qualification Contract

## Why

Story 1.3c runs frozen paper tenants but refuses to qualify them, because nothing yet binds a tick
to the exact verified planner invocation that produced it. Story 1.3d records every frozen planner
attempt permanently, links each successful frozen tick to its final invocation by construction, and
lets `paper promote` qualify a frozen deployment from those records and its verified content alone.

## Capabilities

- id: CAP-1
  intent: Every frozen planner attempt leaves one permanent, append-only evidence record.
  success: Each dispatch the supervisor decided to run a child for writes exactly one immutable row
    naming its deployment, request, phase, snapshot, exact request and bars digests, Phase A binding,
    and either a validated result digest or a stable failure code with bounded exit metadata; rows
    cannot be updated or deleted.
- id: CAP-2
  intent: A successful frozen tick is bound to the final invocation that produced it.
  success: A frozen tick row can only be inserted carrying a link to a successful Phase B record of
    the same deployment and snapshot, at most one tick per record; the link cannot change afterwards;
    failed or crashed attempts can never yield an admissible tick.
- id: CAP-3
  intent: Forward evidence for a frozen deployment counts only fully linked observations.
  success: The forward gate excludes every frozen tick without a valid link and otherwise applies the
    unchanged Story 1.2 epoch rules and admissibility filters.
- id: CAP-4
  intent: A frozen deployment can be qualified without trusting the checkout.
  success: `paper promote` (agent and human) takes identity from the recorded descriptor, verifies the
    bundle and environment fresh, refuses with a stable code if they do not verify, and otherwise runs
    the normal forward gate; the `frozen_qualification_pending` refusal is removed from this path.
- id: CAP-5
  intent: Recorded evidence can be replayed deterministically after a restart without Git.
  success: A recorded Phase A or Phase B request replayed through the same verified bundle and
    environment reproduces the recorded result digest exactly.

## Constraints

- The exact DDL, triggers, attempt definition, recorder signature, admissibility rule, promotion
  chokepoint and codes in `frozen-evidence-contract.md` are normative, and form the protected schema
  review for this story.
- Evidence records identity by immutable reference (deployment, then artifact, bundle, environment,
  protocol and strategy through trigger-immutable rows), never by copies.
- Invocation evidence stores canonical request bytes (at most 256 KiB, no bars, credentials, raw
  stderr or handles) and digests; bars stay in immutable data snapshots.
- Frozen live operation, go-live, the live certificate and deployment-bound signing stay refused
  until Epic 2. The raw `registry transition` edge to `forward_tested` stays refused for frozen
  deployments; `paper promote` is the only way in.
- Working-tree and legacy tenants, their evidence and their promotion are unchanged.
- Ticks recorded before this story (1.3c era) have no link and never count; there is no backfill.
- No garbage collection, lease, replay command, order-intent linkage or migration.

## Non-goals

- Migrating working-tree or legacy tenants (controlled-migration story).
- Frozen go-live, live certificates or deployment-bound signing (Epic 2).
- A replay CLI, retention policy for data snapshots, or order/broker traceability (Epic 2, FR12).
- Changing the forward gate's statistics or thresholds.

## Success signal

A frozen tenant's attempts are recorded immutably; its ticks carry a trigger-enforced link to their
final invocation; forward evidence counts only linked ticks of the active epoch; `paper promote`
qualifies it from descriptor identity and freshly verified content without touching the checkout;
a recorded request replays to the same result digest; working-tree and legacy behaviour is
unchanged; and the full repository gate passes.
