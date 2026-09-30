---
id: SPEC-story-1-3c-frozen-paper-execution
companions:
  - frozen-execution-contract.md
  - ../../stories/1-3c-execute-frozen-planners-in-paper.md
  - ../spec-story-1-3b-artifact-environment-contract/SPEC.md
sources:
  - ../../stories/1-3-materialize-and-execute-frozen-planner-artifacts.md
  - ../../sprint-change-proposal-2026-09-25.md
  - ../../../superpowers/specs/2026-09-22-artifact-freeze-design.md
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete contract for
> what to build, test and validate. Source documents remain narrative and traceability evidence;
> they do not override this contract.

# Story 1.3c Frozen Paper Execution Contract

## Why

Paper evidence must come from the exact planner that was qualified, not from whatever the mutable
checkout contains on the day of the tick. Story 1.3c admits new paper strategies against verified
Story 1.3b content and runs each planner phase from that content in a fresh short-lived process,
while the current supervisor keeps every authority, risk control and broker effect.

## Capabilities

- id: CAP-1
  intent: Paper intake admits a new candidate only against verified immutable content.
  success: A successful admission records a `frozen` deployment, its allocation and the `paper`
    stage in one transaction bound to the exact verified descriptor; any preparation or
    verification failure leaves no deployment, allocation or stage change.
- id: CAP-2
  intent: A frozen tenant's planner runs from its bundle and environment, one fresh process per
    phase, exchanging only bounded canonical bytes with the supervisor.
  success: Phase A and Phase B each run in their own child launched from the verified interpreter
    with the bundle as the only `algua` source; results and effect traces equal the in-process
    Story 1.3a planner on the representative normal, early and breach fixtures.
- id: CAP-3
  intent: Any frozen-tenant fault is isolated to that tenant before it can cause an effect.
  success: Missing or corrupt content, protocol faults, timeouts, abnormal exits, oversized output
    and invalid results each yield a stable deployment-bound code, zero cancel/submit/hook/
    successful-tick effects for that tenant, a nonzero `trade-tick`, and a `run-all` that continues
    valid siblings; systemic faults stay systemic.
- id: CAP-4
  intent: Frozen behavior is independent of the mutable repository.
  success: Editing, deleting or checking out other code after activation does not change a frozen
    tenant's results; missing content is never rebuilt from `HEAD`; after restart the same stored
    artifact resolves without a Git checkout.
- id: CAP-5
  intent: Frozen deployments cannot be qualified before their evidence contract exists.
  success: Until Story 1.3d, `paper promote` refuses every frozen deployment with
    `frozen_qualification_pending` before any gate evaluation, token use or stage change, and the
    live lane refuses to tick a frozen deployment.

## Constraints

- The wire files, schemas, float encoding, limits, launch argv/environment, invocation directory,
  result validation, failure taxonomy and module placement in `frozen-execution-contract.md` are
  normative.
- The nine protected limits in the story are code constants; raising one requires protected review
  and a new frozen-wire version. An operational setting may only tighten a bound.
- Every new paper admission is frozen. Existing `working_tree` deployments and the fixed legacy
  cohort keep their current tick paths unchanged until the controlled-migration story.
- The supervisor never imports a frozen tenant's strategy module from the checkout. Its view of a
  frozen tenant comes only from the recorded manifest's resolved configuration plus the gate
  universe.
- The child cannot open the registry, construct provider or broker clients, reserve capital, invoke
  hooks or persist operational state; an import-linter contract enforces its import surface.
- Tick execution performs no Git, uv, dependency resolution or download. Content is verified
  offline with the Story 1.3b verifier before a tenant's first dispatch in each cycle.
- The subprocess boundary is capability hygiene and reproducibility, not a sandbox against
  malicious same-UID code.
- No schema change. Invocation evidence, frozen forward evaluation and the removal of the promotion
  block belong to Story 1.3d.

## Non-goals

- Recording invocation evidence or making frozen ticks admissible forward evidence (Story 1.3d).
- Migrating or re-freezing existing working-tree or legacy deployments (controlled-migration story).
- Frozen execution in the live lane, model-backed strategies or non-daily timeframes.
- Garbage collection, leases, remote artifact storage, containers or sandboxing.
- Changing live signing, capital limits, lifecycle authority or the planner's decision logic.

## Success signal

A candidate admitted after this story runs in paper from its verified frozen bundle and environment
through two fresh child processes per decision tick; its results and effects match the in-process
planner on the parity fixtures; every named failure isolates the tenant with a stable code; frozen
promotion is refused with `frozen_qualification_pending`; working-tree and legacy tenants behave
exactly as before; and the full repository gate passes.
