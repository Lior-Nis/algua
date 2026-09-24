---
stepsCompleted: [1, 2]
status: epics-approved-third-story-prepared
scope: phase-1-operating-kernel
inputDocuments:
  - docs/PRD.md
  - docs/architecture.md
  - docs/vision-reconciliation.md
  - docs/superpowers/specs/2026-09-22-artifact-freeze-design.md
externalInputs:
  - https://github.com/Lior-Nis/algua/issues/624
  - https://github.com/Lior-Nis/algua/issues/661
---

# Algua — Phase 1 epic breakdown

## Overview

Lior approved an artifact-freeze-first Phase 1 cycle on 2026-09-24. The canonical vision was
merged in PR #665 (`c4e8c8f`). Lior subsequently approved both delivery outcomes and preparation
of the first planner story. This does not reopen the vision or authorize a change to a safety wall.

The current milestone is one strategy whose qualified decision artifact can keep operating
while development continues, followed by evidence that it can operate within approved capital
and safety limits. No profitability, gate pass or completion date is promised by the milestone.

Later PRD phases remain governed by PRD §25. Phase 1 must accommodate daily/hourly evolution
without implementing hourly execution or full model/market breadth in this cycle. Research
memory and autonomous repair remain planned; existing provenance and reviewed release mechanics
are prerequisites for this slice, not claims that those later phases are complete.

## Requirements inventory

### Functional requirements

- FR1: Execute one strategy through research qualification, paper evidence and human-authorized
  live operation with the same behavior-affecting artifact (PRD §§5, 7, 10, 24).
- FR2: Separate decision computation from registry, broker credentials, cancellation, submission,
  reconciliation and account-wide controls. A versioned planner accepts exact recorded inputs
  and returns target weights and order intents (artifact-freeze design, slice 2).
- FR3: Preserve current tick results, safety failure semantics, warm-up behavior, sizing and
  order/hook ordering while extracting the in-process planner. No new operating permissions.
- FR4: Record immutable artifact identity, resolved strategy configuration, complete environment
  fingerprint and planner protocol version; recover the executable artifact from that identity.
- FR5: Record explicit deployment activation/retirement and stamp each tick with its deployment.
  Evaluate forward evidence within one deployment epoch; no pre-activation back-crediting,
  mixed-artifact evidence or reuse of abandoned epochs.
- FR6: Execute the planner from immutable deployed content while the current supervisor owns the
  shared registry, credentials, account state, risk controls and broker effects. Planner access
  to those authorities is prevented by the reviewed isolation design.
- FR7: Migrate qualified strategies through a controlled, auditable workflow covering research
  qualification, approval/token identity and family anchors. Failed/interrupted migration cannot
  leave a strategy falsely qualified or silently active. No ad hoc database stage edits.
- FR8: Bind authenticated live authorization to the exact deployment and manifest that earned
  evidence, preserving freshness, single-use challenges, revocation and existing human authority.
- FR9: Keep decision-affecting repairs as new deployments requiring new qualification/evidence.
  Preserve prior artifacts and evidence for audit; outside-planner fixes must not mutate them.
- FR10: Make missing/corrupt artifacts, unsupported planner versions, stale inputs and invalid
  results prevent new exposure, with diagnosable evidence and a safe recovery path.
- FR11: Enforce the approved experimental account's long-only stocks, unleveraged ETFs,
  no-borrowing/no-derivatives policy and capital ceiling; implement the vision's 10% pause under
  owner-decided measurement/response/resumption semantics (PRD §§19–21).
- FR12: Preserve per-order traceability from deployment and input snapshot through decision,
  sizing, risk, intent, broker execution and resulting positions (PRD §15).
- FR13: Demonstrate 72 hours unattended in paper/shadow either operating correctly or reaching
  a predefined safe state, including controlled restart/reconciliation and incident evidence.
  Real-money acceptance additionally requires human authorization and account readiness.
- FR14: Validate releases in paper/shadow and preserve the normal evidence and human activation
  pipeline. Signed-relaxation policy under the new budget requires an owner decision; neither
  the old dollar rungs nor this plan grant an exception (PRD §§16–19).

### Non-functional requirements

- NFR1: Reproducibility covers code, artifact, data/configuration and environment; behavior-changing
  dependency/interpreter/ABI/platform differences cannot silently reuse an artifact identity.
- NFR2: Deterministic, inspectable and testable construction, risk and execution; preserve
  point-in-time data, anti-look-ahead timing, closed-bar decisions and all current gate contracts.
- NFR3: One product and repository, a modular monolith with isolated planner/runtime boundaries;
  no production fork, speculative service split or generic plugin framework.
- NFR4: Fail closed on invalid authority or unverifiable risk state. Agents cannot expand their
  own permissions; external/runtime text is evidence, never instructions granting authority.
- NFR5: Full root quality gate stays green. Tests must exercise real parity and failure behavior;
  import boundaries and module-size ratchets stay intact without weaker tests or exemptions.
- NFR6: CLI success/error responses remain parseable JSON with meaningful exit codes.
- NFR7: Existing compute/resources first. No new paid commitment or capital increase outside the
  approved budget/authority. Building this cycle does not authorize a trade or account transfer.
- NFR8: Permanent audit/provenance survives failures and repairs. Logs are operational evidence,
  not a tamper-evident trust anchor. Preserve that accepted threat-model distinction.

### Architecture and implementation requirements

- Brownfield work: reuse current `algua/live/live_loop.py::run_tick`, `paper_loop.py::decide`,
  strategy loading, tick ledgers, registry gates and operator seams. No starter template.
- At present `run_tick` reads provider/broker state and calls side-effecting hooks before and
  after decision computation. Its frozen boundary must use captured values, not move broker,
  registry or hook objects into a nominally pure planner.
- Preserve held-symbol valuation during warm-up, universe-only decision input, unusable/stale
  mark rejection, reconciliation failures, drawdown/gross checks and pre-/post-cancel and
  per-order halt checks. Account buying-power reservation remains supervisor-owned.
- `tests/test_live_loop.py`, mark-freshness/risk tests and `tests/test_lane_parity.py` are core
  regression sources. Characterize existing behavior before extracting it; an extraction that
  changes decisions or error ordering needs an explicit design decision, not a hidden fix.
- Deployment tables and epoch enforcement require protected registry/schema review; migrations
  run only from the current supervisor, never from a frozen checkout.
- The complete environment fingerprint includes dependency digest, Python interpreter, ABI and
  platform. Sharing environments is permitted only when the full identity matches and deployed
  contents are immutable. Do not copy `.env` or credentials into an artifact.
- Select artifact representation and transport in the detailed architecture/story preparation
  from evidence about current deployment constraints. First extraction stays in-process; it
  does not need a premature packaging or IPC framework.
- Narrowing `code_hash` to selected policy callables is separately proposed wall-changing work
  (slice 7), excluded from this cycle's default authorization and critical path.
- Missing/corrupt-artifact supervisor policy, artifact retention/garbage collection and migration
  recovery must be settled before their implementation stories become ready for development.
- The inherited calendar-purity discrepancy remains flagged in the reconciliation record.
  This cycle must not resolve it by weakening imports or safety contracts incidentally.

### UX requirements

No new monitor/UI scope. Existing typed CLI and JSON behavior is the integration surface for this
cycle. The web viewport CI check passed on retry before PR #665 merged; no web change was made.

## Approved epic list

### Epic 1: Preserve qualified strategy behavior while development continues

Outcome: a strategy can accumulate trustworthy paper evidence against a recoverable immutable
deployment while unrelated repository development proceeds. Deliver the planner extraction,
deployment/epoch records, frozen execution and controlled migration as separately reviewed slices
within this outcome. Each slice must leave the existing system usable; explicit temporary
limitations remain documented until frozen execution is delivered.

Coverage: FR2–FR7, FR9–FR10; the paper portion of FR1 and FR12. Uses existing daily operation.
Does not depend on live sign-off or Epic 2 to be useful. Linked work: #661 slices 2–5.

### Epic 2: Operate one approved deployment within the personal-capital envelope

Outcome: the same evidence-bearing deployment can be human-approved for live operation, with
traceable execution, enforced experimental-capital policy and demonstrated unattended safe failure.
Build on Epic 1; do not rebuild or re-freeze the artifact during live activation.

Coverage: live portion of FR1 and FR12; FR8, FR11, FR13–FR14. Linked work: #661 slice 6 and #624.
The owner tasks below gate policy-dependent stories, not Epic 1's planner extraction.

### Requirement coverage map

| Requirements | Delivery outcome |
|---|---|
| FR1 | Epic 1 paper qualification, Epic 2 signed live activation |
| FR2–FR7 | Epic 1 |
| FR8 | Epic 2 |
| FR9–FR10 | Epic 1; Epic 2 retains the behavior |
| FR11 | Epic 2 |
| FR12 | Epic 1 deployment/tick linkage, Epic 2 broker/position traceability |
| FR13–FR14 | Epic 2 |
| NFR1–NFR8 | Both epics; enforced per story |

## Human decisions and board links

Assigned to Lior on the Algua board, without invented deadlines:

- [10% pause/resumption policy](https://app.todoist.com/app/task/6hcQ3C4cVCgX8HQG): equity reference,
  cash flows, threshold equality, orders/positions and human-reviewed resumption.
- [Signed-relaxation policy](https://app.todoist.com/app/task/6hcQ3C8gGj7Vrx5p): explicit treatment
  of authenticated exceptions under the new experimental budget, before live qualification.
- [Account/deployment prerequisites](https://app.todoist.com/app/task/6hcQ3CHr49rCRV3p): capital and
  instrument restrictions, key custody, protected merges and immutable signing anchor before
  activation. Coordinate with the existing VPS task; never paste credentials into task comments.

## Prepared implementation stories

### Story 1.1: Extract the in-process decision planner

As the operator, I want decision computation separated from broker and registry effects so that
we can introduce immutable execution without changing qualified strategy behavior.

Given a recorded daily tick and equivalent strategy state, when the extracted planner runs,
then weights, ordered intents, errors and supervisor effect ordering match the current path.
Given warm-up, invalid marks or reconciliation failure, when the supervisor processes the tick,
then existing early returns and safety checks still occur before prohibited effects.
Given the extracted boundary, when its dependencies are inspected, then the planner receives
explicit values rather than broker, registry, provider or hook authority.

Full acceptance criteria, implementation tasks and current-code traps are in
[Story 1.1](stories/1-1-extract-in-process-decision-planner.md).

Story 1.1 is implemented, independently reviewed and merged in PR #667. Its merge is not a
deployment.

### Story 1.2: Record working-tree deployments and evaluate one explicit epoch

As the operator, I want every newly admitted paper strategy and completed tick bound to an explicit
deployment epoch so forward evidence cannot be back-credited or mixed across deployments.

This story records and enforces the epoch while execution still uses the current working tree. It
also closes the protected same-hash certificate reuse hazard created by explicit redeployment. It
does not mint a frozen executable, migrate the existing fleet or redesign the signed live ceremony.

Full acceptance criteria, transaction boundaries, migration rules and current-code traps are in
[Story 1.2](stories/1-2-record-working-tree-deployments.md).

### Story 1.3: Materialize and execute frozen planner artifacts

As the operator, I want each newly admitted paper strategy's planner to execute from recoverable
immutable content so it can accumulate trustworthy evidence while repository development continues.

The approved design uses a deterministic content-addressed source bundle and copied model assets,
one short-lived planner subprocess per strategy, a shared environment only for a complete matching
dependency/interpreter/ABI/platform fingerprint, Parquet inputs and strictly validated JSON output.
The current supervisor retains every shared authority and broker effect. Existing working-tree and
legacy tenants remain explicit until controlled migration; artifacts are retained without GC.

Full acceptance criteria, transaction/filesystem boundaries, transport validation, failure policy
and current-code integration seams are in
[Story 1.3](stories/1-3-materialize-and-execute-frozen-planner-artifacts.md).

Stories 1.1–1.3 are prepared. Epic 1 migration and Epic 2 require their own detailed stories and
readiness review. Approval of the outcomes is not approval of protected implementation changes. No
sprint completion is claimed.
