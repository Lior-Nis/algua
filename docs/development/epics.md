---
stepsCompleted: [1, 2]
status: epics-approved-story-1-3-decomposed
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

Prepared 2026-10-04 as ten dependency-ordered stories, 2.1–2.10 (see "Prepared implementation
stories" below). The two policy decisions it needed were made on 2026-10-04. The account and
deployment prerequisites task stays open; it gates activation, not construction. Linked issues:
#614, #647, #648, #682, #685.

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

- [10% pause/resumption policy](https://app.todoist.com/app/task/6hcQ3C4cVCgX8HQG): **decided
  2026-10-04** ([#624](https://github.com/Lior-Nis/algua/issues/624)). Drawdown is measured from the
  live account's equity high-water mark, and deposits and withdrawals move the peak by the same
  amount. The pause triggers at a drawdown of 10% or more. On trigger: engage the global live halt,
  cancel resting orders and block new orders; open positions are kept. Resumption only through a
  signed human command after a drawdown report, and it re-bases the peak to current equity.
  Implemented by Story 2.6.
- [Signed-relaxation policy](https://app.todoist.com/app/task/6hcQ3C8gGj7Vrx5p): **decided
  2026-10-04** ([#624](https://github.com/Lior-Nis/algua/issues/624)). Signed relaxations remain
  available for research and paper exploration. A relaxed research gate or forward certificate can
  never authorize go-live; experimental live qualification must pass every gate at its protected
  default. Agents receive no waiver authority. Implemented by Story 2.1.
- [Account/deployment prerequisites](https://app.todoist.com/app/task/6hcQ3CHr49rCRV3p): **still
  open.** A human-access acceptance check before activation: capital and instrument restrictions,
  key custody, protected merges and immutable signing anchor (see #648). Coordinate with the
  existing VPS task; never paste credentials into task comments. It gates activation, not the
  construction of any Epic 2 story. The activation inputs it must supply are the USD ceiling
  equivalent to ₪2,000 and the permitted instrument list (Story 2.5), plus live credentials and the
  live bars provider (Story 2.3).

Decisions surfaced while preparing Epic 2, made by Lior on 2026-10-04:

- **Release validation (Story 2.8):** a supervisor release may run live only after at least one
  completed paper cycle on the identical release that ticked a frozen tenant with a clean reconcile
  and a complete trace audit; live stays refused when the paper book is empty.
- **Alerts (Story 2.10):** email to the owner through the existing kaggler SMTP sender, driven by
  systemd `OnFailure=` hooks and the `fleet health` watchdog.
- **#682:** the raw `registry transition --to forward_tested|candidate` edges are removed for every
  actor; the promote commands are the only ways in (Story 2.1).

Still open:

- **Not blocking:** whether the experimental account keeps the existing liquidating book breaker
  (15% drawdown, 5% daily loss, peak not cash-flow adjusted) alongside the 10% pause; whether the
  pause's global halt should keep stopping the paper lane; whether `live flatten` stays available
  while paused (Story 2.6); and whether the capital ceiling bounds capital at risk rather than the
  account balance (Story 2.5).

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

### Story 1.3: Materialize and execute frozen planner artifacts (parent outcome)

As the operator, I want each newly admitted paper strategy's planner to execute from recoverable
immutable content so it can accumulate trustworthy evidence while repository development continues.

The 2026-09-25 readiness review found that this approved outcome was too large for one implementation
story. It remains the complete requirement and traceability record, but its status is `decomposed`.
The current supervisor retains every shared authority and broker effect. Existing working-tree and
legacy tenants remain explicit until controlled migration; artifacts are retained without GC.

Full acceptance criteria, transaction/filesystem boundaries, transport validation, failure policy
and current-code integration seams are in
[Story 1.3](stories/1-3-materialize-and-execute-frozen-planner-artifacts.md).

The approved implementation sequence is:

1. [Story 1.3a — complete the two-phase planner boundary in-process](stories/1-3a-complete-two-phase-planner-boundary-in-process.md)
   (`done`);
2. [Story 1.3b — materialize and verify recoverable planner artifacts](stories/1-3b-materialize-and-verify-recoverable-planner-artifacts.md)
   (`done`);
3. [Story 1.3c — execute frozen planners in paper](stories/1-3c-execute-frozen-planners-in-paper.md)
   (`done`);
4. [Story 1.3d — bind operational evidence and qualification](stories/1-3d-bind-operational-evidence-and-qualification.md)
   (`done`).

Stories 1.1, 1.2 and 1.3a–1.3d are done, so the Story 1.3 parent outcome is complete.

### Story 1.4: Controlled exit of the legacy paper cohort

As the operator, I want every legacy paper tenant stopped, liquidated and retired so the only
strategies that trade in paper from now on are frozen deployments admitted through the qualified
path.

The owner decided on 2026-10-01 to retire the whole cohort rather than requalify any of it, so the
story fixes the three defects that kept the existing `paper flatten` and `paper -> retired` from
exiting tenants on the shared account, then retires them. Full acceptance criteria are in
[Story 1.4](stories/1-4-controlled-exit-of-the-legacy-paper-cohort.md) and its normative
[contract](specs/spec-story-1-4-legacy-cohort-exit/SPEC.md).

**Epic 2 stories, prepared 2026-10-04** from FR1, FR8 and FR11–FR14, the two owner decisions
recorded on #624, and the current code at `e875b6d`. They are dependency-ordered. Stories 2.1 and
2.2 are independent and design-complete; every later story waits for its predecessor. Each still
requires its own contract and readiness review before implementation (see "Contract and readiness
first" in [story delivery](../agent/story-delivery.md)). None authorizes activation, funding or a
capital change. No sprint completion is claimed.

### Story 2.1: Refuse live qualification built on relaxed gates

Records the exact relaxation set on every new research and forward-gate row, classifies existing
rows once at migration, and refuses go-live, at challenge issuance and at completion, unless both
the deployment's research gate and its forward certificate are unrelaxed. No flag or signature
waives it, and exploration keeps every signed relaxation. It goes first because rows minted before
it carry no recorded set. [Story 2.1](stories/2-1-refuse-live-qualification-on-relaxed-gates.md):
`ready-for-dev`.

### Story 2.2: Drain a strategy's resting paper orders on every paper-lane exit

Fixes #685: every paper-source book exit, including `forward_tested -> live`, cancels the strategy's
own resting paper orders, ingests fills, and refuses while an order remains, failing closed without
the venue. Needed before the signed go-live and the unattended paper run.
[Story 2.2](stories/2-2-drain-resting-paper-orders-on-lane-exit.md): `ready-for-dev`.

### Story 2.3: Bind the signed live authorization to the exact frozen deployment

Changes the go-live signature from the checkout identity to the deployment: id, manifest, bundle and
environment digests, research gate, forward certificate, live bars provider (#614) and live account.
Trade time re-verifies against the deployment record; leaving live or `live revoke` ends the
authorization; go-live becomes frozen-only and never rebuilds. #661 slice 6.
[Story 2.3](stories/2-3-bind-live-authorization-to-the-frozen-deployment.md): `backlog` until 2.1
and 2.2 merge.

### Story 2.4: Hand a frozen deployment off from paper and execute it in the live lane

Adds an agent-allowed `paper handoff` that leaves the paper book flat without invalidating the
certificate, then runs the authorized deployment's planner in the live lane from its verified
bundle, with invocation evidence and tick linkage, while the supervisor keeps every live authority.
[Story 2.4](stories/2-4-hand-off-and-execute-frozen-deployments-live.md): `backlog` until 2.3
merges.

### Story 2.5: Enforce the experimental capital envelope

A protected, fail-closed policy: capital ceiling, long-only, gross at most 1.0, permitted
instruments only, cash (never margin) for buys, refusal of leveraged or shorting deployments at
go-live, and signed human approval for any live allocation increase. Building needs no owner input;
activating needs the USD ceiling and the instrument list.
[Story 2.5](stories/2-5-enforce-the-experimental-capital-envelope.md): `backlog` until 2.4 merges.

### Story 2.6: Pause live trading at a 10% drawdown of the experimental account

Implements the owner's pause: a cash-flow-adjusted high-water mark, an inclusive 10% trigger, global
halt plus cancel of resting orders with positions kept, a latch no agent path can lift, a stored
drawdown report and a signed resume that re-bases the peak. The existing book breaker is unchanged.
[Story 2.6](stories/2-6-pause-live-trading-at-ten-percent-drawdown.md): `backlog` until 2.5 merges.

### Story 2.7: Trace every order from its deployment to the resulting position

An immutable trace row written before each paper and live POST links deployment, invocation, bars
snapshot, decision, sizing and risk decision to the broker order, fills and position. Live intent
becomes crash-safe at parity with paper; `trace order` and `trace audit` read the chain.
[Story 2.7](stories/2-7-trace-orders-from-deployment-to-position.md): `backlog` until 2.6 merges.

### Story 2.8: Validate each supervisor release in paper before it trades live

Records the supervisor release identity on every paper and live cycle and refuses live strategy
ticks on a release that paper has not validated, while every risk-reducing step still runs. Ships
the live operator job and units disabled; enabling them is part of activation.
[Story 2.8](stories/2-8-validate-releases-in-paper-before-live.md): `backlog` until the owner
decides the minimum paper validation and 2.7 merges.

### Story 2.9: Define the safe states and assemble incident evidence

A reviewed catalogue of every state the system may enter on its own, tied to the code by a test;
operator alerts recorded in the registry; `fleet incidents`; and a fleet-health watchdog unit.
[Story 2.9](stories/2-9-define-safe-states-and-incident-evidence.md): `backlog` until 2.8 merges.

### Story 2.10: Run the 72-hour unattended acceptance exercise

A planned, drill-based 72-hour paper run with controlled restart, data outage and halt drills,
judged against the Story 2.9 catalogue, with an evidence report. No real-money acceptance.
[Story 2.10](stories/2-10-run-the-72-hour-unattended-acceptance.md): `backlog` until 2.1–2.9 are
deployed, a frozen tenant is ticking in paper and the owner has chosen the alert destination.

Outside Epic 2, with reasons: #682 (unauthenticated raw human edges) does not widen live authority,
because go-live independently requires a fresh certificate and a signature and frozen deployments
refuse the raw edge. #647 (paper whole-account breaker and lane unification) limits what paper can
validate for live-only code; Story 2.8 records that limit. #648 (GitHub enforces code-owner review
on three paths) belongs to the open prerequisites task.
