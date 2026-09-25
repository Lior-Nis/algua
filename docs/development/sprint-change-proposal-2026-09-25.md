---
date: 2026-09-25
status: approved-for-planning-implementation
changeScope: moderate-backlog-reorganization
mode: incremental
trigger: story-1.3-implementation-readiness
inputDocuments:
  - docs/PRD.md
  - docs/architecture.md
  - docs/vision-reconciliation.md
  - docs/development/epics.md
  - docs/development/implementation-readiness-report-2026-09-25.md
  - docs/development/stories/1-3-materialize-and-execute-frozen-planner-artifacts.md
  - docs/superpowers/specs/2026-09-22-artifact-freeze-design.md
---

# Sprint Change Proposal — Decompose frozen planner execution

## 1. Issue summary

The 2026-09-25 implementation-readiness review found that Story 1.3 is not ready to implement as
one story. Its approved outcome is correct, but the story combines five independently risky
deliverables:

1. content-addressed source and asset publication;
2. immutable records and frozen-intake changes;
3. shared environment provisioning and verification;
4. a two-phase Parquet/JSON child protocol and runtime;
5. supervisor integration, failure isolation, provenance and qualification.

The review also identified five unresolved contracts: Phase A/Phase B process lifetime and binding,
the exact source inventory, quantitative resource limits, unsupported model-asset scope and
admission-time environment acquisition.

This is a delivery-granularity problem discovered before implementation. It is not a rejection of
the artifact-freeze design, a failed implementation, a new product requirement or a reason to roll
back Stories 1.1 and 1.2.

## 2. Impact analysis

### Epic impact

Epic 1 remains achievable with the same outcome, requirements and priority. Story 1.3 becomes a
parent requirement record with four ordered implementation stories. The controlled migration story
remains subsequent work. Epic 2 is unchanged and still consumes Epic 1's evidence-bearing frozen
deployment.

No new epic is required. No epic is obsolete, removed, reduced or resequenced.

### Story impact

- Stories 1.1 and 1.2 remain `done`; their behavior and evidence are preserved.
- Story 1.3 changes from `ready-for-dev` to `decomposed` and remains the parent outcome and
  traceability source.
- Stories 1.3a–1.3d are added in dependency order.
- Only Story 1.3a is initially `prepared-for-readiness-review`; 1.3b–1.3d remain `backlog`.
- No sprint status is initialized until 1.3a passes implementation readiness.

### Artifact impact

The PRD requires no change. Its immutable-artifact operating-kernel direction remains valid.

The architecture and artifact-freeze design require status/decomposition clarifications only. The
planner/supervisor boundary, modular-monolith direction, current authority controls, live wall,
accepted no-sandbox residual and migration boundary do not change.

There is no UX impact. The typed CLI and JSON envelope remain the integration surface.

The following planning artifacts will change after final approval:

- `docs/development/epics.md`;
- `docs/development/README.md`;
- the Story 1.3 parent document;
- four new child-story documents;
- `docs/architecture.md`;
- the status/decomposition note in the artifact-freeze design.

The readiness report is retained unchanged as the evidence that triggered this correction. No code,
schema, CI, deployment, capital or authority state changes as part of this planning correction.

### Technical impact

The eventual implementation still changes protected execution, registry and promotion surfaces, but
each boundary will now receive a separate test-first implementation and independent review. The
decomposition prevents a single review from having to validate filesystem publication, dependency
environments, subprocess hygiene, trading parity and evidence authority simultaneously.

## 3. Recommended approach

Use **Direct Adjustment** within Epic 1.

| Option | Viability | Effort | Risk | Decision |
|---|---|---:|---:|---|
| Direct adjustment | Viable | Medium planning; unchanged total delivery scope | Lower than current story | Selected |
| Roll back Stories 1.1/1.2 | Not viable | High | High | No simplification; discards valid prerequisites |
| Reduce or redefine Phase 1 | Not warranted | Medium | High product risk | Would remove required reproducibility/safety |

The selected path preserves the approved outcome and momentum while making every increment
independently completable, reviewable and recoverable. Timeline precision remains evidence-driven;
the proposal adds review gates rather than promising dates.

## 4. Detailed change proposals

### Proposal A — Story 1.3 delivery shape

**Old**

One ready-for-development Story 1.3 combines artifact materialization, environment provisioning,
subprocess execution, failure handling, provenance and qualification.

**New**

Keep Story 1.3 as the approved parent outcome, marked `decomposed`, with four ordered stories:

1. Story 1.3a — Complete the two-phase planner boundary in-process.
2. Story 1.3b — Materialize and verify recoverable planner artifacts.
3. Story 1.3c — Execute frozen planners in paper.
4. Story 1.3d — Bind operational evidence and qualification.

Each story must be independently implemented, tested and reviewed while leaving the repository
usable. This proposal was approved incrementally by Lior on 2026-09-25.

### Proposal B — Story 1.3a: complete the two-phase in-process boundary

Move the complete behavior-affecting per-strategy decision boundary behind typed, brokerless Phase A
and Phase B functions while continuing to run in-process. The frozen boundary includes closed-bar
timing/freshness, held-symbol valuation, universe filtering, warm-up, decision timestamp,
per-strategy equity/drawdown/realized-gross/reconciliation decisions, construction, overlays,
capacity, weights and ordered intents.

Registry access, external data acquisition, provider/broker authority, account/book controls,
cancellation, submission, hooks and persistence remain in the current supervisor. The story makes
no schema, filesystem, environment or subprocess change.

The phases are stateless and independently invocable:

1. Phase A consumes the complete early-input envelope.
2. If late account/sizing values are required, it returns `snapshot_required` and a digest binding
   its result to the request identity and exact early inputs.
3. Phase B receives the same early inputs, captured late values and the Phase A digest.
4. Phase B recomputes Phase A and rejects a mismatch before continuing.

No process-local state may connect the phases. Golden-master tests must prove parity in outcomes,
errors, early returns and supervisor effect ordering. The existing paper path must use this
in-process boundary before the story is complete.

### Proposal C — Story 1.3b: materialize and verify artifacts

For the recorded clean `source_ref`, materialize every Git-tracked regular file beneath `algua/`,
canonical resolved configuration, and planner protocol/manifest metadata. Read exact Git object
bytes rather than mutable working-tree files. Reject links, special files, traversal and normalized
or case-colliding paths.

Exclude `.git`, `.env`, databases, credentials, trust anchors, logs, datasets, documentation,
tests, web files and mutable host paths. `pyproject.toml`, `uv.lock` and `.python-version` from the
same commit are environment-build inputs rather than executable bundle contents. Record the source
commit, tree digest, complete environment fingerprint and digest-derived locators; do not place
absolute host paths in identity.

Materialization may download only distributions already selected by the committed lockfile. It may
not resolve or upgrade dependencies, install Algua editably or bind the environment to a checkout.
Tick execution is offline and never invokes `uv`. Missing locked distributions fail preparation
with a stable retryable error before activation.

Current paper-tradable strategies produce source-only bundles. Reserve manifest asset entries but
reject non-empty assets until a separately reviewed model lane can copy already-verified bytes
without path rereads.

Provide a JSON-emitting preparation/verification command that creates or verifies the append-only
artifact descriptor, stored bundle and environment without activating a deployment, changing stage
or changing trading behavior.

### Proposal D — Story 1.3c: execute frozen planners in paper

Dispatch `source_kind="frozen"` paper tenants from their verified artifact/environment. New eligible
admissions use frozen execution; working-tree and fixed legacy tenants retain their compatibility
paths. Phase A and, when required, Phase B each run in a fresh short-lived child process. Promotion
of a frozen deployment fails closed with `frozen_qualification_pending` until Story 1.3d lands.

`frozen-planner` wire protocol v1 uses protected code constants. It is a named protocol namespace
in the canonical frozen manifest and does not reinterpret existing working-tree descriptor stamps:

| Bound | Value |
|---|---:|
| Timeout | 60 seconds per phase |
| Request metadata | 256 KiB |
| Canonical Parquet input | 256 MiB |
| Stdout JSON | 1 MiB |
| Stderr capture | 64 KiB |
| JSON nesting | 16 levels |
| JSON collection size | 10,000 elements |
| Persisted sanitized diagnostic | 8 KiB |
| Grace before process-group force kill | 2 seconds |

The invocation directory is private and sealed input-only before launch; no child output file is
accepted. Raising a limit requires protected review and a new protocol version. A future operational
setting may only tighten limits.

Invoke the verified environment's absolute interpreter with fixed arguments, `shell=False`, no
stdin, closed descriptors, a fresh process group, isolation/no-user-site flags and a replacement
environment that removes credentials, authority paths, proxy/cloud values, loader injection,
`PYTHON*`, `ALGUA_*`, `ALPACA_*`, active-venv and uv state.

Accept exactly one bounded UTF-8 JSON response. Reject duplicate keys, trailing data, schema or
identity mismatches, non-finite/boolean numeric values, duplicate/out-of-universe symbols and
inconsistent weights/intents. Current supervisor validation still runs before effects.

Invalid tenants produce no cancel/order/downstream-hook/successful-tick effects. Single-strategy
execution exits nonzero. `run-all` continues valid siblings unless the existing semantics classify
the failure as systemic. Parity and deterministic fresh-process replay cover the complete boundary.

### Proposal E — Story 1.3d: bind evidence and qualification

For every Phase A and Phase B attempt, record deployment/artifact/environment/protocol/strategy/
request identities, snapshot and captured-input identity, exact input-byte digest, Phase A binding
digest, validated result digest or stable failure code, timestamps, bounded execution metadata and
successful tick/order-intent linkage. Do not copy raw bars, credentials, raw stderr or authority
state into audit rows.

Frozen observations count only when the tick belongs to the epoch, request/result binding is
complete, immutable content/environment verifies, the protocol is supported, and no successful tick
was written for a partial/failed invocation.

`paper promote` verifies the active frozen deployment from stored immutable content. It must not
recompute identity from the ambient checkout, rebuild missing content, mix epochs or reuse abandoned
evidence. Story completion removes `frozen_qualification_pending`. Live authorization remains
unchanged and requires the later deployment-bound signed ceremony.

Acceptance covers restart without Git, unrelated worktree changes, missing/corrupt/replaced content,
permission drift, partial phases, timeout/termination, tenant isolation, systemic failures, replay,
forward-evidence inclusion/exclusion and promotion refusal/success. Retention remains indefinite and
garbage collection remains deferred.

### Proposal F — planning-artifact synchronization

Preserve the current Story 1.3 document as the parent requirement record, add a child-story coverage
map and change its status to `decomposed`. Add the four story files with 1.3a marked
`prepared-for-readiness-review` and the rest `backlog`.

Update the epic, development index, current architecture and historical design status/decomposition
note. Do not edit the PRD, `AGENTS.md`, `CLAUDE.md`, code, schema, CI, deployments or authority
controls. Do not initialize `sprint-status.yaml` until 1.3a passes readiness.

All six proposals were approved individually by Lior in Incremental mode on 2026-09-25.

## 5. Implementation handoff

### Classification

**Moderate** — backlog reorganization with technical-contract clarification. It does not require a
product replan, but the child stories touch protected boundaries and require independent review.

### Responsibilities

- **Product Owner / planning agent:** apply the approved decomposition, statuses, coverage map and
  documentation synchronization without changing product scope.
- **Architecture reviewer:** confirm each prepared child preserves the approved supervisor/planner,
  authority, reproducibility and no-sandbox boundaries.
- **Developer agent:** implement only the next readiness-approved story, test-first, on a branch;
  keep the full root quality gate green.
- **Independent reviewer:** review acceptance coverage, failure paths, parity and protected-wall
  preservation before merge.
- **Lior:** approve protected/high-impact implementation or authority changes. No new Todoist task is
  required by this correction because the planning decisions are resolved; repository engineering
  work belongs in GitHub issues.

### Sequence

1. Apply the approved planning changes.
2. Run implementation readiness against Story 1.3a.
3. If ready, initialize sprint status with only 1.3a eligible.
4. Implement, run the full gate, independently review and merge 1.3a.
5. Prepare/review/implement 1.3b, then 1.3c, then 1.3d using the same bounded cycle.
6. Prepare controlled migration only after the new-admission frozen path is qualified.

### Success criteria

- Every parent requirement maps to at least one child story with no silent deletion.
- Each child has one coherent implementation boundary and no forward dependency for its stated
  value.
- Current authority and trading behavior remain unchanged by the planning correction.
- 1.3a passes implementation readiness before development begins.
- Every implementation story passes the repository quality gate and independent review.

## 6. Change-navigation checklist record

| Section | Status | Result |
|---|---|---|
| 1. Trigger and context | Done | Story 1.3 readiness finding and evidence identified |
| 2. Epic impact | Done | Epic outcomes retained; internal decomposition only |
| 3. Artifact conflicts | Done | No PRD/UX conflict; planning/status notes require updates |
| 4. Path forward | Done | Direct adjustment selected; rollback/MVP reduction rejected |
| 5. Proposal components | Done | Impact, edits, sequence, roles and success criteria documented |
| 6. Final review and handoff | Done | Approved by Lior on 2026-09-25; route to PO/developer workflow |

## 7. Approval record

Lior approved each of the six edits incrementally and approved this complete Sprint Change Proposal
for implementation on 2026-09-25. Approval covers the planning-artifact reorganization described
here. It does not authorize a runtime deployment, capital use, live activation or a change to any
current safety/authority boundary.
