# Development cycle

Product authority is [the canonical vision](../PRD.md). Current scope is the artifact-freeze-first
Phase 1 approved by Lior on 2026-09-24; [epics.md](epics.md) contains extracted requirements and
the approved delivery outcomes. This is the working location for BMAD planning artifacts in
this brownfield repository; existing design/spec history stays under `docs/superpowers/`.

Use a bounded cycle for each independently reviewable delivery:

1. `bmad-create-epics-and-stories`: confirm requirements and user outcomes, then create stories
   with testable acceptance criteria, dependencies and PRD references.
2. `bmad-create-story`: inspect current code and tests; prepare the next story with exact paths,
   required behavior, safety boundaries, verification and recovery considerations.
3. `bmad-check-implementation-readiness`: validate that the scoped requirements, architecture
   and story agree. Resolve relevant policy/authority questions before implementation.
4. `bmad-sprint-planning`: record eligible stories in `sprint-status.yaml`. Future/blocked work
   stays backlog; evidence, not file creation alone, determines readiness.
5. `bmad-dev-story`: implement test-first on a branch, preserving contracts and existing behavior
   unless a reviewed story explicitly changes it. Run the full repository quality gate.
6. `bmad-code-review`: review acceptance coverage, correctness, failure paths and boundary
   preservation; fix findings and verify. Protected changes require human review before merge.
7. Validate the permitted release in paper/shadow, record deployment/verification evidence and
   update story status. A code merge is not authorization to deploy, trade or allocate capital.
8. `bmad-retrospective`: capture reusable lessons after the epic and feed the next cycle.

GitHub issues track implementation work and reviews. Repository artifacts retain specifications
and acceptance evidence. The [Algua Todoist board](https://app.todoist.com/app/project/algua-6hGjQph9JgcgpC2G)
tracks Lior's actual decisions/access actions. Avoid duplicating generic engineering chores onto
the human board. Respect existing tasks and assignments; do not infer completion from a title.

The 72-hour exercise and actual live sign-off are later acceptance events. Until they occur,
reports must distinguish tests passing, code merged, deployment verified and capital authorized.

## Current handoff

As of 2026-10-22. Merged code is not a deployment, and nothing below authorizes live trading or
capital.

**Epic 1 (immutable paper operating kernel): done, pending one piece of evidence.** Stories 1.1,
1.2 and 1.3a–1.3d are done; [Story 1.3](stories/1-3-materialize-and-execute-frozen-planner-artifacts.md)
is the decomposed parent (see the
[Sprint Change Proposal](sprint-change-proposal-2026-09-25.md)).
[Story 1.4 — legacy cohort exit](stories/1-4-controlled-exit-of-the-legacy-paper-cohort.md) stays
`review`: its code merged in PR #686, and only AC7 (retire the last legacy tenant, confirm the paper
account is flat, restart the paper timer) still needs recorded evidence. The
[Epic 1 retrospective](epic-1-retrospective-2026-10-04.md) is done; its action items A1–A13 remain
the follow-up list.

**Epic 2 (approved deployment within the personal-capital envelope): in progress.** Stories are in
[epics.md](epics.md) and `stories/2-*.md`.
- [Story 2.2 — paper exit drain](stories/2-2-drain-resting-paper-orders-on-lane-exit.md) is done
  (PR #693; contract and readiness in PR #692).
- [Story 2.1 — refuse live qualification on relaxed gates](stories/2-1-refuse-live-qualification-on-relaxed-gates.md)
  is next and `ready-for-dev`
  ([readiness report](implementation-readiness-report-2026-10-05-story-2-1.md),
  [contract](specs/spec-story-2-1-unrelaxed-live-qualification/SPEC.md)). Only AC10 has shipped,
  via PR #700 (#682).
- Stories 2.3–2.10 are `backlog`. Each has a story file but no contract under `specs/` and no
  readiness report yet; each waits for its predecessor, then its own contract and readiness review.

**Shipped outside the epics:** the [research strategy board](stories/spec-strategy-board.md)
(PR #705) and the [dashboard decision layer](stories/spec-dashboard-decision-layer.md) (PR #706).
The [planner NaN-weight and symbol-label contract](stories/spec-planner-nan-and-label-contract.md)
merged in PR #711; its implementation is in review (PR #712).

[Deferred findings](stories/deferred-work.md) keep pre-existing issues that fall outside the active
story boundary. Open owner decisions are listed in [epics.md](epics.md) ("Human decisions and board
links").

Workflow path bindings: `planning_artifacts = docs/development`,
`implementation_artifacts = docs/development/stories`; use `docs/PRD.md` and
`docs/architecture.md` directly rather than copying them into the planning folder. There is no
repository `_bmad` configuration at preparation time; installed skill defaults apply. Sprint
tracking is `docs/development/sprint-status.yaml`.
