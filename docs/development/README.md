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

[Story 1.1 — planner extraction](stories/1-1-extract-in-process-decision-planner.md) is implemented
and independently reviewed, with local verification recorded in the story. This is not a main
merge or deployment; subsequent stories remain unprepared. The
[readiness assessment](implementation-readiness-report-2026-09-24.md) covers only this story,
not either whole epic. [Deferred findings](stories/deferred-work.md) retain pre-existing issues
outside the behavior-preserving extraction.

Workflow path bindings: `planning_artifacts = docs/development`,
`implementation_artifacts = docs/development/stories`; use `docs/PRD.md` and
`docs/architecture.md` directly rather than copying them into the planning folder. There is no
repository `_bmad` configuration at preparation time; installed skill defaults apply. Sprint
tracking, when initialized, belongs at `docs/development/stories/sprint-status.yaml`.
