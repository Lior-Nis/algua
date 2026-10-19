# Implementation readiness — Research strategy board

**Date:** 2026-10-19  
**Branch:** `feat/strategy-board` at `76ebfd4`  
**Status:** ready after human approval of the draft story

## Scope and seams

The story is one cohesive read-only vertical slice. The registry already contains stable strategy
IDs and scalar gate-run metrics. The new projection can select one latest gate per identity without
a migration, backfill, broker access, holdout read, or lifecycle change. The CLI remains the only
monitor integration seam; FastAPI only invokes the typed JSON command.

## Entry-point inventory

- Domain read: `strategy_board_payload(repo)`.
- Human/agent JSON read: `algua runs board`.
- Monitor read: `GET /api/strategy-board`.
- UI consumption: Research route `StrategyBoard`.
- Existing `/api/runs`, strategy detail, promotion, transition, and series paths remain unchanged.

No path reaches a guarded state change.

## Risk audit

- **Selection bias:** latest is fixed by timestamp and ID; metric ordering cannot choose another run.
- **Identity:** only `strategy_id` joins; legacy name-only evidence remains unattached.
- **Holdout:** only existing scalar columns are projected; no series accessor is called.
- **Completeness:** registry is the left side, so unevaluated strategies cannot disappear.
- **Presentation:** no composite score or recommendation; comparison is labeled exploratory.
- **Compatibility:** additive CLI/API contract; existing run views and detail routes are untouched.

## Readiness verdict

No safety or architecture blocker was found. Implementation should proceed test-first after the
story's frozen intent is approved. The only planned behavior change is replacement of the Research
screen's displayed multi-run ranking with the complete latest-evidence strategy board.

