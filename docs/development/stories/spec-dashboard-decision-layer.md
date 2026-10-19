---
title: 'Dashboard decision layer'
type: 'feature'
created: '2026-10-19'
status: 'done'
baseline_commit: 'c9aa0daa2b62b32238e4199c4fcceb075ce821e3'
context:
  - 'docs/PRD.md'
  - 'docs/development/stories/spec-strategy-board.md'
  - 'docs/development/specs/spec-strategy-board/strategy-board-contract.md'
  - 'docs/superpowers/specs/2026-08-27-monitor-human-facing-redesign-design.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Research and Strategy Detail expose trustworthy data but flatten decisions, supporting evidence, and raw provenance into long report-like pages. Operators must read excessive prose and reconcile lifecycle, gate, and health signals themselves before knowing what matters.

**Approach:** Make Research the graphical control room and Strategy Detail the trader-facing decision report. Lead with concise state, consequence, reason, and next checkpoint; retain terminal-dense evidence behind explicit progressive disclosure.

## Boundaries & Constraints

**Always:** Keep the monitor read-only. Preserve one board row per registry identity, latest-run selection by `(created_at DESC, id DESC)`, lifecycle order as the default, exploratory-only sorting, explicit OOS/WF labels, null-as-unknown semantics, binding/advisory distinctions, stale/partial failure honesty, and the `t→t+1` and frozen data contracts. Keep all evidence reachable, keyboard accessible, and non-color-dependent; keep every route within a 390px viewport.

**Ask First:** Any API/schema change, new derived trading or promotion conclusion, change to gate semantics, removal of existing evidence, or work outside Research, Strategy Detail, their shared evidence components, fixtures, and tests.

**Never:** Add a composite score, winner label, promotion action, “recommended strategy,” best-run selection, hidden substitution of zero for missing data, per-bar holdout disclosure, fabricated funnel conversion rates, or a second dashboard framework.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| Research healthy | Fresh ops, registry, board and ideas | Compact operating state, lifecycle inventory, visual comparison, collapsed idea summaries | N/A |
| Research degraded | Any required source stale, failed or absent | Decision layer says partial/blocked and names the affected source; never shows all-clear | Independent evidence remains available where trustworthy |
| Strategy decision | Detail with stage, gates and/or paper state | Shows state, consequence, reason and next checkpoint before charts | Missing fields produce honest unknowns, not inferred claims |
| Advisory conflict | Gate passes with advisory failures | Overall result and advisory weakness are synthesized without changing the gate verdict | Binding/advisory evidence remains inspectable |
| Thin evidence | Empty/loading/error series or run data | Compact semantic status row replaces a large false-empty chart | Loading, failed, stale and genuinely empty states remain distinct |

</frozen-after-approval>

## Code Map

- `web/frontend/src/screens/Research.tsx` -- research decision layer and compact idea pipeline.
- `web/frontend/src/components/StrategyBoard.tsx` -- protected population view and progressively disclosed controls.
- `web/frontend/src/components/ScatterISOOS.tsx` -- visual honesty check and source-state handling.
- `web/frontend/src/screens/StrategyDetail.tsx` -- strategy decision report and evidence disclosures.
- `web/frontend/src/components/{ReturnOverlay,TrialDistribution,EquityChart}.tsx` -- dense evidence with compact thin-data states.
- `web/frontend/src/theme.css` -- graphical decision hierarchy, disclosure styling, touch targets and desktop adaptation.
- `web/frontend/src/fixtures/` -- realistic decision and evidence states.
- `web/frontend/src/invariants/word-budget.test.tsx` -- lower persistent-copy budget while testing expanded evidence separately.

## Tasks & Acceptance

**Execution:**
- [x] `web/frontend/src/screens/Research.tsx` and tests -- replace the separate loop/funnel report with a fail-closed decision header; compact ideas to one-line summaries with one-at-a-time evidence disclosure.
- [x] `web/frontend/src/components/StrategyBoard.tsx` and tests -- consolidate stage/order choices behind one control while retaining active-query visibility and every selection-bias protection.
- [x] `web/frontend/src/screens/StrategyDetail.tsx` and tests -- add a pure, tested decision derivation; lead with its four-part report and group research/operational evidence into accessible disclosures.
- [x] Evidence components and tests -- distinguish loading, error, stale, incomplete and genuine-empty states; collapse thin evidence without changing populated chart internals.
- [x] `web/frontend/src/theme.css`, fixtures, viewport and word-budget tests -- establish graphical-overview/terminal-evidence hierarchy, ≥44px interactive hit areas, a purposeful desktop layout, and a materially lower collapsed word count.

**Acceptance Criteria:**
- Given either target route loads, when the first meaningful view appears, then the operator can identify current state, consequence and next checkpoint without expanding evidence.
- Given evidence is expanded, when the operator inspects it, then all existing gate, return, trial, lifecycle and provenance detail remains available in a compact terminal-like presentation.
- Given Research is filtered or reordered, when rows render, then the representative run remains the latest identity-bound gate and the UI still labels ordering exploratory.
- Given a mobile or desktop viewport, when the route renders, then there is no horizontal overflow and desktop uses available comparison space without stretching prose.

## Spec Change Log

## Design Notes

Use three explicit layers: **decision** (graphical and immediately visible), **supporting evidence** (charts and gate summaries), and **raw evidence** (dense histories, reasons, IDs and hashes). Progressive disclosure changes visibility, never truth or availability. Snapshot stage counts are inventory, not conversion rates.

## Verification

**Commands:**
- `cd web/frontend && npm run check && npm run build` -- frontend tests, types and production build pass.
- `uv run --project web pytest web/backend/tests -q` -- web API compatibility remains green.
- `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` -- complete repository gate passes.

## Implementation Evidence

Delivered the approved decision/supporting-evidence/raw-evidence hierarchy without backend, API,
schema, safety or live-lane changes. Research now fails closed across its independent sources,
preserves the identity-bound latest-gate board, compacts ideas and exposes detailed evidence on
demand. Strategy Detail now synthesizes lifecycle state, operational consequence, reason and next
checkpoint before progressively disclosed performance, research and provenance evidence.

The bounded review found only patch-level defects. Corrections make operational paper health
dominate advisory context, choose the lifecycle-relevant gate deterministically, refuse missing
required forward evidence, preserve stale plus thin-data meaning, target retries at the failing
source, block on absent/unknown merge-back state, and keep mobile dates and 44px navigation targets.

Verification (2026-10-19):
- Frontend: 43 files / 226 tests; TypeScript, production/demo builds and fixture isolation pass.
- Viewport: all five demo routes fit 390px; mobile evidence date and 44px targets are asserted.
- Web backend: 266 tests pass.
- Root: 6,291 passed, 5 skipped; Ruff and mypy clean; 29 import contracts kept, 0 broken.
- `git diff --check` passes.

## Suggested Review Order

**Operator decision layer**

- Start with the Research entry point and its fail-closed source synthesis.
  [`Research.tsx:68`](../../../web/frontend/src/screens/Research.tsx#L68)

- Verify lifecycle-aware gate selection and risk-first decision precedence.
  [`strategyDecision.ts:90`](../../../web/frontend/src/screens/strategyDecision.ts#L90)

- Follow the four-part report into progressively disclosed evidence.
  [`StrategyDetail.tsx:27`](../../../web/frontend/src/screens/StrategyDetail.tsx#L27)

**Evidence integrity**

- Inspect normalized lane eligibility and stale-plus-thin equity states.
  [`EquityChart.tsx:48`](../../../web/frontend/src/components/EquityChart.tsx#L48)

- Confirm return evidence separates primary curves from auxiliary gate context.
  [`ReturnOverlay.tsx:510`](../../../web/frontend/src/components/ReturnOverlay.tsx#L510)

- Confirm trials preserve independent gate context and retry the failing source.
  [`TrialDistribution.tsx:311`](../../../web/frontend/src/components/TrialDistribution.tsx#L311)

**Interaction and density**

- Review consolidated exploratory controls without representative-run substitution.
  [`StrategyBoard.tsx:121`](../../../web/frontend/src/components/StrategyBoard.tsx#L121)

- Review the graphical decision hierarchy, disclosures and responsive workstation layout.
  [`theme.css:1320`](../../../web/frontend/src/theme.css#L1320)

**Regression proof**

- Start with newer-forward-failure and operational-precedence decision cases.
  [`strategyDecision.test.ts:89`](../../../web/frontend/src/screens/strategyDecision.test.ts#L89)

- Check collapsed word budgets and expanded-evidence reachability.
  [`word-budget.test.tsx:197`](../../../web/frontend/src/invariants/word-budget.test.tsx#L197)

- Finish with mobile overflow, date visibility and hit-area assertions.
  [`verify-viewport.mjs:47`](../../../web/frontend/scripts/verify-viewport.mjs#L47)
