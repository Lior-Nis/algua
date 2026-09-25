---
stepsCompleted: [1, 2, 3, 4, 5, 6]
status: ready
targetStory: 1.3a
inputDocuments:
  - docs/PRD.md
  - docs/architecture.md
  - docs/vision-reconciliation.md
  - docs/development/epics.md
  - docs/development/stories/1-1-extract-in-process-decision-planner.md
  - docs/development/stories/1-2-record-working-tree-deployments.md
  - docs/development/stories/1-3-materialize-and-execute-frozen-planner-artifacts.md
  - docs/development/stories/1-3a-complete-two-phase-planner-boundary-in-process.md
  - docs/development/stories/1-3b-materialize-and-verify-recoverable-planner-artifacts.md
  - docs/development/stories/1-3c-execute-frozen-planners-in-paper.md
  - docs/development/stories/1-3d-bind-operational-evidence-and-qualification.md
  - docs/development/stories/deferred-work.md
  - docs/development/specs/spec-story-1-3a-planner-contract/SPEC.md
  - docs/development/specs/spec-story-1-3a-planner-contract/planner-contract.md
  - docs/development/implementation-readiness-report-2026-09-25-story-1-3a.md
  - docs/development/sprint-change-proposal-2026-09-25.md
  - docs/superpowers/specs/2026-09-22-artifact-freeze-design.md
uxDocument: not-applicable-no-ui-scope
duplicateFormats: none
---

# Implementation Readiness Assessment Report

**Date:** 2026-09-25
**Project:** Algua — Story 1.3a contract-closure rerun

## Document Inventory

### Product and architecture authority

- `docs/PRD.md` — canonical whole-document product vision.
- `docs/architecture.md` — current whole-document architecture map.
- `docs/vision-reconciliation.md` — current-control and historical-document reconciliation.
- `docs/superpowers/specs/2026-09-22-artifact-freeze-design.md` — approved design direction.

### Delivery planning and target contract

- `docs/development/epics.md` — approved Phase 1 requirements and epic breakdown.
- Stories 1.1–1.2 — completed prerequisites.
- Story 1.3 — decomposed parent requirement record.
- Story 1.3a — amended target of this readiness rerun.
- Stories 1.3b–1.3d — downstream boundaries used for dependency validation.
- `SPEC.md` and `planner-contract.md` — newly adopted normative Story 1.3a machine contract.
- The initial Story 1.3a readiness report — evidence of the contract blocker this rerun must verify.
- The approved course-correction proposal and deferred-work record.

No whole-versus-sharded duplicate formats were found. No UX specification exists; this is expected
for an in-process planner contract behind the existing CLI/JSON surface. The older platform
architecture design remains historical context under the reconciliation record and is not a
competing canonical architecture. This dedicated rerun report preserves rather than overwrites the
initial assessment.

## PRD Analysis

### Functional Requirements

FR1: Algua must continuously discover market inefficiencies, formulate falsifiable hypotheses, test
them honestly, preserve learning, deploy validated strategies, combine weakly correlated edges,
operate safely, learn from failures, improve its codebase and scale compatible capital.

FR2: LLMs may support research, information extraction, coding, analysis, investigation, repair,
memory synthesis and prioritization, but AI components must prove incremental economic value against
credible simpler baselines.

FR3: One evolving repository must coexist with immutable executable artifacts entrusted with
capital; behavior changes create new artifacts rather than silently mutating evidence-bearing ones.

FR4: Strategy logic must remain a pure broker-agnostic mapping from point-in-time inputs to target
portfolio intent, with broker-specific behavior in execution adapters.

FR5: The same strategy implementation must travel through backtest, shadow, paper, experimental live
and scaled live rather than being reimplemented per environment.

FR6: Strategies may use deterministic, statistical, ML, deep-learning and LLM/agentic components;
behavior-affecting model changes require a new evaluated artifact.

FR7: Research must follow hypothesis, rationale, falsification criteria, experiment, robustness,
conclusion and learned knowledge while allowing new strategy families.

FR8: Every meaningful positive or negative experiment must become permanent structured knowledge
covering hypothesis, rationale, lineage, point-in-time data/config/code/model identity, assumptions,
metrics, robustness, artifacts, conclusion, failure and lessons.

FR9: LLM-dependent experiments must additionally preserve provider, model/version, prompt,
timestamp, inputs, raw response and structured output.

FR10: Agents must search prior experiments, explain materially equivalent reruns, maintain strategy
genealogy and synthesize accumulated findings and unexplored directions.

FR11: A result must be reproducible from code, artifact, data version, configuration and environment
before it can justify promotion.

FR12: Historical evaluation must be point-in-time correct and, where applicable, include OOS and
walk-forward testing, realistic costs/slippage, sensitivity, regimes, liquidity, capacity, simple
baselines, portfolio correlation and multiple-testing awareness.

FR13: Algua must own a canonical local versioned historical dataset, initially covering OHLCV,
fundamentals, timestamped news and timestamped filings; added sources require research justification.

FR14: The first complete market/timeframe scope must support liquid US stocks and ETFs at daily and
hourly horizons; added markets/horizons require economic justification and HFT is excluded.

FR15: Portfolio construction must consider return, uncertainty, volatility, drawdown, correlation,
liquidity, capacity and incremental contribution, initially using conservative risk budgets and
hard limits rather than aggressive Kelly sizing.

FR16: Monitoring must distinguish engineering failure from economic deterioration and apply
predetermined reduce/pause/retire/research rules; replacements follow the normal evidence pipeline.

FR17: Uncertainty such as stale data, discrepancies, impossible state, model failure, unexplained
positions, rejected orders or corrupt artifacts must prevent new exposure, trigger reconciliation
and evidence collection, and resume only after safe verification.

FR18: Every live order must be traceable through artifact, inputs, signal, sizing, risk decision,
intent, broker execution and resulting position.

FR19: Paper and shadow operation must remain permanent facilities for candidate/release validation,
discrepancy detection, repair testing and risk-free evidence accumulation.

FR20: Autonomous engineering must close the telemetry-to-verified-deployment loop through issue,
reproduction, regression test, root cause, minimal fix, independent review, gates, permitted merge,
paper/shadow validation and post-deployment verification.

FR21: Runtime correctness detection must cover crashes and invariant failures including
reconciliation, duplicate orders, stale inputs, accounting errors, impossible balances, position
drift and persistent latency.

FR22: Agents may investigate, test, implement bounded fixes, review/run CI, propose refactors,
merge/deploy allowlisted low-risk changes and reduce risk; large structural changes require human
approval and agents cannot expand their authority.

FR23: Human authorization is required for first real-money activation, higher live capital, new paid
commitments beyond budget and protected/high-impact safety or authority changes; agents may reduce
risk but never raise entrusted maximum capital.

FR24: Algua must operate safely without daily attendance and demonstrate 72 unattended hours of
correct operation or autonomous transition to a predefined safe state.

FR25: Initial capital policy must enforce the ₪10,000 allocation, including ₪2,000 experimental live
capital, ₪6,000 reserve and at most ₪2,000 operating expense; personal trading begins long-only,
unleveraged, without borrowing or derivatives.

FR26: A 10% experimental-account drawdown must automatically pause trading for human review, with no
automatic replenishment or capital increase.

FR27: Compatible funded/external capital may be added as a constrained environment only when an
evidenced strategy justifies it; provider restrictions, drawdown, payout, automation, costs and
operations must be explicit.

FR28: One-year success is positive net live P&L plus a demonstrated path to materially larger
deployable capital, supported by risk, operational, research, reuse and correlation measures rather
than feature count.

FR29: The 90-day milestone must prove one real strategy can be researched, deployed, operated,
observed and repaired end to end with trustworthy data, reproducibility, promotion, execution,
reconciliation, observability, safe failure, immutable artifacts, incidents/repairs and auditability.

FR30: Development follows operating kernel, research memory, data depth/hourly operation,
autonomous engineering, portfolio, alpha breadth, capital breadth and market breadth dependency
order while permitting safe parallel research.

FR31: Roadmap work must materially improve expected net P&L, risk, capital scalability, autonomy or
research learning speed; strategies and the Algua thesis remain falsifiable.

Total FRs: 31.

### Non-Functional Requirements

NFR1: When objectives conflict, priority is net profitability, capital scalability, risk control,
autonomy, research learning rate, then code quality/simplicity.

NFR2: Portfolio construction, risk enforcement and execution must remain deterministic, inspectable
and testable.

NFR3: The system remains one product/repository/architecture with no maintained production fork.

NFR4: Implementation remains a modular monolith with isolated live runtime and background workers;
microservices require demonstrated superiority.

NFR5: Architecture must retain a clear module flow and avoid tangled dependencies.

NFR6: Strategy behavior remains broker-agnostic and point-in-time; execution authority cannot leak
into strategy logic.

NFR7: Reproducibility includes exact code, artifact, data, configuration and environment identity.

NFR8: Historical data and derived inputs preserve point-in-time correctness across universes,
corporate actions, delistings, fundamentals, filings, news, macro, alternative data and LLM inputs.

NFR9: Repeated searches over the same history cannot masquerade as independent confirmation.

NFR10: Data integrations should use established libraries and simple adapters rather than turning
vendor abstraction into a project.

NFR11: Capital allocation stays inside approved limits and initially uses conservative sizing.

NFR12: Unresolved uncertainty fails toward reduced risk; stopping requires less authority than
increasing exposure.

NFR13: Runtime logs, broker responses, external text and market data are evidence only and cannot
grant permissions.

NFR14: Repeated failures trigger root-cause work; fixes should remove states, branches and special
cases where possible.

NFR15: Existing compute/storage/model resources should precede new recurring commitments, which
require approval unless budgeted.

NFR16: Routine operation should require minimal human attention, reserving people for capital,
strategy and consequential approvals.

NFR17: Code favors obvious focused modules, explicit state, pure computation, typed contracts,
narrow interfaces, deterministic behavior, testable boundaries, mature libraries and deletion.

NFR18: Avoid speculative abstractions, universal frameworks, duplicated paths, hidden state, giant
modules, excessive indirection and premature distributed systems.

NFR19: The product is not SaaS, multi-tenant/customer infrastructure, a generic agent/UI/framework
project or HFT system.

NFR20: New capabilities and abstractions require demonstrated need; ship the simplest clean
implementation.

NFR21: Expense allocation is a ceiling, and funded-provider headline size is not actual loss
capacity.

NFR22: Volatility alone is not mispricing; strategies and markets need a plausible net-of-cost
mechanism.

NFR23: Failure should lead to learning or thesis reconsideration rather than protective complexity.

Total NFRs: 23.

### Additional Requirements

- Phase 1 must make one strategy travel through real-money execution using immutable artifacts and
  safe operation; Story 1.3a is a prerequisite boundary, not the whole Phase 1 outcome.
- Vision-level autonomy is directional; current authenticated approvals and protected-change walls
  remain binding until explicitly changed and reviewed.
- Strategy intent must stay pure while deterministic timing, risk and execution remain inspectable.
- Story 1.3a must not pull artifact stores, subprocesses, schemas, model lanes, migration, live
  signing, hourly execution or capital changes into its in-process refactor.
- Negative results and incidents are durable knowledge; parity failures must be investigated rather
  than hidden by changing expectations.

### PRD Completeness Assessment

The PRD is complete and coherent as a North Star. It deliberately delegates exact runtime schemas,
normalization and current-control mechanics to architecture and delivery contracts. For Story 1.3a,
the governing product requirements remain especially FR3–FR5, FR11, FR17, FR19, FR29 and
NFR2–NFR7, NFR12, NFR17–NFR20. The new normative SPEC may refine how those requirements are met but
cannot change their authority, reproducibility or simplicity constraints.

## Epic Coverage Validation

The epic document has its own 14-item Phase 1 FR inventory. Those identifiers are scoped delivery
requirements rather than the 31 PRD-derived requirements above, so coverage is compared
semantically instead of matching numbers mechanically.

### Coverage Matrix

| PRD FR | Requirement area | Epic coverage | Status |
|---|---|---|---|
| FR1 | Autonomous trading-company loop | Epic 1 + 2 cover the operating kernel; later phases cover memory, engineering, portfolio and breadth | Partial by roadmap |
| FR2 | AI/LLM value | Phase 6; outside this cycle | Roadmap only |
| FR3 | Evolving repo plus immutable artifacts | Epic 1 scoped FR4, FR6, FR9–FR10; Stories 1.2–1.3d | Covered |
| FR4 | Pure broker-agnostic strategy intent | Epic 1 scoped FR2–FR3, FR6; Stories 1.1 and 1.3a | Covered |
| FR5 | One implementation across lifecycle | Epic 1 paper plus Epic 2 live scope | Covered |
| FR6 | Model capability/artifact replacement | Epic 1 scoped FR9 covers replacement; model breadth remains Phase 6 | Partial by roadmap |
| FR7 | Falsifiable research loop | Existing research system and later phases | Roadmap only |
| FR8 | Permanent experiment knowledge | Phase 2 | Roadmap only |
| FR9 | LLM experiment provenance | Phases 2 and 6 | Roadmap only |
| FR10 | Prior search/genealogy/synthesis | Phase 2 | Roadmap only |
| FR11 | Full reproducibility | Epic 1 scoped FR4–FR6, FR9; NFR1 | Covered |
| FR12 | PIT and robust evaluation | Scoped NFR2 and preserved research gates | Covered |
| FR13 | Canonical versioned data | Phase 3 | Roadmap only |
| FR14 | Daily/hourly markets | Daily now; hourly Phase 3, market breadth Phase 8 | Roadmap only |
| FR15 | Portfolio construction/allocation | Epic 2 current constraints; portfolio Phase 5 | Partial by roadmap |
| FR16 | Engineering/economic deterioration | Epic 1 strict repair/requalification; broader policy later | Partial by roadmap |
| FR17 | Fail safe under uncertainty | Epic 1 scoped FR10; Epic 2 scoped FR13–FR14; NFR4 | Covered |
| FR18 | Order traceability | Scoped FR12 across both epics | Covered |
| FR19 | Permanent paper/shadow | Epic 2 scoped FR14; Phase 1 validation flow | Covered |
| FR20 | Autonomous engineering loop | Phase 4 | Roadmap only |
| FR21 | Detect correctness failures | Epic 2 exercise scope; full detection Phase 4 | Partial by roadmap |
| FR22 | Engineering permissions | Scoped NFR4/NFR7 retain boundaries; Phase 4 broadens later | Partial by roadmap |
| FR23 | Human authority | Epic 2 scoped FR8, FR11, FR14 and owner decisions | Covered |
| FR24 | 72-hour unattended target | Epic 2 scoped FR13 | Covered |
| FR25 | Initial capital/instruments | Epic 2 scoped FR11 | Covered |
| FR26 | 10% pause/no automatic capital | Epic 2 scoped FR11 plus owner policy task | Covered |
| FR27 | Funded/external capital | Phase 7 | Roadmap only |
| FR28 | One-year economic success | Governs outcomes; not promised by this milestone | Roadmap outcome |
| FR29 | 90-day operating milestone | Epic 1 + Epic 2 | Covered |
| FR30 | Eight-phase sequence | Current epics implement Phase 1 and explicitly defer later phases | Covered as roadmap |
| FR31 | Admission test/falsifiability | Epic outcomes and non-promises align | Covered |

### Story 1.3a Traceability

Story 1.3a directly implements Epic 1 scoped FR2–FR3 and prepares scoped FR6. Its normative contract
keeps the strategy context pure, moves behavior-affecting inputs behind typed values, preserves
deterministic risk semantics and prevents the logical boundary from acquiring operational authority.
That is a necessary path to PRD FR3–FR5, FR11, FR17, FR19 and FR29 without claiming to complete
those larger outcomes alone.

### Missing Requirements

No Phase 1 requirement needed by Story 1.3a is missing from the epics. Eleven PRD-level capabilities
remain roadmap-only and six are partial because their remaining behavior deliberately belongs to
later phases. Those are portfolio-planning limitations, not a readiness defect in this in-process
boundary story.

No epic-only requirement lacks a PRD/design anchor. The scoped FR numbering differs from the
PRD-derived numbering, but the epic cites the governing PRD sections and outcome mapping explicitly.

### Coverage Statistics

- Total PRD-derived FRs: 31.
- Detailed current-epic coverage: 14.
- Partial current-epic coverage: 6.
- Roadmap-only coverage: 11.
- Phase 1 scoped FR inventory: 14 of 14 mapped to Epic 1 or Epic 2.
- Story 1.3a governing scoped requirements: fully mapped.

## UX Alignment Assessment

### UX Document Status

No UX document exists in the planning artifacts.

### Alignment Issues

None. The PRD names a UI product and unnecessary dashboards as non-goals. The architecture defines
the typed CLI with JSON stdout as the integration surface, and the epic explicitly adds no monitor
or UI scope. Story 1.3a and its normative SPEC change only an in-process planner contract behind
existing paper commands; they add no screen, visual flow or human interaction.

### Warnings

No missing-UX warning is warranted. The externally observable CLI must retain parseable JSON and
meaningful exit behavior; Epic NFR6, Story AC8 and the SPEC constraints preserve that contract.
Existing web code remains outside this story and is not a dependency.

## Epic Quality Review

### Epic Structure

Both approved epics describe operator outcomes rather than technical milestones. Epic 1 produces a
recoverable paper deployment whose evidence survives unrelated development. Epic 2 consumes that
same deployment for human-approved live operation inside the experimental-capital envelope. Epic 1
is useful alone, and Epic 2 does not depend on any later epic or rebuild the artifact.

### Story Sequence and Dependency Map

| Story | Dependency | Independently useful result | Assessment |
|---|---|---|---|
| 1.1 | Existing paper path | Pure brokerless weight/intent planner | Done; valid prerequisite |
| 1.2 | 1.1 | Explicit deployment epochs and exact-epoch evidence | Done; valid prerequisite |
| 1.3a | 1.1, 1.2 | Complete two-phase in-process behavior boundary with parity | Ready; no forward dependency |
| 1.3b | 1.3a | Read-only preparation/verification of recoverable content | Correct ordered dependency |
| 1.3c | 1.3a, 1.3b | Frozen paper dispatch with qualification deliberately blocked | Usable increment without 1.3d |
| 1.3d | 1.3a–1.3c | Bound evidence and qualification from immutable content | Correctly consumes prior results |

The parent Story 1.3 remains a traceability record rather than an implementation unit. No child
requires behavior first delivered by a later child. Story 1.3a creates no database entity;
deployment entities arrived when first needed in Story 1.2, and frozen-content persistence remains
in Story 1.3b. The sequence is correctly brownfield: existing integration points, compatibility
cohorts and later migration are explicit. A starter template is not applicable.

### Target Story Sizing and Acceptance Quality

Story 1.3a is one bounded refactor: implement pure typed values, two stateless in-process phases,
one current-supervisor adapter and parity tests. It excludes schema, filesystem, environment,
subprocess, deployment activation and authority work. Its operator value is concrete: the existing
paper path completes through the final immutable-execution seam without changing behavior.

The story's eight acceptance criteria use Given/When/Then semantics and cover happy paths, early
returns, risk failures, binding mismatch, compatibility and effect order. The adopted SPEC and
field-level companion now define:

- every common, Phase A, captured-late and Phase B field and result variant;
- null-versus-empty and ordered-versus-unordered semantics;
- an explicit captured exchange-calendar selector instead of the prior settings read;
- the logical bar/config component digests and canonical root preimage;
- exact separation between logical anti-mixing binding and future wire-byte evidence;
- static timeframe fail-fast behavior before acquisition plus immutable Phase A revalidation;
- the state machine and branch-by-branch parity matrix.

The initial readiness blocker is therefore resolved. Implementation no longer has to invent what
crosses either phase or how equivalent inputs bind.

### Review Corrections Applied During This Rerun

1. Raw bars now bind in exact captured order before Phase A performs the baseline stable index sort;
   production canonical-order/uniqueness validation remains intact.
2. Unsupported timeframes retain the supervisor's pre-acquisition fail-fast check and are rechecked
   by Phase A, preserving effect order without trusting mutable supervision alone.
3. `calendar_code` is now an explicit bound early input, allowing planner freshness logic to use the
   config-free calendar leaf instead of reading operational settings.

These were planning-contract corrections only. They did not change runtime code, schema, authority
or deployment state.

### 🔴 Critical Violations

None.

### 🟠 Major Issues

None. The prior incomplete machine-contract issue is closed by the normative SPEC and companion.

### 🟡 Minor Concerns

None blocking Story 1.3a. Epic 2 still requires detailed stories before its implementation begins,
but that is future planning and not a dependency of this target.

### Best-Practices Compliance

| Check | Epic 1 | Epic 2 | Story 1.3a |
|---|---:|---:|---:|
| Delivers operator value | Pass | Pass | Pass |
| Independent at its declared layer | Pass | Pass using Epic 1 | Pass |
| Appropriately sized | Pass | Pass at epic level | Pass |
| No forward dependency | Pass | Pass | Pass |
| Database entities introduced when needed | Pass | Pass at epic level | Not applicable / pass |
| Clear, fully testable acceptance criteria | Pass | Pass at epic level | Pass |
| Requirements traceability | Pass | Pass | Pass |

### Quality Gate Result

The epic structure and child sequence are sound. Story 1.3a is correctly sized, has complete
machine-contract and parity criteria, and is **ready for implementation** after its status and
sprint tracking are updated through the final assessment.

## Summary and Recommendations

### Overall Readiness Status

**READY.** Story 1.3a has complete product traceability, a bounded independent outcome, explicit
authority exclusions, exact logical Phase A/Phase B schemas, a deterministic binding preimage and a
branch-complete parity contract. It can proceed to implementation without waiting for Stories
1.3b–1.3d or inventing unresolved semantics.

### Critical Issues Requiring Immediate Action

None. The initial major contract blocker is closed. The three inconsistencies discovered during the
rerun—raw-bar ordering, pre-acquisition timeframe failure and implicit calendar selection—were
resolved in the normative contract before this verdict.

### Recommended Next Steps

1. Change Story 1.3a from `prepared-for-readiness-review` to `ready-for-dev` and initialize sprint
   tracking with only Story 1.3a ready; Stories 1.3b–1.3d remain backlog.
2. Execute Story 1.3a test-first from its story plus normative SPEC/companion. Begin with red tests
   for the field schemas, logical binding vectors, phase mismatch and parity matrix.
3. Run BMAD independent code review and the full sequential repository gate before marking the
   story done. No merge or deployment follows merely from readiness.

### Final Note

This rerun identified zero unresolved issues. It applied three planning-contract corrections in one
category and verified that none changes runtime code, schema, capital, deployment or authority.
Epic 2 still requires detailed stories before its own implementation begins, but that future work
does not block Story 1.3a.

**Assessed:** 2026-09-25 by the BMAD Implementation Readiness workflow (Codex).
