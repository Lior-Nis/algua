---
stepsCompleted: [1, 2, 3, 4, 5, 6]
status: needs-work
targetStory: 1.3a
inputDocuments:
  - docs/PRD.md
  - docs/architecture.md
  - docs/vision-reconciliation.md
  - docs/development/epics.md
  - docs/development/stories/1-3-materialize-and-execute-frozen-planner-artifacts.md
  - docs/development/stories/1-3a-complete-two-phase-planner-boundary-in-process.md
  - docs/development/stories/1-3b-materialize-and-verify-recoverable-planner-artifacts.md
  - docs/development/stories/1-3c-execute-frozen-planners-in-paper.md
  - docs/development/stories/1-3d-bind-operational-evidence-and-qualification.md
  - docs/development/stories/deferred-work.md
  - docs/development/implementation-readiness-report-2026-09-25.md
  - docs/development/sprint-change-proposal-2026-09-25.md
  - docs/superpowers/specs/2026-09-22-artifact-freeze-design.md
uxDocument: not-applicable-no-ui-scope
duplicateFormats: none
---

# Implementation Readiness Assessment Report

**Date:** 2026-09-25
**Project:** Algua — Story 1.3a

## Document Inventory

### Product and architecture authority

- `docs/PRD.md` — canonical whole-document product vision.
- `docs/architecture.md` — current whole-document architecture map.
- `docs/vision-reconciliation.md` — current-control and historical-document reconciliation.
- `docs/superpowers/specs/2026-09-22-artifact-freeze-design.md` — approved design direction.

### Delivery planning

- `docs/development/epics.md` — approved Phase 1 requirements and epic breakdown.
- Story 1.3 — decomposed parent requirement record.
- Story 1.3a — target of this readiness assessment.
- Stories 1.3b–1.3d — downstream child boundaries used for dependency/scope validation.
- Stories 1.1–1.2 — completed prerequisites.
- `docs/development/implementation-readiness-report-2026-09-25.md` — parent-story readiness evidence.
- `docs/development/sprint-change-proposal-2026-09-25.md` — approved decomposition decisions.
- `docs/development/stories/deferred-work.md` — explicitly deferred findings.

No whole-versus-sharded duplicate formats were found. No UX specification exists; this is expected
for an in-process planner-contract slice whose external interface remains the existing CLI/JSON
surface. The earlier same-date parent-story report is preserved; this dedicated file assesses only
Story 1.3a.

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
liquidity, capacity and incremental contribution, initially using conservative risk budgets and hard
limits rather than aggressive Kelly sizing.

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

The PRD is complete and coherent as a North Star. It intentionally delegates process protocols,
module placement and current-control mechanics to architecture and delivery artifacts. For Story
1.3a, the governing PRD requirements are especially FR3–FR5, FR11, FR17, FR19, FR29 and NFR2–NFR7,
NFR12, NFR17–NFR20. Detailed parity, phase-binding and authority exclusions must therefore be
validated against the epic, parent design and child story rather than inferred from the PRD alone.

## Epic Coverage Validation

The epic document uses a scoped FR1–FR14 inventory for the Phase 1 operating kernel; those numbers
are not the same taxonomy as the 31 PRD-derived requirements above. Coverage is therefore compared
semantically.

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

Story 1.3a directly implements Epic 1 scoped FR2–FR3 and prepares scoped FR6. Its parity and
authority rules enforce scoped NFR2–NFR6. Its complete behavior boundary is also a necessary path to
PRD FR3–FR5, FR11, FR17, FR19 and FR29 without claiming to complete those larger outcomes alone.

### Missing Requirements

No Phase 1 requirement needed by Story 1.3a is missing from the epic. Eleven PRD-level capabilities
remain roadmap-only and six remain partial because their remaining behavior deliberately belongs to
later phases. Those are portfolio-planning limitations, not a readiness defect in this in-process
boundary story.

No epic-only requirement lacks a PRD/design anchor. The scoped FR numbering could be mistaken for
the PRD-derived numbering, but the document explicitly cites its PRD sections and delivery outcomes;
this assessment preserves semantic rather than numeric traceability.

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

None. The PRD explicitly lists a UI product and unnecessary dashboards as non-goals. The current
architecture defines the typed CLI with JSON stdout as the product command surface, and the epic
explicitly states that this cycle adds no monitor/UI scope. Story 1.3a changes only an in-process
planner contract behind existing paper commands; it adds no screen, visual flow or human interaction.

### Warnings

No missing-UX warning is warranted. CLI behavior remains externally observable, so Story 1.3a must
preserve parseable JSON and meaningful exit behavior; that requirement is already captured by Epic
NFR6 and Story AC8. Existing web code is outside this story and is not an implied dependency.

## Epic Quality Review

### Epic Structure

Both approved epics are outcome-oriented rather than infrastructure milestones:

- Epic 1 lets an operator accumulate trustworthy paper evidence against recoverable immutable
  behavior while unrelated repository development continues.
- Epic 2 lets the operator take that same evidence-bearing deployment through human-approved live
  operation inside the experimental-capital envelope.

Epic 1 is useful without Epic 2. Epic 2 consumes Epic 1's deployment rather than depending on a
later epic or rebuilding the artifact. The sequencing therefore satisfies epic independence. The
absence of detailed Epic 2 stories is a future planning requirement, not a dependency defect for
Story 1.3a.

### Story Sequence and Dependency Map

| Story | Dependency | Independently useful result | Assessment |
|---|---|---|---|
| 1.1 | Existing paper path | Pure brokerless weight/intent planner | Done; valid prerequisite |
| 1.2 | 1.1 | Explicit deployment epochs and exact-epoch evidence | Done; valid prerequisite |
| 1.3a | 1.1, 1.2 | Complete two-phase in-process behavior boundary with parity | No forward dependency |
| 1.3b | 1.3a | Read-only preparation/verification of recoverable content | Ordered dependency; no activation claim |
| 1.3c | 1.3a, 1.3b | Frozen paper dispatch with qualification deliberately blocked | Usable execution increment; no dependency on 1.3d for its stated result |
| 1.3d | 1.3a–1.3c | Bound evidence and qualification from verified immutable content | Correctly consumes prior results |

The decomposed parent Story 1.3 is a traceability record, not an implementation unit. No child
requires functionality first delivered by a later child. Story 1.3a makes no database or filesystem
change, so entity creation timing is appropriate: deployment entities already arrived with Story
1.2, while frozen-content fields remain in Story 1.3b. This is a brownfield sequence with explicit
compatibility and migration boundaries; a starter-template story is not applicable.

### Target Story Sizing and Acceptance Quality

Story 1.3a is now a focused refactor: pure typed contracts, two stateless in-process phases, one
existing supervisor integration and parity tests. It avoids the schema, filesystem, environment,
subprocess, activation and authority work that made the parent story epic-sized. Its operator value
is enabling but concrete: the existing paper path completes through the new boundary with unchanged
behavior, leaving a separately reviewable seam for isolation.

The eight acceptance criteria use Given/When/Then semantics, cover normal paths, early returns,
risk breaches, compatibility and effect ordering, and require observable parity rather than an
unsupported assertion. The scope and dependency direction are clear. One machine-contract gap,
however, still prevents deterministic implementation.

### 🔴 Critical Violations

None. The target is no longer epic-sized, has no circular or forward dependency, and does not pull a
future authority or storage capability into the current story.

### 🟠 Major Issues

1. **The Phase A/Phase B contract and its binding are not closed enough to implement without
   inventing semantics.** AC1 asks for strategy/deployment/request identity, resolved bounds and
   “every value needed”; AC4 asks for captured sizing/NAV, venue belief and per-strategy state; AC5
   describes the result only by examples. Neither the story nor its approved proposal enumerates the
   exact fields, types, optional/empty distinctions and result variants that cross each phase.
   More importantly, AC3 requires SHA-256 over a “canonical request identity” and “exact early-input
   identity” while raw pandas bars are part of that input, but it does not define the canonical byte
   representation or say whether the binding covers values directly or named component digests.
   Two conforming implementations could therefore produce different bindings, omit different
   behavior-affecting values, or pre-commit a representation that conflicts with Story 1.3c's later
   Parquet/JSON protocol. Add an explicit logical schema for both requests and all result variants,
   plus the versioned canonical binding preimage and normalization rules. The in-process schema may
   remain transport-independent, but the digest must be stable across the two future fresh-process
   invocations and must bind every behavior-affecting input exactly once.

### 🟡 Minor Concerns

- Several acceptance criteria intentionally bundle related parity assertions. The implementation
  plan should map each branch/result variant to a focused test so one broad golden-master assertion
  cannot hide an uncovered path.
- “Cryptographic phase binding” could be read as authorization. This digest is an integrity and
  anti-mixing binding only; the existing scope correctly grants it no authority. Rename or state
  that distinction when the contract is specified.

### Best-Practices Compliance

| Check | Epic 1 | Epic 2 | Story 1.3a |
|---|---:|---:|---:|
| Delivers operator value | Pass | Pass | Pass |
| Independent at its declared layer | Pass | Pass using Epic 1 | Pass |
| Appropriately sized | Pass | Pass at epic level | Pass |
| No forward dependency | Pass | Pass | Pass |
| Database entities introduced when needed | Pass | Pass at epic level | Not applicable / pass |
| Clear, fully testable acceptance criteria | Pass | Pass at epic level | **Partial: contract schema/binding gap** |
| Requirements traceability | Pass | Pass | Pass |

### Quality Gate Result

The epic and child-story decomposition is structurally sound, and Story 1.3a is correctly sized and
ordered. Story 1.3a is **not yet implementation-ready** because its central phase boundary and digest
preimage remain underspecified. This is a bounded story-definition correction, not another epic
decomposition or architecture change.

## Summary and Recommendations

### Overall Readiness Status

**NEEDS WORK.** The Phase 1 requirements, architecture, two-epic structure and 1.3a–1.3d dependency
sequence are aligned. Story 1.3a is correctly scoped and independently completable, but development
should not begin until its central logical contract is explicit enough to admit one deterministic
implementation and review standard.

### Critical Issues Requiring Immediate Action

No critical-severity architecture, authority or dependency violation was found.

One major implementation blocker requires correction: enumerate the exact Phase A and Phase B
request/result fields and variants, then define the versioned canonical normalization and SHA-256
preimage used to bind the phases. The definition must preserve null-versus-empty distinctions, cover
all behavior-affecting values and remain stable when Story 1.3c moves each phase into a fresh child
process.

### Recommended Next Steps

1. Amend Story 1.3a with a contract table for the common envelope, Phase A result union, Phase B
   late-input envelope and Phase B result union, including field types, optionality and invariants.
2. Define a transport-independent canonical binding format: domain/version separator, ordered
   component names, normalization for timestamps/floats/collections/bars and the exact Phase A
   outcome fields included in the digest. State explicitly that it is integrity evidence, not
   authorization.
3. Map every result variant and parity branch to focused red tests, rerun this targeted readiness
   check, and only then change Story 1.3a to `ready-for-dev` and initialize sprint status.

### Final Note

This assessment identified three issues in epic/story quality: one major contract blocker and two
minor test/wording concerns. No missing Phase 1 requirement, UX dependency, forward dependency,
authority expansion or story-sizing defect remains. The correction is local to Story 1.3a and does
not require changing the canonical PRD, architecture, approved parent outcome or decomposition.

**Assessed:** 2026-09-25 by the BMAD Implementation Readiness workflow (Codex).
