---
baseline_commit: dc2a222ef811dc1c3a835d656a411de58423796c
---

# Story 1.3a: Complete the two-phase planner boundary in-process

Status: ready-for-dev

Prepared: 2026-09-25. Baseline: Story 1.3 readiness baseline `dc2a222` (PR #671).
Epic: 1. Parent: Story 1.3. Requirements: FR2–FR3, FR6, FR9–FR10 and NFR1–NFR6.
Depends on: Stories 1.1 and 1.2 (`done`).

## Story

As Algua's operator,
I want the complete behavior-affecting per-strategy decision path represented by a stateless
two-phase planner contract while it still runs in-process,
so that later process isolation can preserve paper behavior without moving broker or registry
authority into the planner.

## Scope and authority

Story 1.1 extracted pure weight/intent computation. This story completes the future frozen boundary
around it. The planner receives captured values and owns the behavior-affecting timing, freshness,
warm-up, held-symbol valuation, gate-bound universe, decision timestamp and per-strategy risk
semantics that determine whether or what to trade.

The current supervisor continues to own and perform provider/broker calls, registry access,
reconciliation data acquisition, account/book controls, cancellation, submission, buying-power
reservation, hooks, audit and persistence. The planner may evaluate captured reconciliation and
per-strategy risk state; it may not acquire that state or perform effects.

This story is in-process only. It adds no artifact files, environment provisioning, subprocess,
schema migration, deployment activation, stage change, live authority or capital permission. It
does not reinterpret the currently stamped planner protocol. The frozen wire protocol receives a
new version only when Story 1.3c introduces it.

## Normative planner contract

The [Story 1.3a machine contract](../specs/spec-story-1-3a-planner-contract/SPEC.md) and its
[field-level companion](../specs/spec-story-1-3a-planner-contract/planner-contract.md) are normative
for this story. Implementers and reviewers must read both. They close the logical contract without
selecting Story 1.3c's Parquet/JSON byte encoding.

| Value | Exact role |
|---|---|
| `EarlyPlannerInput` | verified request/deployment/config identity, explicit clock/timeframe/calendar, raw bars, early positions, gate universe and drawdown bound |
| Phase A result | `EarlyNoDecision`, `PlannerRiskFailure`, `PlannerInputFailure` or `SnapshotRequired` |
| `CapturedStrategyState` | sizing and drawdown equity, quantities, market values, persisted peak and tagged disabled/enabled venue belief |
| Phase B result | `PhaseBindingFailure`, `PlannerRiskFailure`, `PlannerInputFailure`, `LateNoDecision` or `Decision` |

The full SHA-256 Phase A binding is defined over a versioned canonical logical preimage. Bars and
resolved configuration enter through named full-digest components; mappings, timestamps, floats,
nulls and tagged unions have explicit normalization. This binding prevents phase mixing only. It is
not a signature, approval or authority token, and later exact wire-byte hashes remain separate
evidence.

## Acceptance criteria

1. **Typed common envelope.** Given one paper tick, when the supervisor prepares `EarlyPlannerInput`,
   then it contains exactly the identity, clock, timeframe, raw-bar, early-position, gate-universe,
   resolved-configuration and drawdown-bound fields in the normative contract. The pure strategy
   implementation is verified against that identity as execution context rather than serialized as
   a callable. Neither value contains registry, provider, broker, hook, connection, credential,
   persistence or callback authority.
2. **Phase A owns early behavior.** Given the early envelope, when Phase A runs, then it returns a
   typed early no-decision/error result or `snapshot_required`. Closed-bar timing, held/universe
   filtering, warm-up and freshness outcomes exactly match the baseline path. The supervisor does
   not acquire late sizing/account values unless Phase A returns `snapshot_required`. The supervisor
   retains the baseline static timeframe fail-fast check before any input acquisition, and Phase A
   revalidates the captured timeframe as defense in depth.
3. **Integrity phase binding.** Given `SnapshotRequired`, then Phase A returns the full SHA-256
   binding produced by the normative domain/version, component digests and canonical normalization.
   Equivalent logical inputs under the same request identity bind identically; any behavior-affecting
   identity, configuration, bar, position, bound or Phase A outcome change binds differently. This
   digest has no authorization semantics and is distinct from future hashes of exact wire bytes.
4. **Stateless Phase B.** Given the exact `CapturedStrategyState`, when Phase B runs, then it receives
   the original early input, captured late value and supplied Phase A binding, recomputes Phase A and
   returns `PhaseBindingFailure` before inspecting late risk or invoking decision code if the binding
   or outcome differs. No mutable process-local state connects the phases. Tagged venue belief
   preserves disabled versus enabled-empty semantics.
5. **Complete decision boundary.** Given a valid Phase B request, when it completes, then it returns
   exactly one normative result variant with the same no-decision state, typed risk breach or
   decision as the baseline. Derived equity/peak/reconciliation/realized-gross state, decision
   timestamp, target-weight order and ordered intents match the normative result schema; broker
   submission fields never cross into the planner result.
6. **Supervisor effect order is unchanged.** Given normal, early-return and breach paths, when the
   existing paper command runs through the two-phase in-process API, then provider/broker acquisition,
   cancellation, submission, hooks, audit and tick effects occur in the same permitted order as the
   Story 1.1 baseline. Prohibited effects remain absent.
7. **Parity is demonstrated, not asserted.** Golden-master tests cover warm-up, held-symbol
   valuation, dropped universe members, stale/missing/non-finite marks, invalid equity, drawdown,
   reconciliation and realized-gross breaches, empty/no-op decisions, fractional rules, overlays,
   capacity, gross utilization, ordered intents and next-bar behavior. Results, domain errors and
   effect traces are identical to the baseline.
8. **Compatibility and walls remain intact.** Working-tree and legacy tenants continue through the
   existing supervisor, deployment/epoch verification is unchanged, `t -> t+1` remains enforced,
   CLI output remains JSON, import contracts/module-size pins are not weakened and the full root
   quality gate passes.

## Tasks / subtasks

- [ ] Characterize the complete pre-cancel path with red golden-master tests (AC2, AC5–AC7).
- [ ] Implement the exact immutable Phase A/Phase B inputs and result unions from the normative
  machine contract in a pure contract location (AC1–AC5).
- [ ] Implement the normative component digests, normalization and Phase A logical binding without
  authority-bearing values or wire-format coupling (AC3–AC4).
- [ ] Move early timing/warm-up/freshness evaluation behind Phase A (AC2).
- [ ] Move captured per-strategy risk and decision evaluation behind Phase B (AC4–AC5).
- [ ] Route the existing in-process paper path through both phases and preserve effect order (AC6).
- [ ] Add structural tests proving planner inputs cannot carry registry/provider/broker/hooks (AC1).
- [ ] Run independent review against this story before changing its status (AC7–AC8).

## Development notes

- Start from `algua/live/live_loop.py`, `algua/live/planner.py` and the Story 1.1 parity tests.
- Prefer pure frozen dataclasses/unions and explicit canonicalization. Do not add I/O to
  `algua/contracts` or side effects to the planner.
- Preserve the distinction between the union bar frame used to value held positions and the
  gate-bound universe used for decisions.
- Capture the configured exchange calendar code explicitly. Planner code may construct the pure
  `MarketCalendar` leaf from that value; it may not call the settings-backed calendar factory.
- Bind raw bars in their exact captured row order, then perform the baseline stable index sort inside
  Phase A; production provider schema validation still requires canonical order and uniqueness.
- Phase B recomputation is deliberate preparation for two fresh child processes. Do not optimize it
  away with a hidden cache or object identity.
- `live_loop.py` is size-pinned. Extract focused modules rather than increasing the ratchet.
- A parity failure is a design decision or bug to investigate, not a test expectation to update.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

## References

- [Parent Story 1.3](1-3-materialize-and-execute-frozen-planner-artifacts.md)
- [Normative Story 1.3a machine contract](../specs/spec-story-1-3a-planner-contract/SPEC.md)
- [Normative field and binding contract](../specs/spec-story-1-3a-planner-contract/planner-contract.md)
- [Approved Sprint Change Proposal](../sprint-change-proposal-2026-09-25.md)
- [Parent Implementation Readiness Report](../implementation-readiness-report-2026-09-25.md)
- [Story 1.3a READY rerun](../implementation-readiness-report-2026-09-25-story-1-3a-rerun.md)
- [Artifact-freeze design](../../superpowers/specs/2026-09-22-artifact-freeze-design.md)
