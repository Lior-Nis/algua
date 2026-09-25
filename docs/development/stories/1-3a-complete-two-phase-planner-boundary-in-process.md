---
baseline_commit: dc2a222ef811dc1c3a835d656a411de58423796c
---

# Story 1.3a: Complete the two-phase planner boundary in-process

Status: prepared-for-readiness-review

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

## Acceptance criteria

1. **Typed common envelope.** Given one paper tick, when the supervisor prepares the early planner
   input, then the value contains explicit strategy/deployment/request identity, `now`, timeframe,
   raw fetched bars, early held positions, gate-bound universe, resolved bounds/configuration and
   every value needed to reproduce closed-bar selection and mark-freshness behavior. It contains no
   registry, provider, broker, hook, connection or callback authority.
2. **Phase A owns early behavior.** Given the early envelope, when Phase A runs, then it returns a
   typed early no-decision/error result or `snapshot_required`. Closed-bar timing, held/universe
   filtering, warm-up and freshness outcomes exactly match the baseline path. The supervisor does
   not acquire late sizing/account values unless Phase A returns `snapshot_required`.
3. **Cryptographic phase binding.** Given `snapshot_required`, then Phase A returns a deterministic
   SHA-256 binding over the canonical request identity, exact early-input identity and Phase A
   outcome. Equivalent inputs bind identically; any behavior-affecting change binds differently.
4. **Stateless Phase B.** Given captured sizing/NAV, venue-belief and per-strategy state, when Phase
   B runs, then it receives the original early envelope, captured late values and Phase A binding,
   recomputes Phase A and fails before decision computation if the binding or outcome differs. No
   mutable process-local state connects the phases; `None` and an empty venue belief remain distinct.
5. **Complete decision boundary.** Given a valid Phase B request, when it completes, then the typed
   result contains the same no-decision result, risk breach or decision as the baseline, including
   equity/peak/drawdown, captured reconciliation result, realized-gross state, decision timestamp,
   target weights and ordered intents.
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
- [ ] Define immutable Phase A/Phase B inputs and result unions in a pure contract location (AC1–AC5).
- [ ] Implement canonical request/Phase A binding without authority-bearing values (AC3–AC4).
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
- [Approved Sprint Change Proposal](../sprint-change-proposal-2026-09-25.md)
- [Implementation Readiness Report](../implementation-readiness-report-2026-09-25.md)
- [Artifact-freeze design](../../superpowers/specs/2026-09-22-artifact-freeze-design.md)

