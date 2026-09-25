---
id: SPEC-story-1-3a-planner-contract
companions:
  - planner-contract.md
  - ../../stories/1-3a-complete-two-phase-planner-boundary-in-process.md
sources: []
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete,
> preservation-validated contract for what to build, test and validate. Source documents remain
> narrative and traceability evidence; they do not override this contract.

# Story 1.3a Two-Phase Planner Contract

## Why

Algua must move all behavior-affecting per-strategy planning behind a stateless boundary before it
can execute that behavior from immutable artifacts. Story 1.3a must close that boundary while it is
still in-process, preserving today's paper behavior and keeping broker, registry and effect
authority in the current supervisor.

## Capabilities

- id: CAP-1
  intent: The supervisor can submit one complete early-input value and receive either an early
    terminal outcome or a request for captured late state.
  success: Every baseline pre-snapshot outcome and error is reproduced without acquiring late
    state when the early result is terminal.
- id: CAP-2
  intent: Phase A can bind a snapshot-required result to the exact logical request and early input
    without retaining process-local state.
  success: Equivalent logical inputs produce the same SHA-256 binding; any behavior-affecting
    identity, configuration, bar, position, bound or Phase A outcome change produces a different
    binding.
- id: CAP-3
  intent: Phase B can recompute Phase A, verify its binding and evaluate captured per-strategy risk
    and decision state.
  success: A mismatch terminates before decision computation; a match yields the same no-decision,
    risk-breach or decision outcome as the baseline.
- id: CAP-4
  intent: The current paper supervisor can use the two-phase API without changing acquisition,
    cancellation, submission, hooks, audit or persistence ordering.
  success: Golden-master tests show identical results, domain failures and effect traces across all
    enumerated parity branches.
- id: CAP-5
  intent: The logical contract can cross two future fresh-process invocations without redesign.
  success: Story 1.3c can encode and decode the values while preserving this logical binding; wire
    byte hashes remain separate evidence and do not redefine the binding.

## Constraints

- The exact schemas, variants, normalization and state transition rules in `planner-contract.md`
  are normative.
- The planner receives values and pure strategy behavior only; it receives no provider, broker,
  registry, connection, hook, credential, callback or persistence authority.
- Phase B receives the original early input and recomputes Phase A. No cache, object identity,
  closure or mutable singleton may connect the phases.
- The binding is integrity and anti-mixing evidence, not authorization, authenticity or approval.
- The current `t -> t+1` rule, risk walls, deployment verification, CLI JSON contract, import
  boundaries and module-size pins remain unchanged.
- Story 1.3a adds no filesystem artifact, IPC encoding, subprocess, schema migration, deployment
  activation, stage transition, live authority or capital permission.

## Non-goals

- Defining Parquet/JSON wire bytes, child-process limits or subprocess lifecycle.
- Materializing or activating frozen artifacts and environments.
- Migrating working-tree or fixed legacy tenants.
- Changing risk policy, strategy semantics, live signing or capital authority.
- Treating the planner boundary as a hostile-code sandbox.

## Success signal

The existing paper path completes through two stateless in-process phases, every named baseline
branch has parity coverage, Phase B rejects altered early or late bindings before decision work, and
the full repository gate passes without authority or compatibility changes.
