---
baseline_commit: 216c8ecc5d6b857f1f173060bdba4ec1eb300962
---

# Story 1.3c: Execute frozen planners in paper

Status: review

Prepared: 2026-09-25. Readiness: 2026-09-30. Baseline: Story 1.3b merge `216c8ec` (PR #676).
Epic: 1. Parent: Story 1.3. Requirements: FR4, FR6, FR9–FR10 and NFR1–NFR8.
Depends on: Stories 1.3a and 1.3b reviewed and merged.

## Story

As Algua's operator,
I want newly admitted paper strategies to execute their planner from verified immutable content,
so that unrelated repository development cannot change the behavior earning paper evidence.

## Scope and authority

This story activates frozen execution for eligible new admissions and dispatches each required
planner phase in a fresh short-lived subprocess. Existing `working_tree` deployments and the fixed
legacy cohort retain their compatibility paths until controlled migration.

The current supervisor remains singular and owns shared ingest, registry, broker/account state,
reconciliation acquisition, book risk, buying power, cancellation, submission, hooks, audit and
tick persistence. The subprocess boundary provides reproducibility and capability hygiene, not a
malicious-code sandbox.

Frozen promotion remains blocked with `frozen_qualification_pending` until Story 1.3d binds and
verifies its operational evidence. Live signing and authority are unchanged.

## Normative contract

The [Story 1.3c machine contract](../specs/spec-story-1-3c-frozen-paper-execution/SPEC.md) and its
[field-level companion](../specs/spec-story-1-3c-frozen-paper-execution/frozen-execution-contract.md)
are normative. They settle admission, routing, the supervisor view, phase dispatch, wire bytes,
launch, limits, result validation, failure classification, promotion refusal and module placement.
Implementers and reviewers must read both.

## Frozen-wire protocol-v1 protected limits

`frozen-planner` wire version 1 is its own named protocol namespace recorded in the canonical
frozen manifest. It does not reinterpret the existing integer protocol stamp on working-tree
descriptors. The exact persisted representation must preserve both identities without changing old
rows.

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

These values are protected code constants. Raising one requires protected review and a new frozen
wire protocol version. A future operational setting may only tighten a bound.

## Acceptance criteria

1. **Frozen intake is atomic.** Given an eligible qualified candidate and verified prepared content,
   when paper intake succeeds, then it records `source_kind="frozen"` and atomically creates the
   deployment/allocation/stage transition using the exact descriptor. Preparation happens before
   the write transaction. Failure creates no active deployment, allocation or stage change.
2. **Fresh stateless phases.** Phase A runs in one fresh process. Only after a validated
   `snapshot_required` may the supervisor acquire late values; Phase B then runs in a second fresh
   process with the same early inputs, captured late values and Phase A binding. No child remains
   alive while authority-bearing values are acquired.
3. **Canonical bounded transport.** Raw bars cross as Parquet/Arrow with explicit index preservation
   and exact schema validation. Metadata crosses as bounded canonical JSON. Both sides validate UTC
   index, names, types, order, uniqueness, null/empty rules and all request identities. Pickle/object
   payloads are forbidden and oversize inputs fail before launch.
4. **Hygienic process launch.** Dispatch uses the verified environment's absolute interpreter,
   fixed argv, `shell=False`, no stdin, closed descriptors, a fresh process group and compatible
   isolated/no-user-site flags. `cwd` and import roots resolve only the frozen bundle. A replacement
   environment removes credentials, operational paths, proxy/cloud values, loader injection,
   `PYTHON*`, `ALGUA_*`, `ALPACA_*`, active-venv and uv state.
5. **Read-only invocation.** Each private invocation directory contains bounded input only and is
   sealed read-only before launch. No child output file is accepted; stdout is the only result
   channel. Tick execution performs no dependency resolution, download or `uv` call.
6. **Strict result validation.** Stdout is exactly one bounded UTF-8 JSON document. Reject duplicate
   keys, trailing data, excessive nesting/collections, unknown/missing fields, booleans as numbers,
   NaN/Infinity, identity/timestamp mismatches, duplicate/out-of-universe symbols, invalid sides and
   inconsistent weights/intents. The supervisor reruns current decision/intent validation before
   effects.
7. **Bounded teardown and diagnostics.** Timeout or abnormal exit terminates and reaps the process
   group using the approved grace. Raw stderr is never persisted or treated as instruction. At most
   the approved sanitized diagnostic plus structured exit/signal/truncation metadata is retained.
8. **Tenant failure is safe.** Missing/corrupt/unsupported content, environment mismatch, protocol
   failure, timeout/signal/nonzero exit, excessive output or invalid result produces zero cancel,
   submit, downstream-hook and successful-tick effects for that strategy and a stable
   deployment-bound failure. `trade-tick` exits nonzero; `run-all` continues valid siblings unless
   existing semantics classify the fault as systemic.
9. **Supervisor remains current and singular.** One current supervisor performs shared ingest,
   snapshot refresh, reconciliation acquisition, account/book/buying-power sequencing and effects.
   The child cannot open/migrate the registry, construct provider/broker clients, reserve capital,
   invoke hooks or persist operational state.
10. **Frozen parity and independence.** Fresh-process results and effect traces match Story 1.3a
    across all representative normal/early/breach fixtures. Working-tree edits after activation do
    not change results; missing content is never rebuilt from current `HEAD`; restart resolves the
    same stored artifact without a Git checkout.
11. **Qualification is fail-closed.** Until Story 1.3d lands, `paper promote` rejects every frozen
    deployment with `frozen_qualification_pending`. Working-tree/legacy behavior is not silently
    changed, and no live authority is inferred from frozen execution.
12. **Repository contracts remain green.** All CLI success/errors remain JSON, import boundaries and
    module-size pins remain intact, protected surfaces receive review and the full root gate passes.

## Tasks / subtasks

- [x] Add the process-containment primitive with timeout, grace, group kill, reap and bounded
  stdout/stderr, test-first (AC4, AC7).
- [x] Add the frozen-wire codec: request/response schemas, float/timestamp encodings, Arrow IPC bars,
  limits and every rejection, with round-trip and digest tests (AC3, AC6).
- [x] Build the child entry point and bootstrap: bundle-only `algua`, protocol/identity checks,
  phase execution, stdout-only result (AC2, AC4–AC6, AC9).
- [x] Build the supervisor dispatcher: sealed invocation directory, launch, teardown, strict result
  validation, sanitization, tenant-failure mapping (AC2–AC8).
- [x] Route ticks by `source_kind`, build the frozen supervisor view and per-process offline
  verification cache, and refuse frozen deployments in the live lane (AC8–AC10).
- [x] Switch intake to frozen admission with prepare-then-verify before the write transaction
  (AC1).
- [x] Add the `frozen_qualification_pending` promotion refusal and register all new codes (AC11).
- [x] Prove parity (normal/early/breach, effect traces), determinism, restart and mutable-checkout
  independence, and per-code tenant isolation in `trade-tick` and `run-all` (AC8–AC10).
- [x] Protect the new modules (CODEOWNERS, hygiene set, import-linter contract), keep size pins, run
  the full root gate and obtain independent review before merge (AC12).

### Review Findings

Independent review round 1 (2026-09-30): Codex was out of quota, so the story-delivery fallback ran
BMAD Blind Hunter + Edge Case Hunter over three bounded file groups (wire/child/primitive; port and
planner; registry and CLI), an Acceptance Auditor over the whole story, and an opt-in test against a
real provisioned environment. Every finding was either patched or accepted as a recorded residual.

- [x] [Review][Patch] The supervisor trusted the kind of result a child chose, so current drawdown,
  reconcile, realized-gross and mark walls could be skipped or a false dark-feed breach reported;
  strategy-free walls are now shared functions the supervisor re-derives authoritatively
  [algua/live/frozen_dispatch.py, planner_early.py, planner_late.py, planner_validation.py]
- [x] [Review][Patch] A printing strategy corrupted the child's stdout result channel
  [algua/live/frozen_child.py]
- [x] [Review][Patch] Whole numbers in float config fields made the strict decoder refuse a
  qualified candidate forever [algua/contracts/types.py, algua/contracts/float_fields.py]
- [x] [Review][Patch] Raw `registry transition --to forward_tested` and the go-live path bypassed
  the frozen promotion refusal [algua/registry/transitions.py, algua/cli/registry_cmd.py]
- [x] [Review][Patch] Story 1.3b environment digests were not reproducible (uv writes the staging
  path into `bin/activate.csh`), so every admission published a new ~1 GB environment
  [algua/registry/planner_environment.py]
- [x] [Review][Patch] Orphaned `__pycache__` bytecode (including three cache-only directories on the
  main checkout) blocked every frozen intake as `frozen_source_drift` [algua/registry/frozen_source.py]
- [x] [Review][Patch] float32 strategy weights at a cap breached only on the supervisor's rerun
  [algua/live/planner_decision.py]
- [x] [Review][Patch] Supervisor-side disk/descriptor errors were misreported as the tenant's
  `frozen_launch_failed`; an unreadable entry module escaped as a raw error
  [algua/live/frozen_dispatch.py, frozen_invocation.py]
- [x] [Review][Patch] Compressed or multi-batch `bars.arrow` files were accepted
  [algua/live/frozen_wire_arrow.py]
- [x] [Review][Patch] Failure vocabularies are append-only per wire version so older bundles keep
  decoding [tests/test_failure_vocabularies.py]
- [x] [Review][Patch] Decomposed-Unicode configs are refused at preparation, before publication
  [algua/registry/artifact_preparation.py]
- [x] [Review][Patch] Breach text, reconcile text and realized-gross summation made identical across
  frozen and in-process execution; a drawdown breach no longer reads the venue belief first
- [x] [Review][Patch] The §10 opt-in real prepare → verify → dispatch test
  [tests/test_frozen_real_environment.py]
- [x] [Review][Patch] Acceptance gaps: `frozen_planner_rejected` in the CLI isolation matrix, a
  direct no-uv/Git-at-tick test, deployment id on trade-tick refusals, architecture map
- [x] [Review][Defer] Bars digests are recomputed per phase; at ~80% of the 256 MiB bound both
  phases would time out (today's bar sets are 10–25 MiB) — latent, recorded
- [x] [Review][Defer] A breach message longer than 8 KiB (e.g. listing a ~1,000-symbol universe) is
  truncated on the frozen side only — text only
- [x] [Review][Defer] Intake re-provisions the environment for candidates it then refuses on
  capital (uv cache is private per build); a hung shared environment can cost 124 s per frozen
  tenant per cycle — cost and scheduling, recorded
- [x] [Review][Defer] Tests write fixture strategy modules into the source tree (existing pattern);
  a killed run leaves the checkout dirty until cleaned

## Development notes

- The frozen-wire protocol is named and versioned independently from currently stored working-tree
  protocol stamps. Version and route the new transport explicitly.
- Prefer focused `live` protocol/dispatcher modules and registry materialization modules. Do not grow
  the size-pinned command or loop modules.
- `KeyboardInterrupt`, `SystemExit`, SQLite faults, global halt and account-wide reconciliation/risk
  failures remain systemic; do not catch them as tenant setup failures.
- Scrubbing inherited variables reduces accidental authority. It does not defend against malicious
  same-UID code with filesystem/network access.

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
- [Artifact-freeze design](../../superpowers/specs/2026-09-22-artifact-freeze-design.md)

## Dev Agent Record

### Implementation Plan

Contract-first (PR #678), then disjoint slices implemented test-first by Claude subagents and
committed by the coordinator: containment primitive; shared canonical encoder; planner port and
failure vocabularies; wire codec; child entry; supervisor port; registry tenant resolution and
strict decoder; frozen-only intake; promotion refusal; CLI wiring; protection. Then one bounded
independent review round and fixes.

### Completion Notes

- New admissions are frozen; working-tree and legacy tenants keep their paths (pinned by traces).
- A frozen tenant's Phase A and Phase B each run in a fresh child launched from its verified bundle
  and environment; the supervisor re-derives every strategy-free risk wall, the Phase A binding and
  the late state, owns all effects, and isolates any child fault as one stable, deployment-bound
  tenant failure; systemic faults stay systemic.
- `paper promote`, the forward gate, raw transitions and go-live refuse frozen deployments until
  Story 1.3d.
- Evidence: end-to-end tests run real children for every failure class and prove checkout
  independence across restarts; the opt-in real-environment test passed in 131 s against a real uv
  environment; parity is exact (canonical encodings and run_tick effect traces).

### File List

Production: algua/primitives/contained_process.py; algua/contracts/{canonical,float_fields,types}.py;
algua/live/{frozen_wire,frozen_wire_json,frozen_wire_arrow,frozen_wire_result,frozen_child,
frozen_dispatch,frozen_invocation,planner,planner_contract,planner_decision,planner_early,
planner_late,planner_validation,live_loop,paper_loop}.py; algua/risk/limits.py;
algua/registry/{frozen_runtime,frozen_view,frozen_tenant_errors,intake,deployment,deployment_runtime,
paper_runtime,gating,forward_promotion,transitions,artifact_errors,artifact_contract,artifact_manifest,
artifact_preparation,artifact_recording,environment_contract,frozen_manifest_contract,frozen_source,
planner_environment,planner_environment_probe}.py; algua/registry/store/deployment.py;
algua/strategies/base.py; algua/cli/{paper_cmd,paper_venue,registry_cmd,_common,errors,lane_refresh}.py.
Protection and docs: CODEOWNERS, pyproject.toml (import contract), tests/test_repo_hygiene.py,
tests/test_module_size_ratchet.py, CLAUDE.md, docs/architecture.md, docs/contracts/cli-error-envelope.md,
the Story 1.3c spec folder. Tests: tests/test_frozen_*.py, tests/_frozen_harness.py,
tests/_frozen_paper_world.py, tests/primitives/test_contained_process.py, tests/contracts/,
tests/test_contract_float_fields.py and updates to existing planner, paper, intake and deployment tests.

### Change Log

- 2026-09-30: Contract and readiness (PR #678); implemented frozen paper execution; review round 1
  applied; moved to review.
