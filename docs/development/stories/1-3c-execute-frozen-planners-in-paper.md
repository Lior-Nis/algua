---
baseline_commit: dc2a222ef811dc1c3a835d656a411de58423796c
---

# Story 1.3c: Execute frozen planners in paper

Status: backlog

Prepared: 2026-09-25. Baseline: Story 1.3 readiness baseline `dc2a222` (PR #671).
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

- [ ] Add canonical Parquet/request/result codec tests and protocol limits (AC3, AC6).
- [ ] Build the child entry point and strict response parser (AC2–AC7).
- [ ] Implement process-group timeout/output handling and environment scrubbing (AC4–AC7).
- [ ] Extend new candidate intake to activate only preverified frozen content (AC1).
- [ ] Dispatch frozen tenants while preserving compatibility paths (AC8–AC10).
- [ ] Add the explicit frozen-promotion block (AC11).
- [ ] Prove parity, determinism, restart and mutable-worktree independence (AC8–AC10).
- [ ] Run protected independent review before enabling new frozen intake (AC12).

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
