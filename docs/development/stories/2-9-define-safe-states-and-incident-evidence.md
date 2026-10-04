---
baseline_commit: e875b6d
---

# Story 2.9: Define the safe states and assemble incident evidence

Status: backlog

Prepared: 2026-10-04. Baseline: `e875b6d` (main after PR #686). Epic: 2.
Requirements: FR13 (predefined safe state, incident evidence), NFR6, NFR8. PRD §§15, 20, 24.
Depends on: Stories 2.5, 2.6 and 2.8, whose refusals and halts join the catalogue.
Gated by: Story 2.8 merged, then contract and readiness review. No owner decision is needed to build
it; the owner must choose where alerts go before Story 2.10 runs.

## Story

As Algua's operator,
I want a reviewed list of every safe state the system may enter on its own, and one command that
turns those entries into incidents with their evidence,
so that the 72-hour exercise, and every unattended day after it, can be judged against states
defined in advance rather than explained after the fact.

## Context

PRD §15 sets the default response to uncertainty: prevent new exposure, reconcile, collect evidence,
create an incident, repair where authorized, verify, resume only when safe. FR13 requires a
*predefined* safe state and incident evidence.

The safe states exist but are scattered and undocumented as a set: tenant setup errors
(`strategy_setup_error`), kill-switch trips (`kill_switch_trip`), refresh and planning failures
(`bars_refresh_failed`, `cycle_plan_failed`), reconcile deferral and drift halts, dark-feed halts
(`live_mark_freshness_halt`, `paper_mark_freshness_halt`, `book_stale_marks_halt`), the book breaker
(`book_circuit_breaker`), operator anomaly alert kinds (`marker_corrupt`, `operator_lock_stuck`,
`calendar_out_of_bounds`, `session_gap`, `completion_unconfirmed`, which today exist only as log
records), and after Stories 2.5–2.8 the envelope deferrals, `live_paused` and
`release_unvalidated`. Production's audit log on 2026-10-04
holds 1,499 `strategy_setup_error` and 133 `bars_refresh_failed` rows.

Operator alerts are a structured log record plus an optional command (`algua/operator/alerts.py`,
`ALGUA_ALERT_CMD`); they are not stored in the registry. `fleet health` is a liveness gate "for an
external watchdog" (`algua/cli/fleet_cmd.py`), but no unit runs it.

## Scope and authority

In scope: a safe-state catalogue as a reviewed contract; a test tying it to the code; registry rows
for operator alerts; a read-only incident command; a watchdog unit.

Must not: add new halt behavior; change any trigger, threshold or recovery authority; open GitHub
issues or repair anything automatically (PRD Phase 4); treat log text as authority.

## Acceptance criteria

1. **Catalogue.** A contract document lists every state the system may enter without a human: its
   trigger, the audit action or actions that evidence it, what it prevents, what it preserves
   (positions, evidence, orders), its scope (tenant, cycle, lane, account), and the recovery command
   with the authority it needs (agent, human, signed human).
2. **Tied to the code.** A test collects every audit action written on a halt, refusal, deferral or
   isolation path and requires each to be in the catalogue, and every catalogue entry to have a
   writer, so a new safe state cannot ship undocumented.
3. **Alerts land in the registry.** Every operator alert also appends an `operator_alert` audit row
   with its kind and job. The write is best-effort and can never crash or fail the run.
4. **Incident command.** `fleet incidents --since <ts>` (read-only JSON) groups catalogue entries
   and operator alerts into incidents: trigger, time, scope, affected strategies, the state rows at
   entry (halt, latch, kill switch), the audit trail, the recovery actions taken and whether the
   state is still active. It exits non-zero while any incident is open.
5. **Watchdog.** `algua-fleet-health.{service,timer}` runs `algua fleet health` on a fixed cadence
   and calls the alert hook when it exits non-zero. The installer renders it and prints its enable
   command (it is a read-only monitor, unlike the live job).
6. **Green.** Full gate passes; protected review of the catalogue.

## Tasks / subtasks

- [ ] Inventory every safe-state path in the code; write the catalogue contract (AC1).
- [ ] Catalogue tie test (AC2).
- [ ] Alert audit rows (AC3).
- [ ] `fleet incidents` (AC4).
- [ ] Watchdog unit and installer (AC5); full gate; review (AC6).

## Dev notes

- The operator package is a stdlib-leaf by design (`algua/operator/jobs.py` docstring). Record alert
  rows from the CLI composition point (`algua/cli/operator_cmd.py`), not from `algua.operator`.
- Group repeated entries into one incident by trigger and scope until a recovery or a clean cycle
  closes it; production's repeated setup errors show why grouping matters.

## Owner decisions

None to build. Before Story 2.10: choose the alert destination for `ALGUA_ALERT_CMD` (human access;
keep credentials out of the repository and task comments).

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

## References

- `docs/PRD.md` §§15, 20, 24; [epics.md](../epics.md) FR13
- `deploy/systemd/README.md` (alert hook, operator anomalies)
- Todoist:
  [Run the 72-hour unattended acceptance exercise](https://app.todoist.com/app/task/run-the-72-hour-unattended-acceptance-exercise-6hfCrg5WwPWmfPqp)

## Dev Agent Record

### Agent Model Used

### Completion Notes

### File List
