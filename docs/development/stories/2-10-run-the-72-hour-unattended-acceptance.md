---
baseline_commit: e875b6d
---

# Story 2.10: Run the 72-hour unattended acceptance exercise

Status: backlog

Prepared: 2026-10-04. Baseline: `e875b6d` (main after PR #686). Epic: 2.
Requirements: FR13, FR14 (paper validation of the release under test), NFR4, NFR8. PRD §§15, 20, 24.
Depends on: Stories 2.1–2.9 merged and running on the operator checkout.
Gated by: those merges plus operating preconditions an agent cannot create (below). No code is
expected; defects found become new stories.

## Story

As Algua's owner,
I want the system to run for 72 hours without anyone attending it, either operating correctly or
reaching one of its predefined safe states on its own, with the evidence written down,
so that I can judge whether the operating kernel is ready for my separate real-money decision.

## Context

PRD §20's acceptance target is "72 hours unattended while either operating correctly or autonomously
reaching a predefined safe state". FR13 adds controlled restart, reconciliation and incident
evidence and says real-money acceptance additionally needs human authorization and account
readiness.

There is no shadow lane, so this exercise runs in paper. It exercises the always-on machinery
(timers, bar refresh, frozen dispatch, reconcile, halts, the operator wrapper, the watchdog) and the
paper validation of the release under test. Live-only walls (pause, envelope, live reconcile) are
covered by their stories' tests, not by this exercise; showing them on the real account is part of
real-money acceptance.

Production on 2026-10-04: the research, leap, forage and merge-back timers run; the paper timer is
off until Story 1.4's exit completes; the paper book is empty after the last legacy tenant retires;
no frozen deployment exists yet; no strategy is live.

## Scope and authority

In scope: a reviewed exercise plan, pre-scheduled drills, the 72-hour run and the evidence report.

Must not: activate any strategy, enable the live job, fund or transfer money, allocate live capital,
or patch code during the window. A defect found becomes a new story and the exercise is re-run.
Drills must not damage a real tenant's evidence: no drill trips a kill switch on a real tenant,
corrupts its content or skips one of its sessions.

## Preconditions

- Stories 2.1–2.9 merged and deployed to the operator checkout; schema migrated.
- All timers enabled, including paper and the fleet-health watchdog; the live job stays disabled.
- At least one frozen deployment ticking in the paper book with a clean recent cycle. This depends
  on the research factory producing a qualified candidate; an agent cannot create one.
- `ALGUA_ALERT_CMD` routes alerts by email through the kaggler SMTP sender (owner decision
  2026-10-04).
- `fleet health` green and `fleet incidents` empty of open incidents at the start.

## Acceptance criteria

1. **Plan first.** Before the window opens, a reviewed plan fixes: a window of at least 72
   contiguous hours spanning at least three completed exchange sessions; each drill, when it runs
   and the catalogued safe state it should produce; abort criteria; the evidence to collect. Drills
   are scheduled scripts, so no one acts interactively during the window.
2. **Controlled restart.** A drill kills the paper operator mid-cycle and another restarts the
   services or host. On the next fire the system recovers on its own: stranded orders recovered,
   reconcile clean or deferred then clean, the session completed exactly once, no duplicate order,
   trace audit complete.
3. **Data outage.** A drill blocks the bars refresh for one fire: the cycle ends `refresh_failed`
   with no order, and the next fire succeeds.
4. **Account-wide stop.** A drill engages the global halt between sessions: the watchdog alerts, the
   incident is recorded, and recovery through the catalogued path resumes the next session with no
   loss of evidence coverage.
5. **Outcome.** For the whole window, every cycle either completed correctly (reconcile clean, trace
   audit complete, fleet health green) or entered a catalogued safe state autonomously with its
   incident recorded. There is no uncatalogued halt, orphaned position, unattributable fill or
   duplicate order.
6. **Evidence report.** A report committed under `docs/development/` gives the timeline, every
   cycle's outcome, each drill against its expected state, the incident list, the trace audit, the
   watchdog log and every defect found with its new story or issue.
7. **No real-money claim.** The report states that real-money acceptance was not performed and lists
   what it still requires: the owner's account and deployment prerequisites, the activation inputs
   (ceiling, permitted instruments), a signed go-live, a signed allocation and enabling the live
   job.
8. **Failure is reported, not hidden.** A window that ended in a catalogued safe state passes this
   criterion. One that ended anywhere else fails; the defect is fixed in a new story and the
   exercise is re-run from the start.

## Tasks / subtasks

- [ ] Confirm the preconditions and record them.
- [ ] Write and review the exercise plan and drill scripts (AC1).
- [ ] Run the window; collect evidence (AC2–AC5).
- [ ] Write the report; file defects (AC6–AC8).

## Dev notes

- Schedule drills after the session's paper cycle has completed, so no real tenant misses a session
  and its forward-gate coverage is unaffected.
- A global halt drill makes `fleet health` red by design; that is the expected alert, not a failure.
- `paper resume-all` is the catalogued agent recovery for a global halt; it never lifts the live
  pause.

## Owner decisions

Alert destination, decided 2026-10-04 (Lior): **email** to the owner through the existing kaggler
SMTP sender on the box, driven by systemd `OnFailure=` hooks and the `fleet health` watchdog. The
owner decides afterwards, separately, whether to proceed to real-money acceptance; this story does
not ask for it.

## References

- `docs/PRD.md` §§15, 20, 24; `docs/vision-reconciliation.md` (unattended operation)
- [Story 2.9](2-9-define-safe-states-and-incident-evidence.md) (catalogue and incidents)
- Todoist:
  [Run the 72-hour unattended acceptance exercise](https://app.todoist.com/app/task/run-the-72-hour-unattended-acceptance-exercise-6hfCrg5WwPWmfPqp)

## Dev Agent Record

### Agent Model Used

### Completion Notes

### File List
