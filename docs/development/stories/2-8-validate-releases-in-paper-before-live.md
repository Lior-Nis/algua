---
baseline_commit: e875b6d
---

# Story 2.8: Validate each supervisor release in paper before it trades live

Status: backlog

Prepared: 2026-10-04. Baseline: `e875b6d` (main after PR #686). Epic: 2.
Requirements: FR14 (release validation in paper), FR13 (unattended live operation after activation),
FR9 (outside-planner fixes ship without requalification), NFR1, NFR4, NFR6. PRD §§16–18, 20.
Depends on: Story 2.7 (trace audit) and Story 2.6 (risk-reducing steps that must still run).
Gated by: Story 2.7 merged, then contract and readiness review. The owner decision on the minimum
paper validation was made on 2026-10-04 (below).

## Story

As Algua's owner,
I want the live lane to refuse to trade on supervisor code that has not first run correctly in
paper, and a live operator job that is ready to enable when I activate,
so that every software change reaches real money only after the paper rehearsal has exercised it,
and live operation does not need me every day.

## Context

Frozen deployments make strategy decisions immune to repository changes, but the supervisor that
wraps them (sizing, risk walls, broker adapter, reconcile, ledgers) runs from the current checkout.
PRD §16 keeps paper permanently to "validate software releases", and §17's loop puts "paper/shadow
validation" before deployment. Today a merged supervisor change reaches the next `live run-all`
directly; nothing records which code a cycle ran on.

- Autonomous merge-back may change only `algua/strategies/**` and `kb/**`
  (`algua/operator/diff_policy.py:53-55`). Frozen tenants never import the checkout's strategy
  modules, so those paths do not change the supervisor.
- A clean-checkout check already exists (`algua/registry/frozen_source.py:264`,
  `assert_clean_head`).
- The operator runs only one session-gated job, `paper` (`algua/operator/jobs.py:89-110`); there is
  no live job, and `deploy/systemd/` ships no live unit.
- There is no shadow lane. Paper exercises the shared tick engine, frozen dispatch and paper
  ledgers; live-only paths (live ledger and reconcile, book breaker, pause, envelope) run only in
  live. Paper validation is therefore real for the shared code and partial for live-only code, which
  tests cover. Unifying the lanes (#647 part 2) is outside Epic 2.

## Scope and authority

In scope: a supervisor release identity; an append-only cycle record for both lanes; a release
validation check that blocks live strategy ticks; a live operator job and units shipped disabled.

Must not: enable the live job or any timer (enabling is part of the owner's activation); block a
risk-reducing step (ingest, reconcile, pause evaluation, book breaker, cancels, flatten) on an
unvalidated release; change paper behavior beyond recording its cycles; grant merge or deployment
authority.

## Acceptance criteria

1. **Release identity.** One function returns the supervisor release identity of the running
   checkout: a digest of the tracked tree excluding `algua/strategies/**` and `kb/**`, the
   dependency lock digest and the interpreter identity. A checkout with any tracked modification, or
   one that cannot be verified, has no release identity.
2. **Cycle record.** Every paper and live `run-all` writes an append-only cycle row: lane, release
   identity (or none), start and end, outcome (completed, deferred, failed, systemic, refused), the
   number of frozen tenants ticked and the bars snapshot id.
3. **Validation rule.** A release is validated when it meets the owner-decided minimum paper
   evidence (see below), evaluated from cycle rows and the Story 2.7 trace audit.
4. **Live refuses unvalidated releases.** On a release that is unvalidated or has no identity,
   `live run-all` runs every risk-reducing step but no strategy tick and no new order, records the
   cycle as `refused`, and exits non-zero with `release_unvalidated`.
5. **Live operator job, disabled.** `OPERATOR_JOBS` gains `live`, running
   `algua live run-all --refresh` with a completion predicate like paper's that also rejects
   `release_unvalidated` and `live_paused`. `algua-live.{service,timer}` fire after the paper grid
   so a release paper validated that session can trade. `install-user-units.sh` renders them but
   never prints an enable command for them; the README documents enabling as an owner activation
   act.
6. **Visibility.** `fleet status` reports the current release identity and whether it is validated.
7. **Green.** New modules protected; full gate passes; protected review.

## Tasks / subtasks

- [x] Owner decision recorded (below).
- [ ] Contract and readiness: identity inputs, cycle DDL,
      validation rule, job manifest, unit timing.
- [ ] Release identity, test-first (AC1).
- [ ] Cycle records in both lanes (AC2).
- [ ] Validation rule and the live refusal (AC3–AC4).
- [ ] Live job and disabled units; installer and README (AC5).
- [ ] Fleet visibility (AC6); full gate; independent review (AC7).

## Dev notes

- Exclude only what autonomous merge-back can change and what frozen tenants never execute. A human
  edit to a strategy module that a non-frozen tenant still imports would not change the identity;
  after Story 2.3 no such tenant can be live.
- Timing: paper fires at :07/:27/:47 after the close and records the session once; give the live
  timer its own offset after that, and let a refused live cycle retry on the next fire, as paper's
  deferred cycles do.
- An empty paper book cannot validate anything, so the live lane would stay refused. That is the
  intended fail-closed outcome unless the owner decides otherwise.

## Owner decisions

**Decided 2026-10-04 (Lior).** A supervisor release may run live only after at least one completed
paper cycle on the identical release that ticked at least one frozen tenant, with a clean reconcile,
no systemic failure and a complete trace audit. When the paper book is empty, live stays refused:
no paper evidence, no live release.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

## References

- `docs/PRD.md` §§16–18, 20; [epics.md](../epics.md) FR14, FR9
- `deploy/systemd/README.md` (operator wrapper, session marker, run lock)
- [#647](https://github.com/Lior-Nis/algua/issues/647) (lane parity; out of scope)
- Todoist:
  [Complete live traceability and safe release validation](https://app.todoist.com/app/task/complete-live-traceability-and-safe-release-validation-6hfCrg6qX8q4fvgG)

## Dev Agent Record

### Agent Model Used

### Completion Notes

### File List
