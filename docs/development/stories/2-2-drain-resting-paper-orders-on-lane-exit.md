---
baseline_commit: e875b6d
---

# Story 2.2: Drain a strategy's resting paper orders on every paper-lane exit

Status: ready-for-dev

Prepared: 2026-10-04. Baseline: `e875b6d` (main after PR #686). Epic: 2.
Requirements: FR13 (safe unattended paper operation), FR1 (the `forward_tested -> live` edge is a
paper-lane exit), NFR2, NFR4–NFR6. Issue: [#685](https://github.com/Lior-Nis/algua/issues/685).
Depends on: Story 1.4's merged code (PR #686), not on Story 1.4's operational exit.
Readiness: READY WITH CONDITIONS
([readiness report, 2026-10-05](../implementation-readiness-report-2026-10-05-story-2-2.md)); all
conditions applied to the contract on 2026-10-05 (M1–M3, m1–m11; see the decision log's
"Readiness corrections" entry). Status stays `backlog` until the docs PR carrying the corrected
contract merges.

## Story

As Algua's operator,
I want every paper-lane exit to cancel the strategy's own resting paper orders and refuse while one
remains,
so that a strategy cannot leave the paper book and have an order fill afterwards into an orphaned
position that halts the whole account.

## Context

Book-exit edges shed the strategy's allocation atomically with the stage change
(`algua/registry/transitions.py:30-33`). The paper-source edges are
`paper -> dormant | retired | candidate` and `forward_tested -> retired | live`. Inside the exit
transaction, `_assert_flat_for_bench` (`algua/registry/store/crud.py:333-357`) checks ledger
positions; it re-lists open orders only when an `ExitLaneGuard` is injected.

Only the live lane injects one. `registry_cmd.py:38-44` records the paper drain as a deferred
follow-up, and `_live_exit_guard` (`algua/cli/registry_cmd.py:47-93`) returns `None` for any
non-live source. A paper strategy can therefore leave its book with a resting order. If that order
fills, the paper account reconcile sees an unexplained position, defers the cycle and, after the
grace window, engages the global halt, which also stops `live run-all`.

Two Epic 2 paths make this urgent. The signed go-live (`forward_tested -> live`, Story 2.3) is a
paper-lane exit. The 72-hour exercise (Story 2.10) runs the paper lane unattended, where an operator
cannot wait for fills by hand. Story 1.4 mitigated by operator discipline (`paper flatten`, wait,
then retire).

The building blocks exist: `owned_open_order_ids(..., kind=LedgerKind.PAPER)`
(`algua/execution/live_ledger.py:605-620`), `paper_scoped_cancel` and `ingest_paper_venue`
(`algua/cli/paper_venue.py:63-79`), and the live guard's shape (`LiveExitGuard`,
`algua/execution/lane_exit.py:43-68`; protocol `algua/contracts/types.py:401-416`).

## Normative contract

The [Story 2.2 machine contract](../specs/spec-story-2-2-paper-exit-drain/SPEC.md) and its
[field-level companion](../specs/spec-story-2-2-paper-exit-drain/paper-exit-drain-contract.md) are
normative. They hold the entry-point inventory and settle the guarded edges (go-live included), where
selection happens, the guard's algorithm, its clock and cursor, the operator lock, every message,
audit action and code, placement, protection and the tests. Implementers and reviewers must read
both; where they refine an acceptance criterion, the Dev notes below say so.

## Scope and authority

In scope: a paper `ExitLaneGuard` wired into every paper-source book exit reachable from
`registry transition`; fail closed when the paper venue cannot be reached; move the guard selection
out of `registry_cmd.py`.

Must not: change live exit behavior or messages; change Story 1.4's dust rule or the positions
check; cancel a sibling's orders or use the account-wide cancel; add a command, schema change or
authority. Exiting a book is risk-reducing and stays agent-allowed. Does not address #647 (paper
whole-account loss breaker).

## Acceptance criteria

1. **Drain before the lock.** For every paper-source edge that sheds the allocation, the transition
   runs a paper guard before the registry write lock: cancel this strategy's own open paper orders
   (scoped through the paper order ledger), then ingest the paper venue activity feed so a fill that
   raced the cancel reaches the ledger.
2. **Re-check under the lock.** Inside the exit transaction the guard re-lists the strategy's
   still-open paper orders; any remaining order refuses the exit with a `TransitionError` naming the
   order ids. The positions check (with the Story 1.4 dust rule) runs as today.
3. **Fail closed.** Missing paper credentials, a failed open-order read, a failed cancel or a failed
   ingest refuses the exit. It never falls back to the positions-only check. The refusal is audited
   (`paper_exit_drain_unavailable` or `paper_exit_drain_failed`) with an actionable message, except a
   re-list failure under the write lock, which rolls back and is reported by the envelope (amended
   2026-10-05, readiness m2; contract §4).
4. **Sibling safety.** A sibling's resting order on the shared paper account survives the exit
   (test). The account-wide `cancel_open_orders` is never called on this path.
5. **Race coverage.** A test fills the strategy's order between cancel and ingest: the fill lands in
   the ledger and the exit is refused as "not flat". A test leaves a non-cancelable order: the exit
   is refused under the lock.
6. **Live unchanged.** Live exits keep their drain, fallback to account-level drain credentials and
   messages, byte for byte. Edges that keep the allocation (`paper -> forward_tested`,
   `forward_tested -> paper`) get no guard.
7. **Structure.** Guard selection for both lanes moves out of `registry_cmd.py` (two lines under its
   446 pin), which shrinks. A structural test asserts that both lanes' book exits carry a drain, so
   the lanes cannot drift on this again. No import-linter exemption.
8. **Green.** Full root gate passes; each new refusal is mutation-checked.

## Tasks / subtasks

- [ ] Contract and readiness: entry-point inventory of paper-source exits (today only
      `registry transition`; `transition_strategy`'s other caller, `evaluation/backtest_run.py:105`,
      moves only to `backtested`) (AC1–AC3).
- [ ] Paper guard, test-first: cancel, ingest, under-lock re-list (AC1–AC2, AC5).
- [ ] Fail-closed construction and audit (AC3).
- [ ] Move guard selection out of `registry_cmd.py`; structural lane test (AC6–AC7).
- [ ] Sibling-safety and live-unchanged regression tests (AC4, AC6).
- [ ] Mutation checks, full gate, independent review (AC8).

## Dev notes

- Placement (settled by the contract, §2, §3.1 and §6).
  - `ingest_paper_venue` and `recover_stranded` move to a new `algua/execution/venue_sync.py`. It
    also gains the drain's `ingest_paper_venue_keep_cursor`. These functions reach only the ledger and
    the audit log, so no new package edge.
  - `PaperExitDrainBroker`, `PaperExitGuard` and `build_paper_drain_broker` live in a new
    `algua/execution/paper_exit_drain.py`. `algua/contracts/types.py` is at its ratchet pin and is
    not edited, and `lane_exit.py` would pass 300 lines with the guard in it.
  - The one selector for both lanes, `select_exit_guard`, lives in `algua/execution/lane_exit.py`.
    `transition_strategy` calls it itself, through a lazy default like
    `_default_forward_certificate_verifier`, so every caller gets the drain and `registry_cmd.py`
    passes nothing.
  - The `exit_guard` parameter becomes `exit_guard_selector`.
- The broker-time `until` comes from `tick_clock(broker.clock)`, as in the paper flatten
  (`algua/cli/paper_cmd.py:1304`), but the exit refuses `tick_clock`'s local-clock fallback (contract
  §3.2). The drain's ingests never advance the shared paper fill cursor: they re-store the cursor
  they read (contract §3.2).
- `ExitLaneGuard.cancel_and_ingest` runs outside the transaction and commits its own ingest. It stays
  outside the `try`/`BEGIN IMMEDIATE`, exactly as the live path does (`store/crud.py:292-300`). The
  store is not edited, apart from one docstring.
- Refinements of the acceptance criteria, all in the contract:
  - AC1: the guard syncs the venue first and skips the cancel when the ledger holds a material
    position, so a premature exit leaves resting liquidation offsets in place.
  - AC1, AC2: it re-reads the strategy's open orders after the cancel, re-listing up to 3 more times
    at 1-second intervals while a cancelled order is still listed (`pending_cancel`), and syncs again
    afterwards. An order still open at the last read refuses the exit under the lock.
  - AC1, AC5: after the final sync it settles every order any drain has asked the venue to cancel
    for the strategy, comparing the venue's `filled_qty` (by-coid lookup) with the ledger's fills for
    that order. A fill the venue has executed but not yet published refuses the exit, on every retry,
    until the ledger holds it. So AC5's race is covered whether the venue publishes the fill at once
    or late.
  - AC3: drain failures before the lock are audited and raise `BrokerError` (`broker_error`),
    including a failed or 404 order lookup in the settle step. Missing credentials raise
    `TransitionError` (`wrong_stage`). A re-list failure under the lock rolls back unaudited, because
    the audit append commits; AC3 is amended to say so.
  - Before each cancel is sent, the guard commits a `paper_exit_drain_cancel_requested` audit row
    whose reason is the broker order id. These rows are the settle step's durable record.
  - Every paper-lane exit, go-live included, takes `operator.lock`. It excludes the paper timer only
    when run from the operator's checkout. A go-live refused for a held lock leaves its challenge
    unconsumed; if the challenge expires, the human issues a new one.
  - AC6: live messages and audit actions are unchanged. Live selection now runs after validation
    and inside the lock, so a live exit refused by validation or a held lock no longer writes a
    `live_exit_drain_*` row (contract §2.4).
- Test isolation: an autouse guard in `tests/conftest.py` refuses HTTP to any `*.alpaca.markets`
  host and fails the test, so a test that forgets the explicit fake-venue fixture cannot reach the
  network (contract §8).
- Corrected references: `_assert_flat_for_bench` is `store/crud.py:319-353`; `LiveExitGuard` is
  `lane_exit.py:43-72`; the protocol is `contracts/types.py:401-417`.
- Coordination with Story 2.1: both edit `transitions.py` (unpinned, 277 lines, and the ratchet
  forbids it reaching 300) and `registry_cmd.py`. This story budgets `transitions.py` to 290 lines
  and frees about 60 lines in `registry_cmd.py`. **This story merges first.** Story 2.1 rebases onto
  it (contract §6).
- Story 1.4 operator: production still holds one legacy tenant at `paper`,
  `liquidity_stable_quality_momentum` (registry copy, 2026-10-05). It is kill-switched with a
  resting UNH sell offset. If this story merges first, its retirement drains through the guard:
  - before the offset fills, the retirement is refused on positions and the offset stays resting;
  - if the offset fills in full, the remaining residual is dust and the retirement commits.

  It needs the paper credentials the box already has.

### Test matrix

The contract's T1–T20 cover:

- each paper-source edge with and without a resting order;
- a sibling order surviving;
- a fill between cancel and ingest, published at once or late (refused until ingested, then the exit
  commits or is refused on positions);
- the drain never moving the shared cursor;
- a non-cancelable order and a `pending_cancel` that clears;
- missing credentials;
- clock, ingest, read, cancel, re-check and settle failures;
- live exits unchanged, with the two timing pins;
- non-revoking edges staying unguarded;
- the structural parity test;
- the suite-wide Alpaca-host guard.

## Owner decisions

None.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

## References

- [#685](https://github.com/Lior-Nis/algua/issues/685); #497 (exit drain design)
- [Story 1.4](1-4-controlled-exit-of-the-legacy-paper-cohort.md) review finding (deferred to #685)
- `docs/architecture.md` (lane parity wall)

## Dev Agent Record

### Agent Model Used

### Completion Notes

### File List
