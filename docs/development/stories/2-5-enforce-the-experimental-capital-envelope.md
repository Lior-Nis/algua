---
baseline_commit: e875b6d
---

# Story 2.5: Enforce the experimental capital envelope

Status: backlog

Prepared: 2026-10-04. Baseline: `e875b6d` (main after PR #686). Epic: 2.
Requirements: FR11 (instrument policy and capital ceiling), NFR4, NFR6, NFR7. PRD §§15, 19, 21.
Depends on: Story 2.4 (live tick carve and frozen live path) and Story 2.3 (go-live checks it
extends).
Gated by: Story 2.4 merged, then contract and readiness review. Building needs no owner decision.
Activating needs two owner inputs from the open account/deployment prerequisites task: the USD
ceiling equivalent to ₪2,000 and the list of permitted instruments.

## Story

As Algua's owner,
I want the live lane to trade only long positions in stocks and unleveraged ETFs I have approved,
with no borrowing and never more than the approved capital,
so that the experimental account cannot take a risk the PRD forbids, even if a strategy, an agent or
a setting asks it to.

## Context

PRD §21 sets the experimental account: ₪2,000 of personal capital, long-only stocks, unleveraged
ETFs, no borrowing, no derivatives. The reconciliation record calls these unenforced (gap 5). In the
code:

- `live allocate` checks only that live allocations sum to at most account equity
  (`algua/cli/live_cmd.py:98-119`, `algua/registry/allocations.py:74-114`). It records
  `actor="human"` without authenticating anyone, although PRD §19 reserves increasing live capital
  to the human and #329 established that a bare human string unlocks nothing.
- Book caps are generic and loosenable from the environment: `book_max_gross=2.0`,
  `book_max_net=1.0` (`algua/config/settings.py:61-68`), consumed by `build_book_exposure`
  (`algua/live/book_exposure.py`). A short position only defers the cycle
  (`book_exposure.py:46-49`).
- The buying-power pool is the broker's `buying_power` (`live_cmd.py:570`), which includes margin on
  a margin account. `AccountState` has no margin or shorting fields
  (`algua/execution/alpaca_broker.py:51-62`, `:210-221`).
- A strategy's contract allows `max_gross_exposure` and `allow_short`
  (`algua/contracts/types.py:73-75`); nothing stops a deployment with leverage or shorting from
  going live.
- Nothing restricts which symbols a live order may buy. Alpaca asset metadata cannot tell a
  leveraged or inverse ETF from an ordinary one, so the policy needs an explicit list.
- `algua/cli/live_cmd.py`, `algua/risk/*` and `algua/config/settings.py` are not in `CODEOWNERS`.

## Scope and authority

In scope: one protected policy module; enforcement at `live allocate`, at go-live, and in every live
cycle; authenticated allocation increases; a read-only readiness report for the activation check.

Must not: transfer, fund or allocate money; set the ceiling or the instrument list (owner inputs at
activation); loosen any existing wall (the policy only tightens: the effective value is always the
stricter of the policy and the existing setting); block a risk-reducing sell or exit; give agents
authority to increase capital. Decreasing a live allocation stays agent-allowed (PRD §19: agents
may always reduce risk).

## Acceptance criteria

1. **Protected, fail-closed policy.** One CODEOWNERS-protected module, in the integrity-critical
   set, holds the envelope: capital ceiling in USD (unset by default), long-only, gross exposure at
   most 1.0 of capital, and the permitted instruments (empty by default). These are code constants,
   not environment settings. With the ceiling unset or the list empty, no live allocation and no
   live buy can happen.
2. **Ceiling on allocation.** `live allocate` refuses when the ceiling is unset or when the sum of
   active live allocations after the change would exceed it. Boundary: equal to the ceiling is
   allowed, one cent over is refused (decimal arithmetic).
3. **Authenticated increases.** A new allocation or a larger one requires an authenticated human: a
   single-use challenge under `algua-go-live` with a distinct purpose, binding strategy, deployment
   id, new capital, the ceiling and the live account id. A bare `--actor human` or an agent cannot
   increase. A decrease needs no signature. The challenge mechanism is a reusable account-level
   signed command (Story 2.6 reuses it).
4. **Deployment policy at go-live.** Go-live (Story 2.3's checks) refuses a deployment whose
   recorded config has `allow_short=True` or `max_gross_exposure` above 1.0, or whose current
   gate-universe membership contains a symbol outside the permitted list. The refusal and the
   challenge summary list the offending values.
5. **Per-cycle orders.** In `live run-all`: a buy of a symbol outside the permitted list is dropped
   before submission and audited (`live_order_outside_envelope`), while sells always pass; the
   buying-power pool is non-borrowed cash, never margin buying power; the book's gross notional is
   capped at 1.0 times the smaller of account equity and the ceiling (tightening the existing caps);
   no order may leave a position short.
6. **Impossible states prevent exposure.** Negative cash or a short position on the live account
   defers the whole cycle with a distinct reason and an alert-visible audit row, with no orders (PRD
   §15). Existing defer and halt semantics are otherwise unchanged.
7. **Readiness report.** `live readiness` (read-only, no orders, no secrets printed) reports each
   activation input: ceiling set, permitted list non-empty, live bars provider set, live account
   readable, the account's broker-reported margin multiplier and shorting status, trust anchor
   present with every line namespace-scoped, at least one key enrolled for `algua-go-live`. It exits
   non-zero when any is unmet. It is evidence for the owner's activation check, not a substitute.
8. **Tests and green.** Unit tests at every boundary; a live-cycle test per enforcement point; agent
   increase refused; agent decrease allowed; unset ceiling and empty list refuse all live buys. New
   modules protected; `live_cmd.py` not grown; full gate passes.

## Tasks / subtasks

- [ ] Contract and readiness: constants, enforcement points, the account-level signed-command
      payload, entry-point inventory (`live allocate`, go-live, `live run-all`, any other allocation
      writer).
- [ ] Policy module and pure checks, test-first (AC1, AC2).
- [ ] Account-level signed command; authenticated `live allocate` increases (AC3).
- [ ] Go-live deployment checks (AC4).
- [ ] Live cycle wiring: instrument filter, cash pool, gross cap, no-short (AC5–AC6).
- [ ] `live readiness` (AC7); protection, full gate, independent review (AC8).

## Dev notes

- Read Alpaca's account fields for non-borrowed cash and the margin multiplier from the official
  documentation or a recorded fixture, not memory; parse them as optional and fail closed when
  absent. `AccountState` lives in `alpaca_broker.py`, pinned at exactly 547 lines: carve before
  adding.
- Compose with the book caps by taking the stricter value; never replace `book_max_gross` or let the
  environment loosen the envelope.
- Filter disallowed buys in the reservation hook (`live_cmd.py:572-589`) or before `submit_sized`,
  and record the drop through the same path Story 2.7 will trace.
- The permitted list is symbols, not an issuer rule; an empty list is the default.

## Owner decisions

Activation inputs (not needed to build): the USD ceiling equivalent to ₪2,000, and the permitted
instrument list. Interpretations the owner may confirm: the ceiling bounds capital at risk (live
allocations and gross exposure), not the account balance; an over-funded account keeps the excess in
cash.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

## References

- `docs/PRD.md` §§15, 19, 21; `docs/vision-reconciliation.md` gaps 5 and 4
- [#624](https://github.com/Lior-Nis/algua/issues/624); #329 (authenticated human actor); #497
- Todoist:
  [Enforce experimental capital envelope and 10% pause](https://app.todoist.com/app/task/enforce-experimental-capital-envelope-and-10-pause-6hfCrg6m7Pj76gXG);
  [account and deployment prerequisites](https://app.todoist.com/app/task/confirm-human-controlled-live-account-and-deployment-prerequisites-6hcQ3CHr49rCRV3p)

## Dev Agent Record

### Agent Model Used

### Completion Notes

### File List
