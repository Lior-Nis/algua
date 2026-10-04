---
baseline_commit: e875b6d
---

# Story 2.6: Pause live trading at a 10% drawdown of the experimental account

Status: backlog

Prepared: 2026-10-04. Baseline: `e875b6d` (main after PR #686). Epic: 2.
Requirements: FR11 (10% pause), NFR2, NFR4, NFR6, NFR8. PRD §§15, 19, 21.
Owner decision: [#624, comment of 2026-10-04, item 1](https://github.com/Lior-Nis/algua/issues/624).
Depends on: Story 2.5 (the account-level signed command it reuses). Functionally independent of
Stories 2.3–2.4; sequenced after 2.5 because it edits the same live cycle and the next schema
version.
Gated by: Story 2.5 merged, then contract and readiness review. No owner decision blocks the build;
two interactions are listed for the owner below.

## Story

As Algua's owner,
I want live trading to stop by itself when the experimental account falls 10% from its high-water
mark, keep the positions, and stay stopped until I sign a resume after reading a drawdown report,
so that a losing streak is interrupted for human review without a forced sale.

## Context

The owner decided on 2026-10-04:

- **Measurement:** drawdown from the live account's equity high-water mark; deposits and withdrawals
  move the peak by the same amount, so a cash flow is never a gain or a loss.
- **Boundary:** triggers at drawdown of 10% or more (inclusive).
- **On trigger:** engage the global live halt, cancel resting orders, block new orders, keep open
  positions; no forced liquidation.
- **Resumption:** only through a signed human command after a drawdown report; resuming re-bases the
  peak to current equity.

What exists, and why it cannot simply be reused:

- The book loss breaker (`algua/risk/book_cycle.py:23-56`, `algua/risk/book_breaker.py:56-100`) uses
  a strict `<` (`book_breaker.py:79`), a 15% default (`algua/config/settings.py:79`) and a peak with
  no cash-flow adjustment (`algua/risk/book_equity.py:18-36`). On a breach `live run-all` engages
  the global halt and closes every position (`algua/cli/live_cmd.py:503-532`).
- `paper resume-all` clears the global halt and every peak, including the book peak, and defaults to
  `--actor agent` (`algua/cli/paper_cmd.py:1345-1387`, `algua/execution/peaks.py:54-65`). A pause
  that lived in `global_halt` or `book_equity_peak` could be undone by an agent.
- The global halt is one row whose reason is overwritten on each engagement
  (`algua/risk/global_halt.py:18-29`). `live halt-all` engages it and stops paper and live
  (`live_cmd.py:718-743`); that is what "global live halt" means in this codebase.
- External capital flows are classified once (`EXTERNAL_CAPITAL_TYPES`,
  `algua/registry/forward_evidence.py:46-49`). Live fill ingest stores every non-fill activity with
  its `net_amount` in `live_activities` (`algua/execution/live_ledger.py:575-585`); malformed ones
  go to quarantine.
- `live run-all` returns before building the broker when no live strategy exists
  (`live_cmd.py:367-370`), and returns early on the global halt (`live_cmd.py:371-373`).
- `fleet health` exits non-zero on a global halt (`algua/cli/fleet_cmd.py:104-111`,
  `algua/execution/fleet_health.py:78-117`).

## Scope and authority

In scope: a separate, durable pause latch and high-water record; the trigger and its response; a
standalone check; the drawdown report; the signed resume; visibility in `fleet status` and
`fleet health`.

Must not: liquidate or offset positions on trigger; change the existing book loss breaker, its
thresholds or its precedence; let `paper resume-all`, `global_halt.clear` or `rebase_all_peaks` lift
the pause or touch its high-water mark; make the threshold configurable from the environment; give
an agent any way to resume. Stopping trading needs less authority than resuming it: engaging the
pause needs none.

## Acceptance criteria

1. **Cash-flow-adjusted high-water mark.** A record separate from `book_equity_peak` holds the
   account's high-water mark. It is set at the first evaluation. Each evaluation first adds the net
   external capital flows since the previous one (deposits raise it, withdrawals lower it), then
   raises it to current equity if equity is higher. Dividends, interest and fees are returns, not
   flows.
2. **Unvaluable flows fail closed.** An external-flow activity without a determinable cash amount
   (missing `net_amount`, an in-kind transfer, an unknown sign), or a quarantined live activity in
   the window, engages the pause with reason `unvaluable_capital_flow`.
3. **Inclusive, exact trigger.** Drawdown is `1 - equity / high-water`, computed in decimal. The
   pause engages at 10% or more. Tests at exactly 10.00% (engages) and 9.99% (does not). Unusable
   equity (non-finite or non-positive) engages it.
4. **Response, in fail-safe order.** Persist the latch (equity, high-water mark, drawdown, flows
   applied, account id, positions, time); engage the global halt with reason
   `live_pause:<latch id>`; audit `live_pause_engaged`; cancel every resting order on the live
   account using cancel-only credentials. Never close or offset a position. A failed cancel leaves
   latch and halt in place, reports `cancel_failed`, and is retried idempotently at every later
   evaluation while latched.
5. **Where it runs.** Every `live run-all` cycle with a `live` strategy or an existing latch
   evaluates the pause before it returns: immediately after the existing book breaker when the cycle
   reaches it, so that breaker's precedence and action are unchanged, and otherwise just before the
   cycle's early return (no allocated strategy, reconcile deferred or halted). Without a trading
   broker it uses read-only and cancel-only credentials. A standalone `live pause check` (no
   strategy needed) evaluates it on demand.
6. **Blocks new orders everywhere.** While the global halt is engaged, `live run-all` stops at its
   halt check as today. If the halt has been cleared but the latch remains (for example after
   `paper resume-all`), it stops right after that check, retries any failed cancel, performs no
   strategy tick, book-breaker liquidation or submission, and exits non-zero with `live_paused`.
   Each live order path's mid-tick halt check reads the latch, so a pause during a tick stops the
   remaining orders. `live allocate` refuses increases. `live flatten` remains available as the
   existing explicit risk-reducing command.
7. **Visibility.** `fleet status` shows the latch and its trigger figures; `fleet health` exits
   non-zero while it is latched; `live pause status` prints it.
8. **Drawdown report.** `live pause report` builds and stores an immutable report for the active
   latch: trigger record, high-water history and flows applied, equity at trigger and now, positions
   at trigger and now (read-only broker), orders cancelled, per-strategy ledger positions, and the
   SHA-256 digest of its canonical form.
9. **Signed resume.** `live pause resume` step 1 prints the stored report and a single-use challenge
   (the Story 2.5 account-level signed command, purpose `live_pause_resume`) binding latch id,
   report id and digest, and live account id. Step 2 verifies the signature over the stored report,
   then in one transaction consumes the challenge, appends the resume record (principal, report
   digest, equity at completion), re-bases the high-water mark to current equity and closes the
   latch. It clears the global halt only if that row is still this pause's engagement; otherwise it
   reports the halt still engaged. A bare `--actor human` or an agent cannot resume.
10. **Nothing else lifts it.** Tests prove `paper resume-all`, `global_halt.clear` and
    `rebase_all_peaks` leave the latch and the high-water mark unchanged, and that a resumed account
    re-triggers only after a new 10% fall from the re-based peak.
11. **Protected and green.** The 10% threshold is a constant in a CODEOWNERS-protected module;
    latch, flow, report and resume records are append-only (only the latch's one-way close is
    allowed); full gate passes; every branch mutation-checked; protected review.

## Tasks / subtasks

- [ ] Contract and readiness: DDL and triggers, evaluation order inside `live run-all`, report
      schema and digest, payload, entry-point inventory of every live order path.
- [ ] High-water record and flow accounting, test-first (AC1–AC3).
- [ ] Trigger response and idempotent cancel retry (AC4).
- [ ] Wiring in `live run-all`, the halt hook, `live allocate`; `live pause check` (AC5–AC6).
- [ ] Fleet visibility and `live pause status` (AC7).
- [ ] Report and signed resume (AC8–AC9); isolation from other resume paths (AC10).
- [ ] Protection, full gate, independent review (AC11).

## Dev notes

- Confirm Alpaca's sign convention for withdrawal `net_amount` against a recorded fixture or the
  official documentation before relying on it; an unexpected sign is "unvaluable", not zero.
- The cancel uses the existing cancel-only drain broker (`build_live_drain_broker`,
  `algua/execution/lane_exit.py:30-40`), which needs no per-strategy authorization. Cancel all
  resting orders on the account, not per strategy: the pause is account-wide.
- Positions at trigger come from the broker read in the same evaluation; store them in the latch so
  the report does not depend on later reads.
- The report the human signs must not change between step 1 and step 2: sign the stored report, not
  a rebuilt one, because market values move within the challenge's 10 minutes.
- `live_cmd.py` is carved by Story 2.4; put the pause in its own protected module and keep the
  command thin.

## Owner decisions

The 2026-10-04 decision settles the build. Two interactions are recorded for the owner, neither
blocking:

- **Existing book breaker.** The 15% drawdown and 5% daily-loss breaker still halts and closes every
  position when it fires in a cycle before the pause has latched (for example, an overnight gap past
  15%). Its peak is not cash-flow adjusted, so a withdrawal can trip it. Once the pause is latched
  the global halt short-circuits later cycles, so the breaker does not run while paused. Decide
  whether the experimental account keeps, retunes or removes that liquidating breaker.
- **Paper stops too.** Engaging the global halt also stops the paper lane (existing `halt-all`
  semantics). Agents may clear the halt for paper through `paper resume-all`; that never lifts the
  pause. Decide whether that coupling is wanted.

`live flatten` stays available while paused as the existing explicit risk-reducing command; the
owner may restrict it.

## Verification

```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```

## References

- [#624 owner decisions](https://github.com/Lior-Nis/algua/issues/624); `docs/PRD.md` §§15, 19, 21
- `docs/vision-reconciliation.md` gap 1; #390 (book breaker); #109 (rebase-then-unhalt ordering)
- Todoist:
  [Enforce experimental capital envelope and 10% pause](https://app.todoist.com/app/task/enforce-experimental-capital-envelope-and-10-pause-6hfCrg6m7Pj76gXG)

## Dev Agent Record

### Agent Model Used

### Completion Notes

### File List
