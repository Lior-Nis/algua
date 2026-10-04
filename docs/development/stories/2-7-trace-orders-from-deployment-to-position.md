---
baseline_commit: e875b6d
---

# Story 2.7: Trace every order from its deployment to the resulting position

Status: backlog

Prepared: 2026-10-04. Baseline: `e875b6d` (main after PR #686). Epic: 2.
Requirements: FR12 (broker and position traceability), FR13 (restart evidence), NFR2, NFR6, NFR8.
PRD §15.
Depends on: Story 2.4 (live frozen ticks carry deployment and invocation links). Sequenced after
Story 2.6 because it edits the same live order path and the next schema version.
Gated by: Story 2.6 merged, then contract and readiness review. No owner decision is needed.

## Story

As Algua's operator,
I want every paper and live order to carry a durable, immutable record linking it to its deployment,
inputs, decision, sizing, risk decision, broker order, fills and resulting position,
so that any position can be explained end to end, and a crash can never leave an order the books do
not know about.

## Context

PRD §15 requires each live order to be traceable through "strategy artifact → inputs → signal →
sizing → risk decision → order intent → broker execution → resulting position". Epic 1 linked ticks
to deployments and frozen invocations. The order end of the chain is thin:

- `live_orders` and `paper_venue_orders` (`algua/registry/db/execution.py:90-104`, `:136-150`)
  record strategy, symbol, side, an optional intended notional, the client order id, the broker
  order id, a status that is never updated, and the submit time. Neither records the deployment,
  invocation, bars snapshot, cycle, decision time, target weight, sizing or risk decision.
- The live lane records an order only after the broker accepts it, and passes `None` for the
  intended notional (`algua/cli/live_cmd.py:180-185`). A crash between the POST and that write
  leaves an order at the broker the ledger has never seen. The paper lane records intent before the
  POST (#249) and retracts it on a no-op (#311) through `before_submit` and `on_noop`
  (`algua/live/live_loop.py:356-379`).
- Sizing happens inside `submit_sized` (`algua/execution/alpaca_broker.py`), which returns only an
  order id. The reservation hook trims notional (`live_cmd.py:572-589`) and records only shortfalls
  (`live_reservations`, `execution.py:201-210`).
- The tick snapshot is written after the orders (`live_cmd.py:254-270`); the Phase B invocation id
  is known before submission (`algua/cli/paper_cmd.py:771`, `port.final_invocation_id`).
- Fills join orders by broker order id (`algua/execution/live_ledger.py:543-575`).
- `alpaca_broker.py` (547) and `live_ledger.py` (620) sit exactly at their size pins.

## Scope and authority

In scope: an append-only order trace in both lanes, written before each POST; crash-safe live intent
recording at parity with paper; one sizing computation exposed to the trace; a read-only trace
command and a read-only trace audit.

Must not: change any sizing, risk, reservation or submission decision; change order ids or
idempotency; store credentials or raw broker payloads beyond the existing ledgers; rewrite
historical orders (pre-story orders report `pre_trace`). No authority change.

## Acceptance criteria

1. **Trace before the POST, both lanes.** Before each broker POST an immutable trace row is written,
   keyed by client order id, with: lane, strategy id, deployment id, final frozen invocation id
   (frozen ticks), tick request id, bars snapshot id, cycle number, decision time, symbol, side,
   target weight, sizing (equity denominator, current market value, delta notional and quantity),
   risk decision (requested notional, permitted notional after the pool and book trims, the reason),
   and time.
2. **Live intent is crash-safe.** The live order row is written before the POST and retracted only
   if no POST happened, matching the paper lane's #249 and #311 behavior. A test kills the process
   between POST and backfill; on restart the order is recovered and its fills attribute.
3. **One sizing computation.** The sizing and trim values recorded are the values the broker adapter
   used, exposed by the adapter rather than recomputed. A test perturbs the sizer and sees the trace
   follow.
4. **Chain to the position.** Broker order ids are backfilled; fills join by broker order id; the
   trace resolves the believed position after the order's fills and the next tick snapshot of the
   same deployment.
5. **`trace order`.** `trace order <client-or-broker-order-id>` (read-only JSON) returns the chain:
   deployment and artifact (manifest, bundle and environment digests), invocation (request and
   result digests), snapshot, intent, sizing, risk decision, order, fills and position. Each missing
   link is a stable code (`trace_link_missing:<link>`), and an incomplete chain exits non-zero.
6. **`trace audit`.** `trace audit --since <ts>` lists every order in the window with a broken chain
   and exits non-zero if any exists. Pre-story orders report `pre_trace` and do not fail the audit.
7. **Immutable and private.** Trace rows are append-only (triggers refuse update and delete) and
   hold no credential, account secret or raw broker payload.
8. **Parity and structure.** Both lanes write the same trace shape through one shared writer; the
   lane parity test covers it. Pinned modules are carved, not grown. Full gate passes; protected
   review.

## Tasks / subtasks

- [ ] Contract and readiness: DDL and triggers, field list, link codes, entry-point inventory of
      every submit path (live and paper `run-all`, paper `trade-tick`, both flatten paths, the book
      flatten).
- [ ] Adapter exposes its sizing and trims; carve `alpaca_broker.py` first (AC3).
- [ ] Shared trace writer; live pre-POST intent and no-op retraction (AC1–AC2).
- [ ] Wire both lanes; extend lane parity (AC8).
- [ ] `trace order` and `trace audit` (AC4–AC6); immutability tests (AC7).
- [ ] Crash and restart test; full gate; independent review.

## Dev notes

- Flatten offsets and the whole-account book flatten (`broker.close_all_positions`,
  `live_cmd.py:524`) are orders too. The contract must say whether each gets a trace row; the
  account close returns no per-order ids, so the audit should report it as a distinct `book_flatten`
  link, not as a broken chain.
- Use the tick's Phase A request id (`algua/live/live_loop.py:283`) as the key from an order to its
  invocations; a frozen tick already reaches its snapshot through `frozen_invocation_id`. Only
  frozen tenants can tick once Story 1.4's exit completes, so the contract need not invent a
  request-id link for in-process ticks; it reports them as `non_frozen`.
- Keep the trace writer in one module both lanes import; the live helper module from Story 2.4 is
  the live call site.

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

- `docs/PRD.md` §15; [epics.md](../epics.md) FR12
- #249 (crash-safe paper intent), #311 (no-op retraction), #312 (stranded order recovery)
- [Story 1.3d](1-3d-bind-operational-evidence-and-qualification.md) (invocation evidence and tick
  link)
- Todoist:
  [Complete live traceability and safe release validation](https://app.todoist.com/app/task/complete-live-traceability-and-safe-release-validation-6hfCrg6qX8q4fvgG)

## Dev Agent Record

### Agent Model Used

### Completion Notes

### File List
