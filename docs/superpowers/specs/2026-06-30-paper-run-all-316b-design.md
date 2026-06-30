# paper run-all: multi-tenant batch driver + scoped cancel (#316b) — design

**Status:** Draft (design approved in brainstorming; pending human review → implementation plan)
**Date:** 2026-06-30
**Issue:** #316 (the run-all half of the #316 split; #316a = #321, the per-strategy tick). Epic #318.
**Builds on:** #316a (#321/PR#322 — `_run_paper_strategy_tick`, `_paper_broker_net`), #313 (`paper_reconcile`), #314 (`build_paper_sizing_snapshot`), Slice 1 (`paper allocate`, Σ≤equity). #316a is unmerged → build branch stacks on it (or rebase onto main after #322 merges).

## 1. Problem

The autonomous operator needs a per-session batch that ticks every active paper strategy on the shared
account — the paper analog of `live run-all`. #316a made the per-strategy tick multi-tenant-correct
and extracted `_run_paper_strategy_tick`; this issue adds the batch driver that reuses it, plus the
scoped-cancel fix the batch requires.

## 2. The scoped-cancel fix (must land here)

#316a's `_run_paper_strategy_tick` breach handler calls account-wide `broker.cancel_open_orders()`
(accepted for single-strategy `trade-tick`). In a batch, that would cancel **sibling** strategies'
resting orders. So #316b:
- **Generalizes `owned_open_order_ids(conn, broker, strategy, *, kind=LedgerKind.LIVE)`** to read
  `_TABLES[kind].orders` (live caller unchanged; `kind=PAPER` reads `paper_venue_orders`).
- Adds **`_paper_scoped_cancel(conn, broker, name)`** (local to `paper_cmd.py`; cli→cli imports are
  banned) — cancels only this strategy's open orders, mirror of live's `_scoped_cancel`.
- **Hardens `_run_paper_strategy_tick`**: add the `reserve_buy` hook to its `TickHooks`; in the breach
  handler, cancel via the supplied `cancel` callable when present
  (`cancel() if cancel is not None else broker.cancel_open_orders()`) — so run-all scopes the cancel
  while single-strategy `trade-tick` (cancel=None) keeps the safe account-wide cancel.

## 3. `paper run-all` (mirror `live run-all`)

```
load active Stage.PAPER strategies
  → filter is_due (Slice 1; today all daily/XNYS are due)
  → ingest venue fills ONCE (_ingest_paper_venue)
  → account reconcile ONCE: cycle = paper_reconcile.next_cycle(conn);
      recon = paper_reconcile.reconcile(conn, _paper_broker_net(broker), cycle)
      recon.halt  → global_halt.engage + exit 1
      not clean   → defer the whole cycle (emit, no trades), return
      clean       ↓
  → reservation pool: pool = {"available": acct.buying_power};
      _paper_reserve_for(name) caps each BUY's notional to remaining pool (mirror live _reserve_for)
  → for each due strategy:
        _run_paper_strategy_tick(conn, name, ..., tick_ts, clock_source, acct,
                                 reserve_buy=_paper_reserve_for(name),
                                 cancel=lambda n=name: _paper_scoped_cancel(conn, broker, n))
        append result; if result["ok"] is False (breach/halt) → stop, keep prior results
  → emit one envelope: {"reconcile": ..., "strategies": [...]}; exit non-zero if any breached
```

The single ingest + single reconcile + shared `acct`/`tick_ts`/`clock_source` are resolved once and
passed to each strategy's tick — exactly the boundary #316a's helper signature was built for.

## 4. Components

- **`algua/execution/live_ledger.py`** — generalize `owned_open_order_ids(..., *, kind=LedgerKind.LIVE)`.
- **`algua/cli/paper_cmd.py`** —
  - `_paper_scoped_cancel(conn, broker, name)`;
  - harden `_run_paper_strategy_tick` (`reserve_buy` hook; scoped cancel in the breach handler via the `cancel` callable);
  - `_paper_reserve_for` + pool;
  - `paper run-all` command.

## 5. Testing

- **`tests/test_cli_paper.py` / a new `tests/test_paper_run_all.py`** (reusing the #316a fake paper
  broker + allocation seeding):
  - two allocated paper strategies both tick, each sized off **its own** allocation/NAV;
  - a breach in one **trips + scoped-flattens only that one**; the sibling still ticks; the envelope
    surfaces both and exits non-zero (live #270 parity);
  - a not-clean account reconcile **defers the whole cycle** (no strategy trades);
  - the reservation pool caps concurrent BUY notional to available buying power (a second strategy's
    buy is trimmed/skipped when the pool is exhausted);
  - **scoped cancel:** a breach under run-all cancels only the breaching strategy's open orders — a
    sibling's resting order is untouched.
- **`owned_open_order_ids(kind=PAPER)`** returns the strategy's own `paper_venue_orders` open ids; the
  live variant (default kind) is unchanged.
- **Regression:** live `run-all` / `_run_strategy_tick` and single-strategy `trade-tick` unchanged.
- TDD; `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.

## 6. Non-goals (this issue)

- The forward-gate relaxation (`single_tenant` → attribution-clean) → #315.
- The autonomous operator's paper intake (allocate candidates into the book) → #317.
- No `run_tick` body change; no schema change; no rename of `live_snapshot`/`live_positions`/
  `LiveSizingError` (deferred).

## 7. Risk

- **Batch over the order path.** Mitigated: reuses #316a's reviewed `_run_paper_strategy_tick` and
  #313's reconcile; mirrors the proven `live run-all` structure (reservation pool, scoped cancel,
  one-envelope breach surfacing).
- **The scoped-cancel hardening changes #316a's helper.** The single-strategy `trade-tick` path keeps
  account-wide cancel (cancel=None), so its behavior is unchanged; only run-all passes the scoped
  cancel. Covered by both the existing single-strategy breach test and the new run-all scoped test.
