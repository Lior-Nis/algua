# paper run-all — multi-tenant batch driver (#316)

> **Scope note (read first).** This document describes the driver that ACTUALLY SHIPPED under #316:
> a simple, long-only, break-on-any-breach batch cycle with a whole-account buying-power pool. An
> earlier draft of this spec described a much larger design (durable reservation-shortfall taint +
> forward-gate exclusion, a paper-lane concurrency lock, per-strategy class-A/B breach continuation,
> explicit long-only guards, per-cycle fairness rotation). Those were NOT built here; they are
> tracked as follow-ups (**#492**, **#493**, **#494**) and summarised in *Deferred follow-ups*
> below. Everything in the *What shipped* section corresponds to code in this PR's diff.

## Why this is not already done
PR#323 (#316b) is shown MERGED, but its merge commit landed on the stacked base branch
`paper-multitenant-tick-316a`, which was then squash-merged to main as #316a WITHOUT the later
#316b commits. So `paper run-all` is genuinely absent from origin/main; only the #316a per-strategy
tick rework landed. This change re-adds the command, re-adapted to the drifted origin/main (notably
the #336 single-sourced `flatten_strategy` breach handler).

## Goal
`paper run-all` — one sequenced cycle over ALL paper-lane strategies: ingest venue fills once,
recover crash-stranded orders, reconcile the account, then tick each strategy (sized off its own
NAV via the #314 snapshot), with a shared whole-account buying-power pool and a scoped cancel on a
breach so one strategy never cancels a sibling's resting orders. Mirrors `live run-all`.

---

## What shipped

### Strategy selection — `stage IN ('paper','forward_tested')`
run-all ticks BOTH `paper` AND `forward_tested` strategies. This is deliberate parity with
`load_gated_strategy` (which admits PAPER and FORWARD_TESTED) and single-strategy `paper trade-tick`:
a `forward_tested` strategy keeps paper-ticking while awaiting the go-live signature so its live-wall
certificate stays fresh (#124). Selection is `list_strategies(PAPER) + list_strategies(FORWARD_TESTED)`,
merged preserving each list's insertion order — NOT a `stage='paper'` filter. The docstring states
this so the naming ("paper" run-all) does not imply a paper-only stage filter. (Test:
`test_run_all_ticks_forward_tested_strategies` — a forced-`forward_tested` strategy is ticked, not
dropped.)

### Cycle body (mirrors `paper trade-tick` and `live run-all`)
Wrapped in the observability triple — `configure_logging()` + `CycleCounters` + `correlation_context()`
with a `finally` `golden_signals` flush so the rollup survives a mid-cycle failure. Sequence:
1. **Empty guard** — no paper-lane strategies ⇒ emit an empty envelope and return.
2. **Global-halt pre-check** — `global_halt.is_engaged(conn)` ⇒ emit a breach payload, exit non-zero.
3. **Ingest + recover** — `_ingest_paper_venue` then `_recover_stranded`, BEFORE reconcile, inside a
   `try/except` that FAILS CLOSED on any transport/venue error (`venue_ingest_failed`, exit non-zero)
   so a partial ingest can never read as reconcile drift.
4. **Account-wide reconcile** — `paper_reconcile.next_cycle` + `reconcile(conn, _paper_broker_net(broker),
   cycle)`: `recon.halt` ⇒ engage the global halt + emit + exit non-zero; `not recon.clean` ⇒ defer the
   WHOLE cycle (no strategy trades this cycle, exit 0); clean ⇒ tick. (Test:
   `test_run_all_defers_whole_cycle_on_unreconciled_account`.)
5. **Tick loop** — over the selected strategies in registry-id order, each via `_run_paper_strategy_tick`
   with a per-strategy `reserve_buy` closure and a per-strategy scoped-cancel closure.

### Whole-account buying-power pool
One shared `pool = {"available": float(acct.buying_power)}` built before the loop. Each strategy gets a
closure `_paper_reserve_for(name)` threaded to `TickHooks.reserve_buy`; `run_tick` calls it ONLY for
BUY orders (`submit_sized(intent, snap, coid, reserve=hooks.reserve_buy)`). The closure:
```python
def _reserve(symbol, notional):
    grant = min(notional, max(0.0, pool["available"]))
    pool["available"] -= posted_notional(grant)   # debit ONLY what actually posts (finding-1/3 fix)
    if grant < notional:                           # the POOL bound this buy -> audit the shortfall
        audit_append(..., action="paper_reserve_trim", reason=f"{symbol} {notional}->{grant}", ...)
    return grant
```
The pool is therefore the FINAL cap on the aggregate of this cycle's buys: a strategy is sized off its
own NAV (the intended notional), and `reserve_buy` trims that to the shared pool's remaining balance.

**Debit-by-`posted_notional` (the finding-1/finding-3 correctness fix).** `posted_notional(grant)`
(new pure helper in `alpaca_broker.py`) returns EXACTLY the notional a BUY of `grant` dollars actually
posts through `submit_sized`: floor to cents (`ROUND_FLOOR`, never up) then `0.0` if that falls below
`MIN_NOTIONAL` (`submit_sized` returns `"skipped"` and posts nothing). The pool debits by this value,
NOT the raw `grant`, so a buy that skips or floors out never phantom-consumes buying power and wrongly
starves a later sibling. Invariant: `pool["available"] == start_BP − Σ(real posted buy notionals)`.
`submit_sized`'s own floor/skip logic is the single source `posted_notional` mirrors; a unit test
(`test_posted_notional_matches_submit_sized`) asserts the two agree on a fractional grant (floors to
cents) and a sub-`MIN_NOTIONAL` grant (skips, posts nothing). Pool tests:
`test_run_all_reservation_pool_caps_concurrent_buys`,
`test_run_all_pool_does_not_phantom_debit_skipped_buys`. Sells are never reserved (pool untouched).

`paper trade-tick` passes no `reserve_buy` (default `None`): no pool cap, unchanged.

### Scoped cancel — `owned_open_order_ids` gains a `kind` selector
`owned_open_order_ids(conn, broker, strategy, *, kind: LedgerKind = LedgerKind.LIVE)` now selects the
order ledger via `kind` (`live_orders` / `paper_venue_orders`) instead of hardcoding `live_orders`. The
paper scoped-cancel helper `_paper_scoped_cancel(conn, broker, name)` passes `kind=LedgerKind.PAPER`
explicitly and cancels only THIS strategy's open PAPER orders. `kind` keeps a `LedgerKind.LIVE` default
so the existing live caller (`live_cmd._scoped_cancel`) is unchanged; every paper caller passes it
explicitly. run-all threads a per-strategy `cancel=lambda: _paper_scoped_cancel(conn, broker, name)`
into `_run_paper_strategy_tick`; its breach path uses `cancel if cancel is not None else
broker.cancel_open_orders` (account-wide fallback for single-strategy trade-tick), so a breach never
cancels a sibling's resting orders. (Tests: `test_run_all_breach_scoped_flatten_surfaces_siblings`,
`tests/test_live_ledger_ledgerkind.py`.)

### Default-ON drawdown breaker + rolling wall-clock window (GATE-2 parity fix)
`run-all` was shipping with the SAME literal defaults as the original single-strategy CLI it mirrors
pre-#390/#452 (`--start 2023-01-01 --end 2023-12-31`, and an omitted `--max-drawdown` left the breaker
OFF) instead of the parity these two commands already established for `paper trade-tick` and `live
run-all`. Both are now threaded through the shared `algua/cli/_common.py` helpers so the three CLIs can
never drift apart:
- `--max-drawdown` defaults to `None` and `--disable-drawdown-breaker` (human-only, audited) is added;
  `resolve_drawdown_breaker` resolves an omitted flag to the default-ON `strategy_max_drawdown_default`
  bound rather than silently disabling the breaker (#390 parity). An explicit `--disable-drawdown-breaker`
  is audited (`drawdown_breaker_disabled`) inside the same connection used for the rest of the cycle.
- `--start`/`--end` default to `None` and `resolve_wall_clock_window` fills them with a recent rolling
  window (`end=today UTC`, `start=today-400d`) instead of the frozen 2023 literal (#452 parity), so a
  default invocation sizes/risk-checks against current data.
(Tests: `test_run_all_omitted_max_drawdown_uses_default_bound`,
`test_run_all_omitted_window_resolves_to_rolling_today`.)

### Breach handling — break on any `ok: False` marker (stop-the-world)
The tick loop `break`s on the FIRST strategy that returns an `ok: False` marker (a breach/halt), the
same conservative posture as `live run-all`. The breaching strategy has already tripped its kill-switch
and been scoped-flattened inside `_run_paper_strategy_tick` (via the #336 `flatten_strategy` handler).
After the loop, if any strategy breached, the top-level envelope is emitted with `ok: False` and the
command exits non-zero (#270 parity: surface the breaching strategy AND every sibling ticked before it
in one envelope; the prior results are not discarded). Per-strategy breach CONTINUATION (letting a
scoped economic breach not block unrelated siblings) is a deferred refinement — see **#494**.

---

## Concurrency — mutual exclusion by operator discipline (for now)
`paper run-all`, `paper trade-tick`, and the paper liquidation/recovery commands each ingest venue
fills and mutate the one shared paper account. Run concurrently they double-ingest, race the reconcile
cursor, and each read the full broker buying-power into their own in-memory pool (cross-process
over-commit). They are currently **mutually-exclusive BY OPERATOR DISCIPLINE only**; `run_all`'s
docstring states this explicitly. An advisory host-local `paper_cycle_lock` (fcntl.flock,
engage-then-lock for the emergency liquidators) that ENFORCES this is a deferred follow-up — see
**#493**.

---

## Files changed (this PR)
- `algua/execution/alpaca_broker.py` — add the pure `posted_notional(grant)` helper (the pool debits
  by it so it can never diverge from what `submit_sized` posts). NOT CODEOWNERS-protected.
- `algua/execution/live_ledger.py` — `owned_open_order_ids` gains `*, kind: LedgerKind =
  LedgerKind.LIVE`, selecting the order ledger table. NOT CODEOWNERS-protected.
- `algua/cli/paper_cmd.py` — add the `run-all` command (selection over both paper-lane stages, the
  observability wrapper, ingest/recover/reconcile, the shared BP pool + `_paper_reserve_for` closure,
  the per-strategy scoped-cancel + reserve wiring, the break-on-breach envelope); add
  `_paper_scoped_cancel`; add `reserve_buy` to `_run_paper_strategy_tick`'s hooks and parameterize its
  breach-path `cancel`. NOT CODEOWNERS-protected.
- Tests: `tests/test_paper_run_all.py`, `tests/test_alpaca_broker.py`,
  `tests/test_live_ledger_ledgerkind.py`, `tests/test_cli_paper.py`.

No CODEOWNERS-protected path is touched; no schema bump.

---

## Deferred follow-ups (filed, out of scope here)
- **#492 — evidence-integrity exclusion (promotion-safety).** The shared pool couples strategies: a
  trimmed strategy's realized paper returns become ordering/load-dependent and under-invested, and that
  contaminated series currently flows into `paper -> forward_tested` promotion evidence unfiltered. The
  deferred fix is a durable per-strategy reservation-shortfall state machine (`paper_reserve_shortfall`
  table, `clear/capped/pending`, written at submission time) projected onto a `tick_snapshots.reserve_trimmed`
  column the forward gate excludes (one SCHEMA_VERSION bump; `forward_promotion.py` is CODEOWNERS-protected).
  Until it lands, treat forward-gate PASSes earned under a contended run-all pool with suspicion.
- **#493 — advisory paper-lane concurrency lock.** Replace the operator-discipline note above with an
  enforced `paper_cycle_lock` (fcntl.flock, non-blocking fail-closed for the cycle drivers,
  engage-then-lock for the emergency liquidators).
- **#494 — breach continuation (class A/B) + long-only fail-closed guards + per-cycle fairness rotation.**
  Per-strategy breach isolation (an economic breach scoped-flattens without stopping siblings; only
  account-level conditions stop the cycle) behind a `global_halt_engaged` marker; explicit pre-submit +
  pre-flatten long-only guards (the shipped construction policies are all long-only, so no short can be
  produced today — this is defense-in-depth for a future short-enabled strategy); and a `cycle_id % n`
  tick-order rotation to de-bias which strategies get pool-trimmed.
- **Two-phase reserve (authorize/commit).** The pool debits in the closure BEFORE the broker POST. This
  is invariant-safe today because a POST failure raises `BrokerError`, which is NOT caught per-strategy
  and ABORTS the whole cycle non-zero (mirroring `trade-tick`) — so a phantom debit can never be observed
  by a later sibling. A full authorize-then-commit debit is required ONLY if submit failures ever become
  non-fatal (in-cycle retry); recorded as the explicit trigger for that follow-up.
- **Paper book-level risk wall (#389-equivalent).** Removes the shared-pool coupling at the source and
  would let trimmed ticks count again, superseding the #492 exclusion posture.

---

## Merge policy
This change touches no CODEOWNERS-protected path (`alpaca_broker.py`, `cli/paper_cmd.py`,
`execution/live_ledger.py`, tests, this doc). The deferred evidence-exclusion work (#492) is where the
CODEOWNERS-protected `forward_promotion.py` edit lives — that PR will require human merge; this one does
not, on its own merits.
