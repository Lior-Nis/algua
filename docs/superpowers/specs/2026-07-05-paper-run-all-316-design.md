# paper run-all — multi-tenant batch driver (#316)

## Why this is not already done
PR#323 (#316b) is shown MERGED, but its merge commit (9c154ba) landed on the STACKED base branch
`paper-multitenant-tick-316a`, which was then SQUASH-merged to main as #316a (bfec476) WITHOUT the
later #316b commits. `git merge-base --is-ancestor 9c154ba origin/main` = false. So `paper run-all`
is genuinely absent from origin/main; only the #316a per-strategy tick rework landed. The reviewed
#316b work survives in commits 6d158de / 743d79c / 00d7bb9 and is the blueprint here, re-adapted to
the drifted origin/main (notably the #336 single-sourced `flatten_strategy` breach handler).

## Goal
`paper run-all` — one sequenced cycle over ALL paper-lane strategies: ingest venue fills once,
recover crash-stranded orders, reconcile the account (attributed), then tick each strategy (sized
off its own NAV via the #314 snapshot), with a shared buying-power reservation pool and
per-strategy breach isolation (scoped cancel + scoped flatten). Mirrors `live run-all`.

## Strategy selection (was implicit — now stated: GATE-1 re-run non-blocking note N3)
run-all ticks BOTH `paper` AND `forward_tested` strategies. This is deliberate parity with
`load_gated_strategy` (which admits PAPER and FORWARD_TESTED) and with single-strategy
`paper trade-tick`: a `forward_tested` strategy keeps paper-ticking while awaiting the go-live
signature so its live-wall certificate stays fresh (#124). Selection query is therefore
`stage IN ('paper','forward_tested')`, NOT `stage='paper'`. `run-all`'s docstring states this
explicitly so the naming ("paper" run-all) does not imply a paper-only stage filter.

---

## Design decisions (forks resolved)
1. **No book-level risk (#389) in paper — but the shared-pool coupling is ENFORCED out of promotion
   evidence, not merely documented (finding 4).** The issue predates #389; `_build_book_exposure`
   is live_cmd-local (cli->cli wall); the reviewed #316b used a simple buying-power-only pool. Keep
   the simple BP pool for THIS change, but the absence of a book-level wall is not free: because
   run-all couples strategies through the shared BP pool (an early strategy's buys can trim a later
   strategy's buys via `reserve_buy`), a TRIMMED strategy's realized paper return series produced
   under run-all is ordering- and sibling-load-dependent, AND the distortion PERSISTS across the
   ticks whose NAV the under-investment actually contaminates — not just the one tick where the cap
   occurred. This is closed by a REAL enforcement path built on DURABLE reservation state (not a
   process-local flag, not a doc note): a durable per-strategy reservation-shortfall record marks
   every tick from a cap through the strategy's demonstrable reconvergence to intended sizing, and
   the forward gate EXCLUDES those ticks from admissible evidence. See **§ Finding 4 (blocking) —
   durable, multi-tick enforced evidence exclusion**. A book-level paper risk wall is deferred
   follow-up (#389-equivalent), out of scope here.
1a. **Per-cycle strategy-order rotation (fairness — GATE-1 re-run non-blocking note N1).** Evidence
   exclusion is inherently order-dependent: strategies ticked EARLIER in the loop get first claim on
   the shared BP pool and are less likely to be trimmed, so a fixed order would systematically bias
   WHICH strategies clear the ≥63-observation floor. run-all rotates the per-cycle tick order by a
   deterministic rotating offset (`cycle_id % n`) over a stable base ordering (registry id), so over
   many cycles no strategy is permanently first and pool-trim pressure is spread evenly. This is a
   zero-cost de-bias, not a full fix; the deferred paper book-risk wall removes the coupling at the
   source. Rotation is documented and asserted by a test (offset advances with cycle id).
1b. **Concurrency exclusion (finding 5).** `paper trade-tick`, `paper run-all`, and the paper
   liquidation/recovery commands (`flatten`, `halt-all`, `resume-all`) all ingest venue fills and/or
   mutate the shared paper account + ledgers; run concurrently they double-ingest, race the reconcile
   cycle, and each cycle driver reads the full broker buying-power into its own in-memory pool
   (cross-process over-commit). An advisory paper-lane lock serializes them, with emergency-correct
   acquisition semantics so an emergency STOP is never blocked. See **§ Finding 5**.
1c. **Long-only constraint (finding 6), enforced on BOTH the normal order path AND the breach path.**
   The class-B pool-safety proof depends on buys being the only pool-consuming order and on a scoped
   flatten only ever SELLING. A short position violates both (opening a short is an unreserved sell
   whose margin the pool never models; covering a short is an unreserved BUY-to-cover). The shipped
   construction policies are long-only; this change makes long-only an explicit, fail-closed guard at
   TWO points: a pre-submit guard in the normal order path (rejects a short-producing intent BEFORE
   any order posts) and a pre-flatten guard on the breach path (rejects a believed short before a
   buy-to-cover). See **§ Finding 6**.
2. **Include the observability wrapper.** Current `live run-all` and `paper trade-tick` both wrap the
   cycle in `configure_logging()` + `CycleCounters` + `correlation_context()` + a `finally`
   golden_signals flush. Mirror that for parity (the reviewed #316b predated full adoption).
3. **Scoped cancel via the #336 handler.** `_run_paper_strategy_tick`'s breach path now calls
   `flatten_strategy(..., cancel=broker.cancel_open_orders)`. Parameterize: pass
   `cancel if cancel is not None else broker.cancel_open_orders` so single-strategy trade-tick keeps
   account-wide cancel and run-all uses the scoped callable — never cancel a sibling's orders.
4. **Two separate signals, do not conflate them.** (a) A **reservation-trim audit**
   (`audit_append action="paper_reserve_trim"`) is OBSERVABILITY-ONLY forensics of a shortfall; it
   enforces NOTHING and is emitted BEST-EFFORT (wrapped so a raising audit sink can neither abort the
   cycle nor desync the pool after the debit/durable-mark already applied — GATE-1 round-3 HIGH). BP
   enforcement is the in-memory `pool["available"]` debit + `submit_sized`'s floor-to-cents/MIN_NOTIONAL
   decision, per findings 1 & 3 — a test proves this holds with the audit sink no-op'd AND with it
   patched to RAISE. (b) The **promotion-evidence taint is DURABLE reservation-shortfall state** written
   at order-submission time and projected onto a per-tick `tick_snapshots.reserve_trimmed` column the
   forward gate reads to EXCLUDE tainted ticks (findings 2 & 4). The gate never parses the free-text
   audit `reason`; enforcement is keyed to exact admissible tick rows.
5. **Same reconcile call as trade-tick** (`paper_reconcile.reconcile(conn, _paper_broker_net(broker),
   cycle)` with defaults); no `--grace-cycles`/`--tolerance` CLI options.

---

## Finding 1 (blocking) — exact `reserve_buy` wiring contract + real-submission pool debit

**Problem the finding raises:** the design must state, and WIRE, that the shared BP pool's
`reserve_buy` is the FINAL cap on every buy's notional — not a hint layered on top of NAV sizing.
Today `_run_paper_strategy_tick` builds its `TickHooks` WITHOUT `reserve_buy` (paper_cmd.py
~L396-407), so as written run-all would size each strategy off its own NAV and submit the full
NAV-sized notional with no shared-pool cap at all. That is the bug this finding forbids.

**The contract (must be stated in the design and enforced in code):**

- **Per-strategy NAV sizing produces the INTENDED notional.** Inside `run_tick`, each strategy is
  sized off its own NAV snapshot (`build_paper_sizing_snapshot`, #314). For one BUY intent,
  `broker.submit_sized` computes `amount = abs(sized.delta_notional)` — the NAV-intended buy
  notional (alpaca_broker.py L283-289).
- **`reserve_buy` is the FINAL cap applied immediately before order construction.** In
  `submit_sized`, for a BUY with a reserve hook (L290-298):
  `permitted = reserve(intent.symbol, amount)`; `permitted <= 0` ⇒ return `"skipped"` (no POST);
  else `amount = min(amount, permitted)`; the submitted notional is `posted_notional(amount)` — see
  the shared helper below — and a sub-`MIN_NOTIONAL` result ⇒ `"skipped"`. **Therefore every buy
  order's actual submitted notional is `posted_notional(min(NAV_amount, reserve_grant))`, NEVER the
  raw NAV-sized value.** Sells are never reserved (pool untouched).
- **The hook is threaded exactly as live's.** `run_tick` calls
  `broker.submit_sized(intent, snap, coid, reserve=hooks.reserve_buy)` (live_loop.py L382). So the
  wiring obligation is: **populate `TickHooks.reserve_buy` in `_run_paper_strategy_tick`.**

**`posted_notional` — single source of truth for "what actually posts" (finding 3 dependency).**
Extract the floor-to-cents + MIN_NOTIONAL skip decision that `submit_sized` already performs into a
pure helper in `alpaca_broker.py`:

```python
def posted_notional(grant: float) -> float:
    """The notional a BUY of `grant` dollars ACTUALLY posts: floor to cents (ROUND_FLOOR, never up),
    then 0.0 if that falls below MIN_NOTIONAL (submit_sized returns 'skipped'). Single source of
    truth shared by submit_sized AND the reserve-pool debit so the pool can never diverge from the
    real submitted amount (finding 3)."""
    floored = float(Decimal(str(grant)).quantize(Decimal("0.01"), rounding=ROUND_FLOOR))
    return 0.0 if floored < MIN_NOTIONAL else floored
```

`submit_sized` is refactored to compute its submitted notional via `posted_notional(amount)` (and
return `"skipped"` when it is 0.0) rather than inlining the quantize/compare — behaviour-preserving,
but now the SAME function the pool uses.

**TickHooks wiring — the shared pool, the closure, and the REAL-submission debit (finding 3).**
`run-all` builds ONE shared pool and a per-strategy closure and passes the closure INTO the tick so
it reaches `TickHooks.reserve_buy`. Crucially, the closure debits the pool by the notional that will
ACTUALLY post (`posted_notional`), NOT by the raw grant — so the "pool decremented only by real
submitted buys" invariant finding 3 depends on is TRUE (see § Finding 3):

```python
# in run-all, before the per-strategy loop:
pool = {"available": _paper_broker_buying_power(broker)}   # float(broker.account().buying_power)

def _reserve_for(strategy_name: str, strategy_id_: int):
    def _reserve(symbol: str, intended_notional: float) -> float:
        # FINAL cap: trim the NAV-intended notional to the shared pool's remaining buying power.
        grant = min(intended_notional, max(0.0, pool["available"]))
        submitted = posted_notional(grant)          # EXACTLY what submit_sized will POST (0 if skipped)
        pool["available"] -= submitted               # ENFORCEMENT: debit ONLY the real submitted buy (finding 3)
        if intended_notional - grant > DEFAULT_TOLERANCE:   # the POOL bound this buy BEYOND tolerance
            # DURABLE taint at submission time (finding 2): commit the reservation shortfall NOW, in
            # this cycle's connection, so a crash/breach AFTER the capped POST cannot lose the taint
            # while the order's NAV effect survives. This is enforcement/durable state.
            mark_reserve_shortfall(conn, strategy_id_, cycle_id)   # durably advance state->capped, last_cap_cycle=cycle_id, commit
            # OBSERVABILITY-ONLY forensics, strictly BEST-EFFORT: a failing/raising audit sink must
            # never abort the cycle or desync the pool AFTER the debit + durable mark already applied
            # (GATE-1 round-3 HIGH). The enforcement path (pool debit + mark) contains no audit read.
            try:
                audit_append(conn, actor="system", action="paper_reserve_trim",   # forensics ONLY
                             reason=f"{symbol} {intended_notional}->{submitted}",
                             strategy=strategy_name)
            except Exception:   # noqa: BLE001 — audit is non-load-bearing; swallow to protect the invariant
                log.warning("paper_reserve_trim_audit_failed",
                            extra={"fields": {"strategy": strategy_name, "symbol": symbol}})
        return grant
    return _reserve

# per strategy (in rotated order, decision 1a):
out = _run_paper_strategy_tick(
    conn, name, strategy, rec, broker, provider, max_drawdown,
    tick_ts, clock_source, acct, cycle_id=cycle,
    cancel=lambda n=name: _paper_scoped_cancel(conn, broker, n),
    reserve_buy=_reserve_for(name, rec.id),   # <-- threaded to TickHooks.reserve_buy (id-keyed shortfall)
    start=start, end=end,
)
```

`_run_paper_strategy_tick` gains `reserve_buy=None` and `cycle_id=None` keyword params. It sets
`reserve_buy` on the hooks. It NO LONGER reads a process-local `trimmed` dict; the per-tick
`reserve_trimmed` column is derived from DURABLE state at snapshot time (finding 4).

**Why debiting in the closure (before the broker POST) is still invariant-safe (GATE-1 round-2
HIGH 3).** The closure debits `posted_notional(grant)` — computed from the SAME helper `submit_sized`
uses — so the two divergence cases the reviewer raised are already covered: (a) a buy that skips or
floors below `MIN_NOTIONAL` yields `posted_notional == 0.0`, so the pool is debited by exactly `0` —
no phantom debit; (b) the only remaining divergence is a broker POST that RAISES (transport/5xx/bad
response ⇒ `BrokerError`) or returns no id. `BrokerError` is NOT caught by `_run_paper_strategy_tick`'s
handlers (`TickHalted`/`RiskBreach`/`LiveSizingError`) NOR by run-all's per-strategy loop: it
propagates to the cycle body's `except Exception` and ABORTS the entire cycle non-zero (mirroring
`trade-tick` L563-573). Therefore a phantom debit for an order that failed to POST can NEVER be
observed by a subsequent sibling — the invariant `pool == start_BP − Σ(real submitted buy notionals)`
holds for every cycle that survives to tick another strategy, which is the only context in which the
pool is read. A full two-phase authorize/commit (debit only after a confirmed POST) is the correct
design ONLY IF submit failures ever become non-fatal (in-cycle retry); this is recorded as the
explicit trigger for the deferred two-phase follow-up. run-all deliberately does NOT catch
`BrokerError` per strategy for this reason.

Single-strategy `paper trade-tick` passes neither `cancel` nor `reserve_buy` (both default None):
account-wide cancel, no pool cap — unchanged. It DOES pass `cycle_id` (it already has `cycle`) so the
durable-taint machinery is uniform: with no `reserve_buy` it never MARKS a shortfall
(`capped_this_cycle`=False), but it still reads `state_before` and ADVANCES the machine, so it
correctly inherits and DRAINS (`capped→pending→clear`) any shortfall a prior `run-all` left for the
strategy — it does NOT reset the taint to 0 (GATE-1 round-5 HIGH).

**Test (finding 1):** a fake broker records every submitted notional; run-all with a pool smaller
than the summed NAV-intended buys asserts (a) each submitted buy's notional is `<= grant` and
strictly less than the NAV-intended amount when the pool bound it — assert `submitted <= grant`, NOT
exact equality: `posted_notional` (ROUND_FLOOR) and the sub-`MIN_NOTIONAL` skip can legitimately make
submitted strictly less than the grant, so an equality assert would be a false red; (b) once
`pool["available"]` reaches 0 the next strategy's buy is `"skipped"` (no POST); (c) the pool ends
decremented by EXACTLY the summed `posted_notional` of submitted buys — a buy that floors/skips does
not over-debit the pool (finding-3 invariant, asserted directly).

---

## Finding 2 (blocking) — `owned_open_order_ids` gets a REQUIRED keyword-only `kind`

**Decision:** make `kind` a **required keyword-only** parameter (no default). A forgotten `kind` is
then a `TypeError` at call time, not a silently-empty scoped-cancel that fails to cancel any orders
(the foot-gun the finding names). Rejected the `default=LedgerKind.LIVE` option (a paper caller that
forgets `kind` would read the LIVE orders table and cancel nothing in paper — silent) and the
split-into-two-functions option (more surface for no added safety over a required kw-only param).

```python
def owned_open_order_ids(
    conn: sqlite3.Connection, broker: OpenOrderReader, strategy: str, *, kind: LedgerKind
) -> list[str]:
    ...
    owned = {
        r["client_order_id"]
        for r in conn.execute(
            f"SELECT client_order_id FROM {orders_table(kind)} WHERE strategy = ?", (strategy,)
        )
    }
    return [o["id"] for o in open_orders if o.get("client_order_id") in owned]
```

- Add a tiny `orders_table(kind) -> str` accessor next to the existing `fills_table(kind)` (returns
  `_TABLES[kind].orders` — `live_orders` / `paper_venue_orders`), so the function no longer hardcodes
  `live_orders`.
- **Update every existing caller to pass `kind=` explicitly** (no default means the tree won't
  type-check/run otherwise). Before editing, run `rg "owned_open_order_ids\("` across the tree to
  enumerate EVERY call site (source + tests) so none is missed — a missed caller is a `TypeError`
  at runtime. Known sites today: `algua/cli/live_cmd.py::_scoped_cancel` → `kind=LedgerKind.LIVE`;
  `tests/test_live_ledger_orders.py::test_owned_open_order_ids_filters_to_strategy` → add
  `kind=LedgerKind.LIVE`. The `rg` sweep is a required checklist step, not advisory.
- Add the paper scoped-cancel helper `_paper_scoped_cancel(conn, broker, strategy)` in paper_cmd.py
  that calls `owned_open_order_ids(conn, broker, strategy, kind=LedgerKind.PAPER)` then
  `broker.cancel_order(oid)` per id.

**Test (finding 2):** `owned_open_order_ids(conn, broker, "s1")` (no `kind`) raises `TypeError`; with
`kind=LedgerKind.PAPER` it reads `paper_venue_orders`; with `kind=LedgerKind.LIVE` it reads
`live_orders`.

---

## Finding 3 (blocking) — breach-loop semantics + the corrected pool-safety proof

**Decision.** Two distinct classes of stop.

**A. Account-level conditions STOP the whole cycle for all remaining strategies.** These are:
- **dirty / halting reconcile** — checked ONCE before the per-strategy loop (mirrors live_cmd
  L482-508 and paper `trade-tick` L529-551): `recon.halt` ⇒ engage global halt + emit + exit(1);
  `not recon.clean` ⇒ defer (no strategy ticks this cycle).
- **global halt already engaged** — checked before the loop; the cycle emits and returns without
  ticking.
- **stale / unvaluable marks (systemic dark feed)** — raised mid-loop by ONE strategy but SYSTEMIC:
  all strategies share one bar provider, so a stale/absent/future/non-finite mark means the risk
  state cannot be trusted for the WHOLE account. `_run_paper_strategy_tick` already engages the
  global halt (paper_cmd.py L422-433) and returns a marker with `global_halt_engaged == True` (see
  the **marker contract** below) and `liquidation_submitted == False`. **run-all treats a returned
  `global_halt_engaged == True` marker as an account-level stop: it `break`s the loop immediately —
  it does NOT proceed to reserve/cancel/tick any remaining sibling this cycle.**
- **long-only violation (finding 6)** — a short-producing intent (normal path) or a believed short
  at breach-flatten time fails CLOSED by engaging the global halt and returning
  `global_halt_engaged == True`; run-all `break`s (class A). See § Finding 6.
- **global halt engaged EXTERNALLY mid-loop** (e.g. a `live halt-all`/killswitch from another actor,
  or a sibling's reconcile-drift halt) — the NEXT strategy's `run_tick` sees `should_halt()` True and
  raises `TickHalted` between cancel and submit. That TickHalted path ALSO sets
  `global_halt_engaged == True` (it re-reads `global_halt.is_engaged(conn)`), so run-all `break`s
  here too. A bare `halted=True` marker is NOT sufficient to distinguish a systemic global-halt stop
  from a per-strategy kill-switch trip.

**B. A single strategy's OWN economic / integrity breach does NOT block unrelated siblings.**
An economic/integrity breach (`drawdown`, `gross_exposure_realized`, `reconcile`,
`non_positive_equity`, …) or a `TickHalted` caused by THIS strategy's OWN kill-switch (with the
global halt NOT engaged) is handled INSIDE `_run_paper_strategy_tick`: it trips/uses the strategy's
kill-switch and performs a SCOPED cancel + SCOPED flatten of THIS strategy only, then returns an
`ok: False` marker with `global_halt_engaged == False`. **run-all records the marker and CONTINUES
ticking the remaining siblings this cycle.**

**Marker contract.** Every `ok: False` marker carries an explicit boolean `global_halt_engaged`,
derived from DB truth at return time — NOT inferred from `halted`/`kind`:
- **stale/unvaluable path** (calls `global_halt.engage`): returns `global_halt_engaged=True`.
- **long-only violation** (finding 6, engages global halt): returns `global_halt_engaged=True`.
- **`except TickHalted`**: `global_halt_engaged=global_halt.is_engaged(conn)`. Kill-switch-only halt
  ⇒ `False` (class B); externally-engaged global halt ⇒ `True` (class A).
- **`except RiskBreach` economic/integrity path** (scoped flatten): `global_halt_engaged=False`.

run-all's loop decision is exactly: `if out.get("ok") is False and out.get("global_halt_engaged"):
break` (class A) `else: continue` (class B). Defense-in-depth: run-all MAY additionally `break` on a
`global_halt.is_engaged(conn)` re-read after each tick — same single source of truth.

**Justification for B (the corrected "concrete account-state reason" proof).** The one piece of
shared mutable account state in the paper cycle is the BP reservation pool (`pool["available"]`). An
economic/integrity breach cannot corrupt it, GIVEN the finding-3 debit fix and the finding-6
long-only guard:
- The pool is decremented ONLY by `reserve_buy`, and — per the finding-1/finding-3 debit fix — ONLY
  by `posted_notional(...)`, i.e. by the notional a buy ACTUALLY posts (0 for a skipped/floored-out
  buy). It is NEVER decremented by the raw grant, so a skipped or sub-`MIN_NOTIONAL` buy leaves the
  pool untouched and the pool balance always equals `start_BP − Σ(real submitted buy notionals)`.
  **This is the invariant the old design broke** (it debited `permitted` before the floor/skip
  decision, over-counting the pool); the fix restores it, and finding 1's test asserts it directly.
- A breaching strategy's earlier real buys this tick legitimately consumed BP; that consumption is
  correct and must persist for siblings.
- The breach response is a scoped FLATTEN, which submits SELL offsets (long-only guaranteed by
  finding 6, so no unreserved buy-to-cover). Sells are never reserved and never touch the pool. So
  the flatten cannot desync the pool.
Therefore after a scoped breach the pool remains a truthful account of remaining buying power, and
siblings can safely continue against it. (Contrast: a stale-mark halt IS a concrete account-state
reason — risk state untrusted — so class A stops.)

**This DIVERGES from live run-all** (which `break`s on ANY `ok: False` marker). Deliberate and
paper-specific: live shares a real-money account where an operator wants stop-the-world on the first
breach; paper run-all is an UNATTENDED evidence-gathering driver whose value is giving every paper
strategy its fair independent tick, and where B is provably pool-safe. Documented and asserted.

**Envelope / exit semantics.** After the loop, if ANY strategy produced a breach/halt marker
(class A or B), the top-level envelope is emitted with `ok: False` and the command exits non-zero
(#270 parity: surface EVERY ticked sibling's result alongside the breach in one envelope).

**Tests (finding 3):**
- class A (systemic feed): a mid-loop `stale_marks` breach on strategy 1 of 3 engages global halt,
  returns `global_halt_engaged=True`, leaves strategies 2 and 3 UNticked, envelope `ok: False`, exit 1.
- class A (external global halt): global halt engaged EXTERNALLY between strategy 1 and 2; strategy 2's
  `run_tick` raises `TickHalted` via `should_halt`; marker `global_halt_engaged=True`; run-all `break`s
  so strategy 3 is UNticked. Envelope `ok: False`, exit 1.
- class B: a `drawdown` breach on strategy 1 of 3 scoped-flattens strategy 1 (marker
  `global_halt_engaged=False`) and STILL ticks strategies 2 and 3. Envelope `ok: False`, exit 1.
- class B vs A discriminator: a per-strategy kill-switch `TickHalted` with global halt NOT engaged
  returns `global_halt_engaged=False` and does NOT break.
- pool-safety under B (the corrected invariant): after strategy 1's economic breach, the pool is
  decremented by exactly strategy 1's real submitted buy `posted_notional`s (not the raw grants, not
  the flatten sells); strategy 2's `reserve_buy` sees that exact remaining balance.

---

## Finding 4 (blocking) — DURABLE, MULTI-TICK enforced evidence exclusion

**The contamination channel.** `paper -> forward_tested` evidence is assembled per strategy from
`tick_snapshots WHERE lane='paper' AND strategy_id=?`
(`forward_promotion.py::assemble_forward_evidence`, L205-210). `reserve_buy` can TRIM a later
strategy's NAV-intended buys when earlier siblings drew down the pool, so a TRIMMED strategy's
realized returns become dependent on sibling ordering and load. **Two properties the prior spec got
wrong, both fixed here:**

1. **The taint must span the ticks the cap actually distorts — not just the cap tick (Codex F1).**
   A capped buy leaves the strategy UNDER-INVESTED relative to its intended sizing. Its NAV — and
   therefore its daily return `r_t = equity_t/equity_{t-1} − 1` — stays distorted on every subsequent
   tick whose holding interval it under-invests, until the strategy demonstrably RECONVERGES to
   intended sizing. Tainting only the cap tick admits contaminated downstream returns into evidence.
2. **The taint must be DURABLE, written at submission time (Codex F2).** A process-local `trimmed`
   dict read only AFTER a successful `run_tick` return is lost by a crash or a breach-path early
   exit — while the capped order (already POSTed) survives and distorts NAV. The taint must be a
   committed DB write in the cycle's connection at the moment the cap is applied.

**Decision: durable per-strategy reservation-shortfall state → per-tick `reserve_trimmed` column →
forward-gate exclusion.**

**(i) Durable reservation-shortfall state machine (new table `paper_reserve_shortfall`).** A tiny
per-strategy state table (mirrors `strategy_peaks`), keyed by the STABLE `strategy_id` — NOT the
mutable name (GATE-1 round-3 MINOR) — so it aligns with the `strategy_id`-keyed evidence and survives
a rename: `paper_reserve_shortfall(strategy_id INTEGER PRIMARY KEY, state INTEGER NOT NULL DEFAULT 0,
last_cap_cycle INTEGER, updated_at TEXT)`. `state` is a THREE-value machine (NOT a bare boolean —
GATE-1 round-4 HIGH): `0 = clear`, `1 = capped` (a beyond-tolerance pool cap is outstanding),
`2 = pending` (the catch-up buy has been SUBMITTED in full but its async paper fill has not yet had an
ingest opportunity). Helpers in `order_state.py`:
- `mark_reserve_shortfall(conn, strategy_id, cycle_id)` — upsert `state=1 (capped),
  last_cap_cycle=cycle_id`, **commit**. Called by the reserve closure the instant a buy is pool-capped
  BEYOND `DEFAULT_TOLERANCE` (finding 1) — the durable, submission-time write (Codex F2). It durably
  advances state to `capped` so a crash between the capped POST and the snapshot cannot lose the taint.
- `reserve_shortfall_state(conn, strategy_id) -> (state: int, last_cap_cycle: int|None)` — read.
- `next_shortfall_state(state_before: int, capped_this_cycle: bool) -> int` — PURE transition (below),
  unit-testable in isolation.
- `set_reserve_shortfall(conn, strategy_id, state: int)` — persist the computed next state, commit.

**(ii) The multi-tick taint decision — SUBMISSION-SIDE, tolerance-aware, with an async-fill-aware
`pending` step (GATE-1 round-2 HIGH 1 + round-3 IMPORTANT 2 + round-4 HIGH).** The signal is derived
ONLY from caps (submission facts), never from post-fill holdings — because paper fills land ASYNC (an
accepted order fills at the next open and is ingested only on a LATER cycle), so a holdings comparison
at snapshot time would read stale, pre-fill belief. A cap is a pool bind BEYOND `DEFAULT_TOLERANCE`
(`intended_notional − grant > DEFAULT_TOLERANCE`); a sub-tolerance trim (a `< $1`/sub-cent remainder,
e.g. a residual that floors below `MIN_NOTIONAL`) is NOT a cap (economically negligible, within the
same tolerance the reconcile/sizing layer treats as "on target").

The `pending` state closes the round-4 gap: a fully-funded catch-up SUBMITTED this cycle does not
FILL until the next open, so the FOLLOWING cycle's return still covers a partly-under-invested
interval and must stay tainted for one extra tick. The machine (`next_shortfall_state`):

| state_before | capped_this_cycle | next state | this tick tainted? |
|---|---|---|---|
| `clear`   | no  | `clear`   | no  |
| `clear`   | yes | `capped`  | yes |
| `capped`  | yes | `capped`  | yes |
| `capped`  | no  | `pending` | yes  (catch-up submitted in full THIS cycle; fill not yet landed) |
| `pending` | yes | `capped`  | yes |
| `pending` | no  | `clear`   | yes  (fill has now had an ingest opportunity; clears NEXT tick) |

`_run_paper_strategy_tick` reads `state_before` at tick ENTRY. After `run_tick` returns SUCCESSFULLY:
```
capped_this_cycle = (last_cap_cycle == cycle_id)          # the closure set last_cap_cycle on a cap
reserve_trimmed   = (state_before != 0) or capped_this_cycle
set_reserve_shortfall(conn, strategy_id, next_shortfall_state(state_before, capped_this_cycle))
```
Worked example (cap at cycle N, pool has room from N+1 on): N `capped`/trimmed=1 → N+1 `pending`/
trimmed=1 (catch-up submitted) → N+2 `clear`/trimmed=1 (catch-up filled between N+1 and N+2, but this
tick's return still covered the tail of the under-investment) → N+3 trimmed=0 admissible. Three
tainted ticks, clearing on the fourth — exactly the round-4 "one additional snapshot after the funded
catch-up has had a fill opportunity." A still-unfundable residual re-caps (`pending`/`capped` → `capped`)
and stays tainted until the pool can fund it; a target-drop (no further buy) walks `capped→pending→clear`
over two quiet cycles (a bounded, fail-SAFE over-exclusion of at most one extra tick, only ever for an
already-capped strategy — a strategy that never caps is never touched).

**Why this does NOT over-exclude actively-rebalancing strategies.** The clear condition keys off CAPS
(pool binds), not off "believed ≠ target" — so a strategy that rebalances every tick but is never
again pool-capped walks `capped→pending→clear` in two quiet cycles regardless of its normal
rebalance/drift lag. This is why the machine is submission-side rather than a holdings-vs-target
comparison (which would keep an active strategy permanently off-target-by-lag and permanently tainted).

Durability: if a crash lands after `mark_reserve_shortfall` advanced state to `capped` but before the
snapshot, the state stays `capped`, so the NEXT cycle's tick sees `state_before=capped` and taints —
the capped order's NAV effect is captured (Codex F2).

**(ii-b) Every snapshot-writing path passes the computed `reserve_trimmed` — the exception paths write
NO snapshot (GATE-1 round-2 HIGH 2).** In `_run_paper_strategy_tick`, `record_tick_snapshot` is
called on the SUCCESS path ONLY; the `RiskBreach`/`TickHalted`/`LiveSizingError` handlers return a
marker WITHOUT recording a snapshot. So a capped-then-breached tick contributes NO evidence row that
could carry a stale `reserve_trimmed=0` — the contamination cannot leak via a defaulted flag on an
exception path, because no row is written. The DURABLE `state=capped` (committed at cap time) instead
propagates the taint to the NEXT recorded snapshot. The success-path call ALWAYS passes the
explicitly-computed `reserve_trimmed` (never relies on the parameter default); the default `False`
exists only for the single-strategy/live writers that structurally never cap. This path-completeness
is asserted by test (a breach after a cap records no snapshot AND leaves `state=capped`).

**(iii) Projected onto `tick_snapshots.reserve_trimmed` and EXCLUDED by the forward gate.** Add a
`reserve_trimmed INTEGER` column to `tick_snapshots`, written by `record_tick_snapshot` from the
value computed in (ii). **BOTH paper paths apply the state machine — single-strategy `paper trade-tick`
does NOT structurally write `0` (GATE-1 round-5 HIGH).** Both run through the SAME
`_run_paper_strategy_tick`, so a `trade-tick` for a strategy that a PRIOR `run-all` left in
`state=capped`/`pending` reads that `state_before` and taints (`reserve_trimmed = state_before != 0`),
then DRAINS the machine: `trade-tick` has no `reserve_buy` so it never caps
(`capped_this_cycle`=False), and the machine walks `capped→pending→clear` across its ticks, correctly
retiring the inherited taint rather than admitting a still-under-invested return. The single
monotonic `paper_reconcile.next_cycle` counter is shared by both commands, so `last_cap_cycle` from a
run-all cap never spuriously matches a later trade-tick cycle. ONLY `live` (and the SimBroker `paper
run` replay, which uses `persist_run` and records NO `tick_snapshots`) structurally never touches the
shortfall table and writes `0`. In `assemble_forward_evidence`, add `reserve_trimmed` to the
admissible-tick SELECT and to
`_inadmissible_reason`: `if row["reserve_trimmed"]: return "reserve_trimmed"`; add `"reserve_trimmed"`
to `_EXCLUSION_FILTERS`. A legacy `NULL` means untrimmed and stays admissible. **This is safe (Codex
round-2 IMPORTANT 7), not a fail-open, precisely because `paper run-all` is genuinely absent from
origin/main** (see § Why this is not already done — `git merge-base --is-ancestor` = false): NO
capped run-all evidence can exist in any database that predates this migration, so every pre-column
row is single-strategy `trade-tick` (which never caps). The migration comment records this invariant;
the finding-4 test asserts a fresh DB's legacy-`NULL` rows are admissible while only run-all caps set
`1`. Tainted ticks never count toward the ≥63-observation floor, coverage, or the Sharpe/vol/drawdown
metrics.

**Granularity is per-tick, per-strategy — deliberately.** A strategy whose OWN buys were never
trimmed (and which carries no outstanding shortfall) is uncoupled even if a SIBLING was trimmed that
cycle; its ticks stay admissible.

**Schema note (single-bump discipline).** ONE `SCHEMA_VERSION` bump (currently 35 -> 36) covers BOTH
DDLs in one `migrate()` step: the `reserve_trimmed` column via the idempotent `_add_missing_columns`
path AND `CREATE TABLE IF NOT EXISTS paper_reserve_shortfall`. Per the operating rule this PR owns
exactly ONE bump number; rebase to the next free integer if a concurrent bump lands first. `db.py`
and `order_state.py` are NOT CODEOWNERS-protected; `forward_promotion.py` IS (already forcing human
merge, which this PR requires anyway).

**Tests (finding 4):**
- taint window (`capped→pending→clear`): a run-all cap at cycle N taints N (`capped`), N+1 (`pending`,
  catch-up submitted) and N+2 (`clear`, fill landed but return still covered the tail) — all
  `reserve_trimmed=1`; N+3 is `0`. An untrimmed sibling is `0` throughout. The pure
  `next_shortfall_state` is unit-tested over all six transition rows.
- durability: after `mark_reserve_shortfall` advances `state=capped`, a simulated crash before the
  snapshot leaves `state=capped`; the next cycle's tick reads `state_before=capped` and taints.
- gate exclusion: a `reserve_trimmed=1` row is EXCLUDED from admissible evidence (counted under the
  new `reserve_trimmed` key); `0`/legacy-`NULL` rows remain admissible; a window whose
  only-otherwise-admissible ticks are all trimmed fails the ≥63-observation floor.

**Deferred follow-up (out of scope, file as issue):** a paper book-level risk wall (#389-equivalent,
scoped to the paper account) that removes the coupling at the source and lets trimmed ticks count
again. Until then, exclusion is the enforced posture.

---

## Finding 5 (blocking) — concurrency-exclusion contract (paper-account mutators)

**The hazard.** `paper trade-tick` and `paper run-all` each ingest venue fills, run
`_recover_stranded`, take a reconcile `cycle`, and tick strategies against the one paper account. The
DB posture (WAL + `busy_timeout=5000`) serializes individual writes but NOT a multi-statement paper
CYCLE. Two paper cycles overlapping the same account: double-ingest the same broker activities, race
`paper_reconcile.next_cycle`/`reconcile` (drift false-positives → spurious global halt), and — worst
— each cycle driver reads the FULL broker buying-power into its OWN in-memory `pool["available"]`, so
the cap is defeated across processes (both reserve the same BP → over-commit). run-all's whole safety
story assumes it is the sole writer of the paper account for the cycle. **The same double-ingest /
account-mutation race applies to the paper liquidation/recovery commands (`flatten`, `halt-all`,
`resume-all`), which the prior spec left unlocked (Codex F4).**

**Contract — one advisory paper-lane lock, but with emergency-correct acquisition.** A context
manager `paper_cycle_lock(*, blocking=False)` (in `paper_cmd.py`) operates a NON-BLOCKING (default)
`fcntl.flock` (`LOCK_EX | LOCK_NB`) on `{db_path.parent}/paper-cycle.lock` (mirrors `clear_staging`
#255 and the kb `.sync.lock` flock idioms). One lock file per DB (settings resolve to ONE paper
account; the pool and reconcile are account-wide, so it need not be keyed per strategy). Released in
`finally` (and by process exit).

**Scope — which commands take the lock, and why (Codex F4 "widen or justify"):**
- **`trade-tick`, `run-all`** — full cycle drivers. Wrap the ENTIRE cycle body (ingest → reconcile →
  tick loop) NON-BLOCKING; on contention FAIL CLOSED: emit
  `{"ok": False, "error": "...", "code": "paper_cycle_locked"}`, exit non-zero, do NO ingest/reconcile/
  tick (lock acquired OUTSIDE the broker/ingest work). They are unattended and retriable — never queue.
- **`flatten`, `halt-all`** — emergency liquidators that ingest + submit offsets / `close_all_positions`.
  They use an **engage-then-lock** order so the emergency STOP is bounded, never queued behind a full
  cycle: FIRST engage the kill-switch / global halt (a single durable DB write, safe under WAL — no
  cycle lock needed to stop trading), THEN acquire `paper_cycle_lock()` NON-BLOCKING to perform the
  LIQUIDATION (the part that races ingest). On contention: the halt is already engaged, so emit
  `{"ok": False, "code": "paper_cycle_locked", "halt_engaged": true, "liquidation_deferred": true}`
  and exit non-zero — the operator (or the in-flight cycle's own breach path) completes the
  liquidation once the cycle releases. **Precise stop guarantee (GATE-1 round-2 HIGH 4).** The claim
  is NOT that an in-flight cycle stops instantly; it is that it stops within AT MOST the one order
  already in flight. This is guaranteed by `run_tick`'s existing per-order re-check: `should_halt()`
  (= `kill_switch.is_tripped OR global_halt.is_engaged`) is evaluated BEFORE the cancel phase and
  BEFORE EVERY submit in the loop (live_loop.py L361-375), so once the emergency command's durable
  `global_halt.engage` commits, the in-flight cycle raises `TickHalted` before its next `submit_sized`
  and posts no further orders. The at-most-one-in-flight-order window is the standard, bounded
  guarantee (an order already handed to `submit_sized` may complete); the wording "never blocked" is
  therefore replaced by this precise bound. The finding-5 test asserts the halt is durably engaged on
  lock contention (so the re-check will fire) even though the liquidation defers.
- **`resume-all`** — clears the global halt and re-bases account-wide peaks; must not run concurrently
  with a cycle (clearing the halt / re-basing peaks mid-cycle corrupts the cycle's risk state). It is
  a recovery, not an emergency stop, so it wraps its body NON-BLOCKING fail-closed like the drivers
  (operator retries once the cycle releases).
- **Explicitly NOT locked (justified):** `run` (SimBroker replay — separate in-memory cash, writes
  `paper_orders`/`fills`, never touches the Alpaca paper account or `paper_venue_*` ledgers, so no
  shared-account race); `kill`, `resume` (single-row kill-switch/peak writes serialized by WAL, no
  ingest/submit cycle); `show`, `account`, `promote` (read-only — `promote` is pure evidence
  assembly). Each is a single-statement or read-only path with no multi-statement account cycle to
  race.

**Locality + the one-DB-per-account invariant (Codex F4 note N2 + round-2 IMPORTANT 6).**
`fcntl.flock` is HOST/PROCESS-local: it serializes concurrent processes on ONE machine, NOT an
account-wide distributed mutex. It is sufficient and correct ONLY under two deployment invariants,
which the docstring states explicitly rather than implying a stronger guarantee: (1) ONE operator
host per paper account (no two machines share one Alpaca account — the deployment does not do this);
and (2) ONE registry DB path per paper account. The lock file is derived from the DB path
(`{db_path.parent}/paper-cycle.lock`), so invariant (2) is what makes the DB-path-keyed lock coincide
with the account: two configs pointing at DIFFERENT DB files but the SAME Alpaca account would take
DIFFERENT locks and could over-commit the shared broker BP. Since the registry DB IS the per-account
state store (ledgers, reconcile cursor, kill-switches all live in it), one-DB-per-account is already
the operating model; the docstring records it as a REQUIRED invariant so a future multi-DB
misconfiguration is a documented violation, not a silent hole. `fcntl` is POSIX; the lock/tests are
skipped on non-POSIX (the CI/runtime target is Linux).

**Tests (finding 5):**
- hold `paper-cycle.lock` (open + `flock LOCK_EX` in the test), then invoke `run-all` and `trade-tick`:
  each exits non-zero with `code="paper_cycle_locked"` and performs NO ingest/reconcile/tick (assert
  the fake broker saw zero calls). Releasing the lock lets a subsequent invocation proceed.
- engage-then-lock: hold the lock, then invoke `halt-all`: the global halt IS engaged (durable write
  landed) AND the payload reports `liquidation_deferred=true`, `code="paper_cycle_locked"`, exit
  non-zero, and `close_all_positions` was NOT called. Same shape for `flatten` (kill-switch tripped,
  liquidation deferred).

---

## Finding 6 (blocking) — long-only enforced on BOTH the normal path and the breach path

**The gap.** The class-B pool-safety proof (finding 3) rests on "buys are the only pool-consuming
order, and a scoped flatten only SELLS." Both fail for a SHORT:
- **Opening a short (normal path)** is an unreserved SELL that drives believed qty negative; its
  margin consumption is never modeled by `pool["available"]`, and a later normal-path buy that
  REDUCES the short is reserved and could be pool-trimmed — wrongly trimming a risk-reducing order.
  The prior spec guarded shorts ONLY at breach-flatten time, so a short could be OPENED on a normal
  tick and silently desync the account (Codex F5).
- **Covering a short (breach path)** is a BUY-to-cover: `flatten_strategy` (execution/flatten.py L140)
  sets `side="buy"` when `offset_qty < 0` and submits via `broker.submit_offset`, which does NOT
  consult the reserve pool — a buy-to-cover consumes BP the pool never debits.

**Decision: paper run-all is LONG-ONLY for this change, enforced by TWO fail-closed guards.** The
shipped construction policies (`portfolio/construction.py`: `top_k_equal_weight`,
`equal_weight_positive`, the clip-negatives proportional policy) all produce long-only target weights,
so this costs nothing today; the guards defend against a short-enabled strategy slipping into the pool
later.

- **Guard 1 — normal order path (pre-submit, the Codex-F5 addition).** Add an optional
  `assert_long_only: Callable[[dict[str, float], Sequence[OrderIntent]], None] | None` hook to
  `TickHooks`. `run_tick` calls it in the submit phase IMMEDIATELY after `decide()` returns and
  BEFORE the cancel/submit loop (so it fires before ANY order posts). `_run_paper_strategy_tick`
  supplies a callable that raises `RiskBreach("short_position_unsupported", detail=...)` if any target
  weight `< -DEFAULT_TOLERANCE` (a short TARGET) — defense-in-depth it also rejects an already-short
  `live_positions()` belief. Because it raises before the submit loop, no short is ever opened. Both
  paper paths (`trade-tick` and `run-all`) supply it; the live path passes `None` (live may support
  shorts under its own #389 book risk). run_tick is shared but not CODEOWNERS-protected; the hook is
  optional so live/sim callers are unaffected.
  - **Why the target-weight check is SUFFICIENT at the intent level (GATE-1 round-2 HIGH 5).** The
    reviewer's scenario — a nonnegative target weight whose SELL intent nonetheless oversells a long
    into a short — cannot occur, because sizing lands the position AT the target, never past it:
    `size_order` (execution/sizing.py) computes `delta_notional = target_weight*equity −
    current_market_value`, so the post-order target market value is exactly `target_weight*equity ≥ 0`
    for any weight `≥ 0`. A sell is sized to reach `target_weight*equity`, i.e. it sells `current −
    target`, stopping AT the nonnegative target — it structurally cannot cross zero. Stale-ledger or
    rounding effects change `current_market_value` (how much to sell) but NOT the target the order
    lands on, so they cannot manufacture a short from a nonnegative weight. Therefore "no target weight
    `< -DEFAULT_TOLERANCE`" (plus the no-pre-existing-short belief check) is NECESSARY AND SUFFICIENT
    to guarantee no intent SEQUENCE opens a short. A post-fill overshoot from market movement between
    decision and fill is a fill-slippage concern OUTSIDE the pool-safety scope (and is what the
    reconcile layer catches), not an intent-level short. Rejected the heavier alternative of
    re-simulating post-intent share quantities in the guard: it would duplicate `size_order`'s math (a
    parity hazard) for zero added safety, since `size_order` is the tested authority that already
    guarantees land-at-target.
- **Guard 2 — breach path (pre-flatten).** In the class-B economic-breach path, BEFORE the scoped
  `flatten_strategy`, inspect `paper_believed_positions(conn, name)`: if ANY believed qty is short
  (`< -DEFAULT_TOLERANCE`), a buy-to-cover would desync the pool. FAIL CLOSED here too.
- **Unified fail-closed handling.** Both `short_position_unsupported` (guard 1, raised as RiskBreach)
  and the guard-2 believed-short condition route to the SAME class-A outcome: engage the global halt
  (reason `paper_short_position_unsupported`), audit it, and return a marker with
  `global_halt_engaged=True`. run-all `break`s (class A) rather than continue siblings against a
  possibly-desynced pool. Concretely: add `"short_position_unsupported"` to the RiskBreach kinds the
  paper handler routes to the global-halt branch (alongside `stale_marks`/`unvaluable_marks`), and add
  the guard-2 believed-short check to the economic branch before the flatten.
- **Constraint statement.** `run_all`'s docstring: "paper run-all requires long-only strategies; a
  short target at submit time OR a believed short at breach-flatten time fails the account closed."

**Alternative considered and deferred (option b).** Extend pool wiring so a buy-to-cover
UNCONDITIONALLY debits `pool["available"]` (never trimmed — a risk-reducing cover must complete in
full). Correct once shorts are enabled, but requires threading a non-reserving pool-debit through
`flatten_strategy`/`submit_offset`; out of scope. Filed as the deferred follow-up.

**Tests (finding 6):**
- normal path (Codex F5): a strategy whose `decide()` produces a target weight `< 0` under run-all
  triggers `assert_long_only` → `RiskBreach("short_position_unsupported")` BEFORE any order posts →
  global halt engaged, marker `global_halt_engaged=True`, run-all `break`s, NO short SELL submitted,
  exit 1.
- breach path: a strategy holding a believed SHORT that hits an economic breach engages the global
  halt (marker `global_halt_engaged=True`), run-all `break`s, and no buy-to-cover is submitted through
  the unreserved offset path; exit 1.

---

## Non-blocking notes (addressed)

- **Order-dependent exclusion bias (note N1) → mitigated by per-cycle rotation (decision 1a),** not
  just documented. A test asserts the tick order's rotating offset advances with the cycle id so no
  strategy is permanently first.
- **`fcntl.flock` locality (note N2) → stated explicitly** in the finding-5 lock docstring
  (host/process-local, single-box Linux; not a distributed/account-wide mutex).
- **FORWARD_TESTED inclusion (note N3) → stated explicitly** in the § Strategy selection section and
  the `run_all` docstring (selection is `stage IN ('paper','forward_tested')`).
- **Stale-marks global-halt stops remaining siblings' pending reservations/cancels.** Handled by the
  explicit `break` on a `global_halt_engaged == True` marker (finding 3, class A). run-all does NOT
  merely rely on each remaining sibling's `should_halt()` (which would still invoke their scoped
  cancel path and hit the broker before halting) — it breaks out of the loop. Asserted by the class-A
  test.
- **BP-pool enforcement comes from `pool["available"]`, not the audit call.** A test injects a
  no-op / raising `audit_append` (patched) and asserts the pool STILL caps submitted notionals and
  skips buys once available hits 0 — enforcement is the in-memory pool debit + `posted_notional`, and
  `paper_reserve_trim` is observability-only.
- **Merge policy: request HUMAN merge — FORCED by CODEOWNERS.** This change enforces promotion-evidence
  exclusion inside `forward_promotion.py` (a CODEOWNERS-protected surface) and alters shared-account
  breach/concurrency behavior (findings 3, 5, 6) — the PR MUST stay open for human merge regardless
  of CI status. Do NOT cite "no CODEOWNERS path touched → auto-merge on green."

---

## Files
- `algua/execution/alpaca_broker.py` — add the pure `posted_notional(grant)` helper; refactor
  `submit_sized` to use it for the floor/skip decision (findings 1, 3). NOT CODEOWNERS-protected.
- `algua/execution/live_ledger.py` — add `orders_table(kind)` accessor; make
  `owned_open_order_ids(..., *, kind: LedgerKind)` required-kw-only reading `orders_table(kind)`
  (finding 2). NOT CODEOWNERS-protected.
- `algua/live/live_loop.py` — add the optional `assert_long_only` hook to `TickHooks`, called after
  `decide()` and before the submit loop (finding 6, guard 1). NOT CODEOWNERS-protected.
- `algua/cli/live_cmd.py` — update `_scoped_cancel` to pass `kind=LedgerKind.LIVE` (finding 2).
  NOT CODEOWNERS-protected.
- `algua/cli/paper_cmd.py` — add `reserve_buy` + `cycle_id` params and wire `reserve_buy` into
  `_run_paper_strategy_tick`'s `TickHooks`; derive the durable multi-tick taint (read `state_before`
  at entry, compute `reserve_trimmed`, advance the `capped→pending→clear` machine via
  `next_shortfall_state`) and thread it into `record_tick_snapshot` (findings 1, 2, 4); add `assert_long_only` supply + the guard-2 believed-short
  check + route `short_position_unsupported` to the global-halt branch (finding 6); add the
  `global_halt_engaged` boolean to every `ok:False` marker (finding 3); parameterize the breach-path
  `cancel` (decision 3); add `_paper_scoped_cancel`, `_paper_broker_buying_power`, and
  `paper_cycle_lock(*, blocking=…)`; wrap `trade-tick` in the lock; add engage-then-lock to `flatten`
  and `halt-all` and a non-blocking wrap to `resume-all` (finding 5); add the `run-all` command
  (shared pool + `_reserve_for` closure; per-cycle strategy-order rotation, decision 1a;
  `stage IN ('paper','forward_tested')` selection; class-A/class-B loop keyed on `global_halt_engaged`;
  `paper_cycle_lock` wrap; long-only + evidence-exclusion + FORWARD_TESTED docstring). NOT
  CODEOWNERS-protected.
- `algua/execution/order_state.py` — `record_tick_snapshot` accepts + persists `reserve_trimmed`;
  add `mark_reserve_shortfall`, `reserve_shortfall_state`, the pure `next_shortfall_state`
  (0/1/2 machine), and `set_reserve_shortfall` (findings 2, 4; round-4 HIGH). NOT CODEOWNERS-protected.
- `algua/registry/db.py` — add `reserve_trimmed INTEGER` to `tick_snapshots` via
  `_add_missing_columns` AND `CREATE TABLE IF NOT EXISTS paper_reserve_shortfall` (keyed by
  `strategy_id INTEGER PRIMARY KEY`); bump `SCHEMA_VERSION` 35 -> 36 (this PR's SOLE bump) (finding 4).
  NOT CODEOWNERS-protected.
- `algua/registry/forward_promotion.py` — add `reserve_trimmed` to the admissible-tick SELECT, a
  `"reserve_trimmed"` exclusion filter in `_inadmissible_reason`, and the key in `_EXCLUSION_FILTERS`
  (finding 4). **CODEOWNERS-protected → PR stays OPEN for human merge.**
- Tests: `tests/test_live_ledger_ledgerkind.py` (finding 2), `tests/test_alpaca_broker.py`
  (`posted_notional` + submit_sized parity), `tests/test_cli_paper.py` (reserve wiring, scoped cancel,
  trade-tick cycle-lock fail-closed, emergency engage-then-lock), new `tests/test_paper_run_all.py`
  (findings 1/3/4/5/6 + rotation + enforcement-provenance), `tests/test_order_state.py`
  (`reserve_trimmed` persistence + shortfall helpers), `tests/test_forward_promotion.py` (finding 4
  gate exclusion).

**ONE schema bump (35 -> 36), one column + one table.** **Request human merge** — forward-gate
CODEOWNERS edit + shared-account breach/concurrency behavior (findings 3-6).

---

## Task list

1. **live_ledger: `orders_table` + required-kw `kind` on `owned_open_order_ids` (finding 2).**
   Run `rg "owned_open_order_ids\("` FIRST to enumerate all call sites (required checklist step).
   Add `orders_table(kind)`; change the signature to `(..., *, kind: LedgerKind)`, read
   `orders_table(kind)`. Update EVERY caller (`live_cmd._scoped_cancel` → `kind=LedgerKind.LIVE`) and
   the existing test. Tests: `tests/test_live_ledger_ledgerkind.py` — missing `kind` ⇒ `TypeError`;
   PAPER reads `paper_venue_orders`; LIVE reads `live_orders`.
   FAST: `ruff && mypy && lint-imports && pytest -q tests/test_live_ledger_ledgerkind.py tests/test_live_ledger_orders.py`.

2. **`posted_notional` shared helper + submit_sized refactor (findings 1, 3) — alpaca_broker.py.**
   Extract the floor-to-cents/MIN_NOTIONAL decision into pure `posted_notional(grant) -> float`
   (0.0 when sub-`MIN_NOTIONAL`); refactor `submit_sized` to use it (return `"skipped"` when 0.0),
   behaviour-preserving. Tests: `tests/test_alpaca_broker.py` — `posted_notional` floors down and
   zeroes sub-min; submit_sized still skips/floors identically.
   FAST: `ruff && mypy && lint-imports && pytest -q tests/test_alpaca_broker.py`.

3. **Durable taint schema + state helpers (findings 2, 4, part 1) — db.py + order_state.py.**
   Add `reserve_trimmed INTEGER` to `tick_snapshots` via `_add_missing_columns` AND
   `CREATE TABLE IF NOT EXISTS paper_reserve_shortfall(strategy_id INTEGER PRIMARY KEY, state INTEGER
   NOT NULL DEFAULT 0, last_cap_cycle INTEGER, updated_at TEXT)` (id-keyed, round-3 MINOR; `state` is
   the 0=clear/1=capped/2=pending machine, round-4 HIGH); bump `SCHEMA_VERSION` 35 -> 36 (SOLE bump;
   rebase if a concurrent bump lands first). Add `reserve_trimmed: bool = False` to
   `record_tick_snapshot` (writes `1`/`0`); add `mark_reserve_shortfall(conn, strategy_id, cycle_id)`
   (→state=capped), `reserve_shortfall_state(conn, strategy_id)`, the PURE
   `next_shortfall_state(state_before, capped)`, and `set_reserve_shortfall(conn, strategy_id, state)`
   (each DB helper commits). Tests: `tests/test_order_state.py` — snapshot round-trips `reserve_trimmed`
   (default `0`); `mark` advances to `capped` + records `last_cap_cycle` (idempotent upsert); the pure
   `next_shortfall_state` covers all SIX rows of the transition table (esp. `capped→pending→clear`).
   FAST: `ruff && mypy && lint-imports && pytest -q tests/test_order_state.py`.

4. **Forward-gate exclusion of tainted ticks (finding 4, part 2) — forward_promotion.py [CODEOWNERS].**
   Add `reserve_trimmed` to the admissible-tick SELECT; add `if row["reserve_trimmed"]: return
   "reserve_trimmed"` to `_inadmissible_reason`; add `"reserve_trimmed"` to `_EXCLUSION_FILTERS`.
   Legacy NULL stays admissible. Tests: `tests/test_forward_promotion.py` — a `reserve_trimmed=1` row
   is excluded (counted under the new key); `0`/NULL admissible; an all-trimmed window fails the
   ≥63-observation floor. PR stays OPEN for human merge (CODEOWNERS).
   FAST: `ruff && mypy && lint-imports && pytest -q tests/test_forward_promotion.py`.

5. **live_loop: `assert_long_only` hook on TickHooks (finding 6, guard 1) — live_loop.py.**
   Add the optional `assert_long_only(weights, intents)` field to `TickHooks`; call it in the submit
   phase right after `decide()` and before the cancel/submit loop; `None` ⇒ skipped (live/sim
   unaffected). Tests: `tests/test_live_loop.py` — a supplied hook that raises is invoked before any
   `submit_sized`; `None` leaves the loop unchanged.
   FAST: `ruff && mypy && lint-imports && pytest -q tests/test_live_loop.py`.

6. **paper_cmd: helpers, marker contract, long-only guards, cycle lock, reserve/taint wiring
   (findings 1, 2, 3, 4, 5, 6; dec 3).** Add `paper_cycle_lock(*, blocking=False)` (fcntl
   LOCK_EX|LOCK_NB, host-local docstring, fail-closed `code=paper_cycle_locked`); wrap `trade-tick`'s
   cycle body; add engage-then-lock to `flatten`/`halt-all` (halt/kill FIRST, then non-blocking lock
   around liquidation, `liquidation_deferred` on contention) and a non-blocking wrap to `resume-all`.
   Add `_paper_scoped_cancel` (`kind=LedgerKind.PAPER`) and `_paper_broker_buying_power`. Add
   `reserve_buy=None` and `cycle_id=None` params to `_run_paper_strategy_tick`; set `reserve_buy` on
   the hooks; read `state_before` at entry; on the SUCCESS path compute `reserve_trimmed =
   (state_before != 0) or capped_this_cycle` (where a cap is the SUBMISSION-SIDE `intended − grant >
   DEFAULT_TOLERANCE`, no async post-fill holdings — round-2 HIGH 1 + round-3 IMPORTANT 2), advance via
   `set_reserve_shortfall(conn, rec.id, next_shortfall_state(state_before, capped_this_cycle))` (the
   0/1/2 machine with the `pending` async-fill step — round-4 HIGH), and thread `reserve_trimmed` into
   `record_tick_snapshot`; the breach/halt/skip paths write NO snapshot (round-2 HIGH 2) so the durable
   `capped` state carries the taint forward. Emit `paper_reserve_trim` audit BEST-EFFORT
   (try/except-swallow) so it cannot abort the cycle (round-3 HIGH).
   Supply `assert_long_only` (target weight `< -tol` ⇒ `RiskBreach("short_position_unsupported")`) and
   add the guard-2 believed-short check before the scoped flatten; route `short_position_unsupported`
   to the global-halt branch. Give EVERY `ok:False` marker a `global_halt_engaged` boolean (stale/
   long-only True; `except TickHalted` re-reads; economic False). Parameterize the breach-path flatten
   `cancel=cancel if cancel is not None else broker.cancel_open_orders`.
   Tests: `tests/test_cli_paper.py` — single-strategy tick account-wide-cancels + no pool cap; a
   supplied `reserve_buy` reaches `submit_sized` and caps the buy; trade-tick under a held cycle-lock
   fails closed with no ingest; `halt-all`/`flatten` under a held lock still engage halt/kill and
   report `liquidation_deferred`; markers carry the right `global_halt_engaged` per path.
   FAST: `ruff && mypy && lint-imports && pytest -q tests/test_cli_paper.py`.

7. **paper_cmd: the `run-all` command (findings 1, 3, 4, 5, 6; decisions 1a, 2, 5; note N3).**
   `paper_cycle_lock` wrap; observability wrapper (configure_logging + CycleCounters +
   correlation_context + finally flush); `stage IN ('paper','forward_tested')` selection;
   per-cycle strategy-order rotation (`cycle_id % n` over registry-id order); ingest-once +
   `_recover_stranded` + account reconcile with the pre-loop account-level stops (halt ⇒ global
   halt+exit1; not clean ⇒ defer); build `pool` + `_reserve_for` closure (debits `posted_notional`,
   durably marks shortfall on a pool-cap); per-strategy loop calling `_run_paper_strategy_tick(cancel=
   scoped, reserve_buy=_reserve_for(name, rec.id), cycle_id=cycle)`; class-A stop (`break` on
   `global_halt_engaged`) vs class-B continue per finding 3; `#270` envelope surfacing all ticked
   siblings + `ok:False`/exit1 on any breach; long-only + evidence-exclusion + FORWARD_TESTED
   docstring.
   Tests: `tests/test_paper_run_all.py` —
   (a) finding 1/3: `submitted <= grant` (floored, not exact) + never the NAV amount when bound; pool
       exhaustion ⇒ later buy `"skipped"`; pool ends decremented by exactly Σ`posted_notional` of
       submitted buys (skip/floor never over-debits).
   (b) finding 3 class A (systemic): mid-loop stale_marks ⇒ global halt, remaining siblings UNticked,
       exit 1.
   (c) finding 3 class A (EXTERNAL): global halt engaged externally between siblings ⇒ next tick
       `TickHalted` (`global_halt_engaged=True`) ⇒ break, later sibling UNticked, exit 1.
   (d) finding 3 class B: mid-loop drawdown ⇒ scoped flatten, remaining siblings STILL ticked, exit 1;
       kill-switch-only `TickHalted` (`global_halt_engaged=False`) also continues.
   (e) pool-safety: post-class-B pool decremented only by real submitted buys, not flatten sells.
   (f) finding 4 taint WINDOW (`capped→pending→clear` machine): a cap at cycle N taints N (`capped`),
       N+1 (`pending`, catch-up submitted), AND N+2 (`clear`, catch-up fill had an ingest opportunity
       but this return still covered the tail) — all `reserve_trimmed=1`; N+3 is `0`; untrimmed sibling
       `0` throughout (round-4 HIGH: the async-fill extra tick). Reconvergence semantics (round-2 HIGH
       1 + round-3 IMPORTANT 2): a still-unfundable residual RE-CAPS beyond `DEFAULT_TOLERANCE` and
       stays `capped`; a sub-tolerance residual (floored below `MIN_NOTIONAL`) is NOT a cap — asserted
       submission-side, no async post-fill holdings read. Path completeness (round-2 HIGH 2): a cap
       FOLLOWED by a breach records NO snapshot yet leaves `state=capped`, so the next recorded snapshot
       taints. Durability: `state=capped` survives a pre-snapshot crash and taints next cycle.
       Cross-command drain (round-5 HIGH): after run-all leaves S in `state=capped`, a subsequent
       single-strategy `paper trade-tick` for S writes `reserve_trimmed=1` and advances
       `capped→pending`/`→clear` — it does NOT reset the taint to 0.
   (g) finding 5: run-all under a held cycle-lock fails closed (`code=paper_cycle_locked`), zero broker
       calls.
   (h) finding 6 normal path: a target-weight `< 0` under run-all ⇒ global halt, break, no short SELL
       submitted, exit 1. And breach path: a believed-short breach ⇒ global halt, break, no
       buy-to-cover, exit 1.
   (i) decision 1a: the per-cycle tick order's rotating offset advances with the cycle id.
   (j) non-blocking enforcement-provenance: with `audit_append` patched no-op/raiser, the pool STILL
       caps notionals and skips at 0.
   FAST: `ruff && mypy && lint-imports && pytest -q tests/test_paper_run_all.py tests/test_cli_paper.py`.

8. **Integration + PR.** Run the FULL gate:
   `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
   Open the PR (feature branch pushed alone) and **request human merge** — do not auto-merge
   (forward-gate CODEOWNERS edit + shared-account breach/concurrency behavior, findings 3-6).

## GATE-1 re-run

**Round 1 (closed).** Codex (gpt-5.5) GATE-1 BLOCK: 5 HIGH — (1) taint persistence/window now DURABLE
+ multi-tick (§ Finding 4); (2) submission-time committed shortfall write (§ Finding 4(ii), § Finding
1 closure); (3) pool debited by `posted_notional` (§§ Finding 1, 3); (4) `paper_cycle_lock` widened to
`flatten`/`halt-all`/`resume-all` with engage-then-lock (§ Finding 5); (5) normal-path pre-submit
long-only guard (§ Finding 6, guard 1) — plus 3 notes folded in (rotation 1a; flock locality;
FORWARD_TESTED inclusion).

**Round 2 (closed — this revision).** Re-running GATE-1 on the round-1 spec returned a second BLOCK
with 5 HIGH; all addressed here:
- **HIGH 1 — reconvergence was fail-open** ("first no-cap tick" cleared the taint even if the residual
  buy floored below `MIN_NOTIONAL` / was skipped, leaving the strategy under-invested). Fixed by
  making reconvergence tolerance-aware (a cap is `intended − grant > DEFAULT_TOLERANCE`). NOTE: the
  round-2 draft of this fix used a post-tick holdings comparison, which round 3 (async fills) and
  round 4 (one-tick-early) SUPERSEDED — the authoritative model is the round-4 submission-side
  three-state machine in § Finding 4(ii). This bullet is retained only as history.
- **HIGH 2 — exception-path snapshots could default `reserve_trimmed=0`.** RESOLVED: breach/halt/skip
  paths write NO snapshot at all; the durable `state=capped` flag carries the taint to the next
  recorded snapshot; the success path always passes the computed value (§ Finding 4(ii-b)).
- **HIGH 3 — pool debited before a confirmed POST.** RESOLVED with an explicit invariant argument: a
  skip/floor debits `0` via `posted_notional`, and the only other divergence (a `BrokerError` POST
  failure) ABORTS the whole cycle (not caught per strategy), so a phantom debit can never reach a
  sibling; two-phase commit is the deferred trigger only if submit failures become non-fatal
  (§ Finding 1).
- **HIGH 4 — engage-then-lock stop timing.** RESOLVED: the guarantee is bounded to at-most-one-in-flight
  order, grounded in `run_tick`'s existing per-order `should_halt()` re-check (live_loop L361-375); the
  "never blocked" wording was replaced by this precise bound (§ Finding 5).
- **HIGH 5 — normal-path long-only guard sufficiency.** RESOLVED with a proof that a nonnegative target
  weight cannot open a short because `size_order` lands the position AT `target_weight*equity ≥ 0`
  (never oversells past target); qty-simulation rejected as a `size_order` parity hazard (§ Finding 6,
  guard 1).
- Round-2 IMPORTANT 6 (one-DB-per-account invariant) and 7 (legacy-NULL safe because run-all is absent
  from main) folded into §§ Finding 5 and 4 respectively.

**Round 3 (closed — this revision).** Re-running GATE-1 on the round-2 spec returned ONE HIGH + one
IMPORTANT + one MINOR; all addressed here:
- **HIGH — the `paper_reserve_trim` audit was still inline in the enforcement path** (after the pool
  debit + durable mark), so a raising audit sink could abort the cycle / desync state, contradicting
  "audit enforces nothing." FIXED: the audit is now emitted BEST-EFFORT (try/except-swallow) after the
  enforcement/durable writes; the enforcement-provenance test additionally patches `audit_append` to
  RAISE (§ Finding 1 closure, decision 4, task 7(j)).
- **IMPORTANT — holdings-based reconvergence collided with ASYNC paper fills** (the sizing snapshot is
  pre-fill, so comparing intended-vs-believed would read stale belief). FIXED: reconvergence is now
  measured SUBMISSION-SIDE and tolerance-aware (a cap is `intended − grant > DEFAULT_TOLERANCE`); a
  genuine shortfall deterministically re-caps on the next tick, a sub-tolerance residual clears — no
  async post-fill read (§ Finding 4(ii)).
- **MINOR — `paper_reserve_shortfall` was name-keyed** while evidence is `strategy_id`-keyed. FIXED:
  the table + helpers are keyed by `strategy_id` (§ Finding 4(i), Files, task 3).

**Round 4 (closed — this revision).** Re-running GATE-1 on the round-3 spec returned ONE HIGH:
- **HIGH — submission-side reconvergence cleared one tick too early under ASYNC paper fills** (a
  fully-funded catch-up SUBMITTED at cycle N+1 only FILLS by N+2, so the N+2 return still covered an
  under-invested interval but was admitted). FIXED: the shortfall is now a THREE-state machine
  `clear→capped→pending→clear` where a no-cap tick after a cap moves to `pending` (catch-up submitted,
  fill not yet landed) and stays tainted for one MORE tick before clearing — Codex's own recommended
  `pending_reconvergence` fix, kept purely submission-side so it does not over-exclude
  actively-rebalancing strategies (§ Finding 4(i)+(ii), the transition table).

**Round 5 (closed — this revision).** Re-running GATE-1 on the round-4 spec returned ONE HIGH:
- **HIGH — single-strategy `paper trade-tick` was stated to "always write 0"**, which would RESET a
  taint a prior `run-all` cap left durable — leaking a still-under-invested return into evidence.
  FIXED: both paper commands run through the SAME `_run_paper_strategy_tick`, so `trade-tick` also
  reads `state_before` and advances the machine; having no `reserve_buy` it never caps, so it only
  ever DRAINS an inherited shortfall (`capped→pending→clear`), never resets it. Only `live` (and the
  no-tick_snapshot SimBroker `paper run` replay) structurally writes `0` (§ Finding 4(iii), Finding 1).

Re-run GATE-1 (Codex read-only adversarial) once more against THIS revision before GATE-2/
implementation sign-off to confirm the round-5 fix holds and no new HIGH was introduced.
