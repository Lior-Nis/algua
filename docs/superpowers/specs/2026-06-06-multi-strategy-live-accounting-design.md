# Multi-Strategy Live Accounting — Design (Sub-project 6, live-execution slices A–C)

**Date:** 2026-06-06
**Status:** Accepted (pending implementation)
**Scope:** A portfolio-accounting / OMS layer so ~100 strategies can trade LIVE simultaneously on ONE
shared Alpaca account, with a per-strategy internal ledger as the source of truth for attribution,
a sequenced portfolio loop, account-level reconcile, and per-strategy capital / drawdown / flatten.

This supersedes the earlier "slice 4 = minimal safety core" framing. It was reviewed as a *design*
by Codex (adversarial), which validated the direction (one custodian account + internal per-strategy
ledger is the standard books-and-records model) but required four CRITICAL corrections; they are
folded in below.

---

## 1. Why / context

`live trade-tick` (slice 3, PR#118) treats the broker account as the strategy's whole book: it sizes
target weights off the ACCOUNT equity, tracks one account-wide drawdown peak, and flattens via
`close_positions(universe)`. That is correct only for ONE strategy owning the account. The operator
wants ~100 strategies live at once. You do NOT open a brokerage account per strategy (KYC, fragmented
capital, minimums, ops do not scale).

**The model (industry books-and-records):** ONE Alpaca account is the custodian and holds the NETTED
book (strategy A's 100 AAPL + B's 50 AAPL → Alpaca shows 150). The platform is the source of truth
for attribution via an internal per-strategy ledger: every order is tagged with its strategy
(`client_order_id` already encodes it), each fill is recorded against that strategy, and a strategy's
position = Σ its own fills.

**Non-goals (later):** max-order-size caps, a web dashboard, sophisticated corporate-action handling
beyond ingesting the activity stream, margin/short-borrow modeling beyond what Alpaca reports,
multi-account/sub-account support.

---

## 2. Decisions settled in brainstorming

- **One account, internal per-strategy ledger** (not an account per strategy).
- **Fixed allocation** per strategy is the **sizing denominator** (operator sets/changes it
  deliberately); NAV (allocation ± P&L) is the **drawdown** basis. Chosen over floating-NAV for
  bounded real-money exposure and clean capital control.
- **Pull the authoritative fill/activity stream** each cycle and attribute by order tag (not poll per
  order).
- **One sequenced portfolio loop** (`live run-all`) ticks all live strategies per cycle — not 100
  racing CLI invocations. The only model under which account-level reconcile + buying-power
  reservation are coherent.

---

## 3. The portfolio loop — `algua live run-all`

One process per cycle, in order:

1. **Re-verify** each live strategy's go-live signature (`verify_live_authorization`, the per-tick
   wall — still per strategy; a strategy that fails is skipped + flagged, others proceed).
2. **Ingest activities** — pull Alpaca account-activities since the stored cursor (ALL activity
   types: FILL, DIV, fees, splits/corporate actions), insert rows and advance the cursor in ONE
   sqlite transaction, idempotent by broker **activity id**, with an overlap-window replay (re-pull a
   small window before the cursor and rely on idempotent upsert). Crash-safe: a crash mid-pull never
   double-counts or drops.
3. **Reconcile the account** (§5) — before any trading. A persistent unexplained drift engages global
   halt and aborts the cycle.
4. **Snapshot buying power** — read account buying power once; it is the shared pool for the cycle.
5. **Tick each live strategy in deterministic order** (§4): build its ledger-backed snapshot → NAV
   drawdown check → target deltas → reserve buying power from the pool (trim/skip + flag if short) →
   cancel ITS OWN stale open orders (scoped) → submit ITS OWN orders → record to `live_orders`.
6. Persist per-strategy NAV / tick snapshot.

`live trade-tick <name>` becomes a thin single-strategy special case (or is retired) once the loop
exists. During Slice A a **hard guard** allows at most ONE live strategy until the Slice B/C controls
land.

---

## 4. Per-strategy tick (inside the loop)

- **Sizing denominator = the fixed allocation `$X`** (NOT account equity). The per-strategy snapshot
  fed to the sizing rule is built from the LEDGER: `equity = allocation`, `positions = believed
  positions` (Σ settled fills for this strategy), marked at current prices from the provider.
- **Drawdown** is measured on the strategy's **NAV** (allocation + realized + unrealized + attributed
  cash) against a per-strategy NAV peak; a breach trips the per-strategy kill-switch + liquidation
  (§6).
- **Buying-power reservation:** each strategy's net BUY notional draws down the cycle's shared pool;
  sells add back; if the pool can't cover a strategy's buys, trim/skip them and flag (so the 100
  strategies' fixed allocations can't collectively exceed real account buying power, which drifts
  with P&L and open orders).
- **Owned orders only:** the strategy submits and cancels only orders whose `client_order_id` belongs
  to it.

---

## 5. The reconcile (classified, not naive equality)

Naive `Σfills == broker book` would false-halt almost every cycle because market orders are async
(submit returns an id, fills land later) and the activity feed lags positions. Instead:

- **Expected account net** per symbol = Σ(believed positions from SETTLED fills, all strategies).
  Open orders do not move positions, so after step-2 ingest this should equal the broker's
  `/v2/positions`.
- **Classify** each per-symbol mismatch:
  - within a **rounding / fractional-share tolerance** → OK;
  - explained by a **just-ingested activity** already applied to the ledger (split/div) → OK;
  - a **transient** gap (a fill in positions but not yet in the activities feed) → record it, flag,
    and re-check next cycle within a bounded **grace window** (N cycles / T seconds);
  - a **persistent or material unexplained** gap → engage **global halt** + flag for human.
- Reconcile state (per-symbol pending mismatches + first-seen cycle) is persisted so the grace window
  survives restarts.

This keeps global-halt for genuine "books disagree with reality" events, not ordinary timing skew.

---

## 6. Accounting & safety semantics

- **P&L / cost basis:** average-cost with **signed** position transitions. Realized P&L is booked on
  reducing trades (and zero-crossings split into close + open); unrealized = (mark − avg_cost) × qty;
  correct for long, short, flip, and partial-close sequences. (Explicitly NOT `Σqty×price/Σqty`.)
- **NAV** = allocation + realized P&L + unrealized P&L + attributed non-trade cash (dividends/fees
  that are symbol- or order-linked; account-level cash with no attribution goes to a **suspense**
  bucket that, if material, blocks trading rather than silently skewing NAV).
- **Order identity:** `client_order_id` (encodes strategy+decision_ts+symbol) is the durable primary
  key; `broker_order_id` is backfilled by reconciling broker orders by client id — covers a submit
  that timed out after Alpaca accepted it.
- **Scoped cancel:** cancel only orders owned by the strategy (by client/broker order id), never
  account-wide. Account-wide cancel is reserved for global halt.
- **Flatten = a liquidation workflow,** not a one-shot offset: freeze the strategy (kill-switch) →
  cancel its own open orders → re-ingest fills → compute its current believed qty → submit
  reduce-only-equivalent offsets → require a clean re-reconcile before resume. Prevents a stale
  believed-position from over-shooting and disturbing siblings.
- **Dual views, both first-class:** per-strategy gross/net exposure from the ledger (risk, drawdown,
  liquidation) AND account net from the broker (custody, reconcile). Broker-net-flat ≠ strategy-flat.
- **Allocation is a lifecycle object:** `live allocate <name> --capital X` writes an audited record
  (effective ts, actor, immutable history). Go-live REQUIRES an allocation. Enforce Σ(live
  allocations) ≤ account_equity × leverage (leverage default 1.0) at allocate time. Deallocation
  requires the strategy flat with no open orders. A re-allocation resets the NAV drawdown peak.

---

## 7. Data model (new sqlite tables; schema bump)

- `strategy_allocations(id, strategy_id, capital, leverage, effective_ts, actor, revoked_ts)` —
  append-only lifecycle; the active allocation is the newest non-revoked row.
- `live_orders(id, strategy, symbol, side, intended_notional, client_order_id UNIQUE, broker_order_id,
  status, submitted_ts)` — client_order_id is the primary identity; broker_order_id backfilled.
- `live_fills(id, activity_id UNIQUE, broker_order_id, strategy, symbol, qty, price, fill_ts)` —
  attributed via broker_order_id→live_orders; idempotent by activity_id.
- `live_activities(activity_id UNIQUE, type, symbol, amount, ts, raw)` — non-fill cash activities
  (DIV/fees/etc.) for NAV attribution + suspense.
- `live_reconcile_state(symbol, expected_qty, broker_qty, first_seen_cycle, status)` — pending
  mismatches for the grace window.
- `live_fill_cursor(name, cursor)` — the activities high-water mark.
- Per-strategy NAV series: a live-namespaced tick snapshot / peak distinct from the existing
  account-equity `tick_snapshots`/`strategy_peaks` (which keep paper/account-equity semantics) — new
  names/series so drawdown controls and reports do not mix meanings.

Derivations (pure, well-tested): `believed_positions(strategy)`, `realized_pnl`/`unrealized_pnl`,
`strategy_nav(strategy, marks)`, `account_expected_net()`.

---

## 8. Build decomposition (shadow-mode-first)

Each slice is its own spec-confirmed plan → subagent build → Codex review → PR, gated so real money
moves only once the controls beneath it exist.

- **Slice A — books foundation (SHADOW MODE).** The tables in §7; idempotent all-activity ingestion
  (atomic cursor + activity-id dedupe + overlap replay); order recording (client-id primary +
  broker-id backfill); average-cost P&L + NAV derivations; `live allocate` lifecycle + Σ≤equity.
  **Hard guard: refuse to put a 2nd strategy live (or to run the loop on >1 live strategy) until
  Slice C lands.** No change to sizing/flatten yet — pure additive books. Tests: fill attribution,
  crash-safe ingestion (simulated crash between pull and commit), cost-basis across long/short/flip/
  partial-close, allocation lifecycle + Σ≤equity enforcement.
- **Slice B — the portfolio loop + account reconcile + scoped cancel.** `live run-all` (re-verify →
  ingest → reconcile → sequenced tick); the classified reconcile with grace window + global-halt on
  persistent unexplained drift; scoped owned-order cancel (new broker cancel-by-id, account-wide
  reserved for halt). Still single-strategy sizing semantics carried over until Slice C. Tests:
  reconcile classification (tolerance / transient-then-clear / persistent→halt), scoped cancel does
  not touch siblings' orders, loop sequencing + crash mid-cycle.
- **Slice C — per-strategy sizing/NAV/drawdown + liquidation flatten + buying-power reservation.**
  Allocation as the sizing denominator; ledger-backed per-strategy snapshot; NAV drawdown; the
  liquidation workflow on breach; the shared buying-power pool reservation. **Lifts the ≤1-strategy
  guard.** Tests: sizing off allocation not account equity, NAV drawdown trip, liquidation offsets
  only the strategy's qty (sibling untouched), buying-power trim when the pool is short.

(Later, separate: max-order-size caps, dashboard, deeper corporate-action handling.)

---

## 9. The walls, end to end (unchanged + extended)

Every live order still requires: live keys in the trusted env; the `LiveAuthorization` tollbooth;
`verify_live_authorization` passing each cycle (the signed human go-live, re-verified per strategy);
not killed / globally-halted / revoked. Slice C adds: an allocation exists, the account reconciled
clean this cycle, and buying power was reserved. An autonomous agent has none of the first three.

---

## 10. Risks / open questions carried into the plans

- **Corporate actions** beyond what the activity stream reports (e.g. a split that adjusts broker qty
  before the activity posts) can transiently trip reconcile; the grace window + ingesting all
  activity types is the mitigation, with a documented manual-resume path.
- **Suspense bucket** policy (account-level cash that can't be attributed): blocks trading if
  material — threshold to be set in Slice A.
- **Throughput:** one sequenced loop over 100 strategies must complete within a cycle; if it doesn't,
  the loop cadence (not parallelism) is the lever. Measured, not assumed, in Slice B.
- **Live acceptance** is manual and documented (tiny capital, 1–2 strategies) — never CI.
