# Multi-Strategy Live Accounting — Slice C (Per-Strategy Sizing & Risk) Design

**Date:** 2026-06-06
**Status:** Accepted (pending implementation)
**Scope:** The finale of the multi-strategy live-accounting OMS — make each live strategy size off
its own **allocation** (not account equity), drawdown off its own **NAV**, liquidate per-strategy on
a breach, reserve account buying power across the loop, and **lift the ≤1-live guard** so many
strategies trade live on one shared account.

Both the sub-project design and this slice were adversarially reviewed by Codex; the CRITICAL
(synthetic exposure on a netted account) and the HIGH/MEDIUM findings are folded in below.

---

## 1. The model — virtual subaccounts on one netted account (the framing that makes this sound)

One Alpaca account is the custodian and holds the **netted** book (strategy A long 100 AAPL + B
short 100 AAPL → the broker shows flat). Per-strategy positions are **internal attributions**
(virtual subaccounts / books-and-records), NOT independent broker positions. This is the standard
multi-strategy model and the only one that scales to ~100 strategies that share symbols.

**Consequence — risk is enforced at TWO levels, both first-class:**
- **Per-strategy gross** (from the ledger): sizing denominator, NAV, drawdown, liquidation — the
  strategy's virtual portfolio.
- **Account net** (from the broker): the Slice-B reconcile (Σ believed == broker net) + the
  buying-power reservation — the real money.

This framing makes **ledger-offset liquidation correct**: closing strategy A's believed quantity
moves the account net by exactly A's contribution, leaving siblings' positions intact. It also means
per-strategy NAV/drawdown are *synthetic* (a virtual-portfolio view), which is exactly what we want
for evaluating and risk-gating a single strategy — provided the account-net level is independently
bounded (it is, by reconcile + BP reservation).

---

## 2. Settled decisions (brainstorming + Codex review)

- **Sizing denominator = `min(allocation, NAV)`** — capped at the fixed allocation (never compounds
  up on gains; operator controls capital) but de-risks down when NAV falls below it, so effective
  leverage never creeps above 1× NAV on losses (Codex HIGH #2).
- **Drawdown basis = per-strategy NAV** (allocation + realized + unrealized), distinct from the
  account-equity peak.
- **Marks = latest closed bar** from the strategy's own data provider; **fail-closed** — a held
  symbol with a missing/stale/non-positive mark skips that strategy this cycle (NO average-cost
  fallback, which would hide a loss and suppress the drawdown trip — Codex MEDIUM #7).
- **Liquidation = single-shot offset** sized to the *fresh* believed qty (re-ingest fills first) +
  trip the kill-switch; **resume requires the strategy ledger-flat** (`believed_positions` empty) so
  a partial-fill residual can't be silently resumed into (Codex HIGH #5).
- **Buying power = shared-pool reservation**, drawn down by each strategy's **gross** buy notional
  (sells never offset buys nor add back within a cycle — Codex HIGH #3); trim/skip + persist the
  reason; deterministic registry loop order (audited; pro-rata fairness a future option — Codex #9).
- **Live trades route only through `run-all`** (the ingest/reconcile gate); `trade-tick` is no longer
  a reconcile-less live path (Codex MEDIUM #6).

---

## 3. The `run_tick` seam (C1)

`run_tick` currently calls `broker.snapshot(universe)` → `TickSnapshot(equity, market_values, qtys)`
and uses `snap.equity` as BOTH the sizing denominator and the drawdown basis. Slice C adds one hook:

```
TickHooks.live_snapshot: Callable[[pd.DataFrame], tuple[SizingSnapshot, float]] | None
```

When supplied (live), `run_tick`:
- builds the sizing snapshot from the hook (not the broker): `SizingSnapshot(equity = min(allocation,
  NAV), qtys = believed_positions, market_values = qty × latest-bar-close)`;
- sizes + computes `current_weights = market_value / snap.equity` off that snapshot equity;
- checks drawdown against the **NAV** the hook returns (decoupled from the sizing denominator);
- the warmup / no-bars early-return paths report **ledger** positions in live mode (not
  `broker.snapshot`).

`SizingSnapshot` is a distinct name from the broker's `TickSnapshot` (Codex LOW #10): one is the
strategy's ledger belief, the other is broker truth. Paper passes no hook → unchanged (`broker.snapshot`
+ equity for both). The hook receives the already-fetched `bars` so marks need no extra network.

`build_live_sizing_snapshot(conn, strategy, allocation, bars) -> (SizingSnapshot, nav)` lives in a new
`algua/execution/live_sizing.py`, reusing `believed_positions` + `position_pnl` (Slice A). It raises a
fail-closed error (→ skip the strategy) when a held symbol lacks a usable mark.

---

## 4. C1 — per-strategy sizing, NAV drawdown, liquidation (keeps the ≤1-live guard)

`_run_strategy_tick` (the shared engine from Slice B):
- looks up `active_allocation(conn, strategy_id)`; a strategy with no allocation is skipped+flagged
  (can't happen past the go-live guard, but defensive);
- passes `live_snapshot=lambda bars: build_live_sizing_snapshot(conn, name, allocation, bars)`;
- drawdown uses the per-strategy **NAV peak** (a new `live_nav_peaks` series keyed by strategy,
  separate from `strategy_peaks`); `update`/`get` mirror the existing peak helpers.
- **Minimal BP preflight** (Codex MEDIUM #8): before submitting, read account buying power; if the
  strategy's intended gross buys exceed it, trim/skip + flag — even for one strategy, since
  allocation can exceed real BP after losses/withdrawals.

**Liquidation on `RiskBreach`** (replaces account-wide cancel + `close_positions(universe)`):
1. trip the per-strategy kill-switch;
2. scoped-cancel the strategy's own open orders (Slice B `_scoped_cancel`);
3. re-ingest fills (so the believed qty is fresh);
4. for each held symbol in `believed_positions(strategy)`, submit ONE offsetting market order sized
   to that believed qty (sell longs / buy back shorts — never flips);
5. the next `run-all` cycle's reconcile + the resume gate verify the residual.

**Resume gate** (Codex HIGH #5): `paper resume` / `resume-all` for a live strategy refuses unless
`believed_positions(strategy)` is empty — a partial-fill residual blocks resume until re-flattened.

---

## 5. C2 — buying-power reservation + lift the ≤1-live guard

- `run-all` snapshots account buying power once at cycle start → a shared pool.
- As the loop ticks strategies in deterministic registry order, each strategy reserves its **gross**
  buy notional from the pool; a buy the pool can't cover is **trimmed** (sized down) or **skipped**;
  sells do not add back or offset within the cycle.
- Every trim/skip persists a row (strategy, symbol, intended_notional, submitted_notional, reason)
  so a starved strategy is visible to the operator (Codex #11).
- The minimal C1 BP preflight generalizes into the shared pool (one mechanism).
- **Lift the ≤1-live guard:** the go-live transition (Slice A) drops the "only one live strategy"
  refusal — now that per-strategy sizing, the reconcile, the reservation, and per-strategy
  liquidation all exist. The allocation requirement + Σ≤equity + signed go-live remain.

C1 is safe to ship under the ≤1-live guard (one strategy can't over-commit beyond its allocation, and
Σ allocations ≤ equity holds); only C2 opens the account to many strategies — after the pool that
makes that safe exists.

---

## 6. Data model (schema bump 14→15)

- `live_nav_peaks(strategy TEXT PRIMARY KEY, peak REAL, updated_ts TEXT)` — per-strategy NAV peak for
  the drawdown breaker (live-namespaced; the account-equity `strategy_peaks` keeps its meaning).
- `live_reservations(id, cycle, strategy, symbol, intended_notional, submitted_notional, reason, ts)`
  — append-only trim/skip audit (C2).

No change to Slice A/B tables. Derivations stay pure/tested.

---

## 7. Testing

- **Sizing off min(allocation, NAV):** a strategy at NAV < allocation sizes off NAV; at NAV ≥
  allocation sizes off allocation; `current_weights` use the same denominator.
- **NAV drawdown:** NAV below the breaker threshold raises `RiskBreach`; the NAV peak ratchets;
  account equity is NOT the basis.
- **Fail-closed marks:** a held symbol with no/zero/negative mark → the strategy is skipped, not
  sized off average cost.
- **Liquidation:** a breach trips the kill-switch, scoped-cancels, re-ingests, and submits offsets
  sized to the fresh believed qty per held symbol (sibling positions untouched at the account net).
- **Resume gate:** resume refuses while `believed_positions` is non-empty; succeeds once flat.
- **BP preflight / pool (C2):** buys are trimmed/skipped when the pool can't cover; gross (not net)
  buys are reserved; a trim persists a `live_reservations` row.
- **Guard lift (C2):** a 2nd strategy can go live (allocation + signature still required); pre-C2 the
  guard still refuses.
- **Paper unchanged:** no `live_snapshot` hook → `run_tick` uses `broker.snapshot` + equity for both
  sizing and drawdown (existing paper/live_loop tests pass).
- **Live acceptance (manual, documented, NOT CI):** two tiny-allocation strategies sharing a symbol;
  confirm per-strategy attribution, a breach flatten that leaves the sibling, and the guard lift.

---

## 8. Risks / open questions

- **Synthetic exposure is intentional:** per-strategy NAV/drawdown measure the *virtual* portfolio;
  the account-net level is bounded separately by reconcile + BP. Documented so it's not mistaken for
  real per-strategy broker positions.
- **Loop-order fairness:** deterministic registry order can systematically favor early strategies
  when BP is scarce; acceptable + audited for now, pro-rata scaling is the future refinement.
- **Mark staleness:** "latest closed bar" assumes the loop ticks near bar close; a far-intraday tick
  marks stale. Fail-closed on missing marks bounds the worst case; intraday quotes remain a future
  option.
- **C1/C2 boundary:** C1 ships the minimal single-strategy BP preflight so it is safe alone under the
  ≤1-live guard; C2 generalizes it and only then lifts the guard.
