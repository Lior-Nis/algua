# Wall-Clock Live Tick Loop (Sub-project 5, Slice B2a)

**Date:** 2026-06-03
**Status:** Accepted (pending implementation)
**Scope:** A single wall-clock "tick" that drives the (live-validated) `AlpacaPaperBroker`: read
Alpaca as the source of truth, compute target weights from the latest closed session, submit
market-order deltas, record + audit. `algua paper trade-live <name> --snapshot <id>`. No
auto-flatten, no global kill-switch, no drawdown breaker — those are B2b.

---

## 1. Context & non-goals

B1 shipped `AlpacaPaperBroker` (account/get_positions/submit notional orders), live-validated
against a real paper account. B2a builds the loop that drives it, distinct from the synchronous
sim `run_paper`: Alpaca fills **asynchronously**, so the loop treats **Alpaca as the source of
truth** — each tick reads live positions/account, computes a target, diffs vs actual positions,
submits market deltas, and lets the next tick reconcile implicitly by re-reading positions.

A tick is **one invocation**; a daily cadence is the scheduler's job (cron), reusing two existing
commands: `algua data ingest-bars --provider alpaca …` (recent bars → snapshot) then
`algua paper trade-live <name> --snapshot <id>`. So bars enter through the existing
`_select_provider(snapshot=…)` seam — no new live data provider is built.

**Decisions locked in brainstorming:** Alpaca = source of truth (no per-order fill polling);
bars via the Alpaca-backed `data ingest-bars` → `--snapshot` seam; B2 sub-sliced (this is B2a).

**Non-goals (B2b):** auto-flatten-on-kill (close-position endpoint); the global kill-switch; the
live drawdown breaker (needs persisted cross-tick peak equity); intra-session idempotency; the
"snapshot account+positions once per tick" optimization (per-symbol reads are fine at daily cadence
for a small universe). **Out of scope:** `paper → live` (sub-project 6, TOTP).

---

## 2. Components

| Module | Responsibility |
|---|---|
| `algua/execution/alpaca_broker.py` (modify) | Add `cancel_open_orders() -> None` (`DELETE /v2/orders`), wrapped in the existing `BrokerError` handling, to clear stale/unfilled orders at tick start. |
| `algua/live/live_loop.py` (new) | `TickResult` dataclass; `run_tick(strategy, broker, provider) -> TickResult` — pure orchestration of one tick over an injected `AlpacaPaperBroker` + bar provider. |
| `algua/cli/paper_cmd.py` (modify) | `algua paper trade-live <name> --snapshot <id> [--start/--end]` — gate, wire broker + provider, run the tick, persist submissions + audit, emit JSON. |

**Reused (one small refactor):** `check_gross_exposure` already lives in `risk/limits`; the
**long-only check is currently inline in `run_paper`**, so the plan **extracts it into
`risk/limits.check_long_only(weights, strategy_name)`** (raising `RiskBreach("long_only", …)`) and
points both `run_paper` and `run_tick` at it — DRY, no behavior change (the existing long-only test
still passes). Also reused as-is: `kill_switch` (the gate + trip-on-breach), `_select_provider`
(snapshot path), `_alpaca_broker_from_settings`, the bar-schema, `target_weights`, `OrderIntent`. `run_tick` takes `AlpacaPaperBroker` concretely (the live loop is
broker-specific, just as `run_paper` takes `SimBroker`); the provider is injected as `Any`
(import-light). Boundary: `algua/live` may import `algua/execution` (loop orchestrates execution);
existing import contracts already forbid `cli` from being imported by these layers.

---

## 3. Tick data flow

`run_tick(strategy, broker, provider)`:
1. `bars = provider.get_bars(strategy.universe, start, end, "1d").sort_index()`; decision session
   `t` = last available bar timestamp; `view = bars.loc[:t]`.
2. `weights = target_weights(view)`.
3. **Guards:** if the number of available sessions `< strategy.execution.warmup_bars` → submit
   nothing (return a `TickResult` with `submitted=[]`). Else `check_long_only` (negative weight →
   `RiskBreach("long_only")`) and `check_gross_exposure(weights, max_gross)`.
4. `broker.cancel_open_orders()`.
5. For each symbol in `set(weights) ∪ set(broker.get_positions().index)` (dropped names get a
   `target_weight=0` exit): build `OrderIntent(symbol, side, target_weight, decision_ts=t)`
   (`side` nominal: BUY if `target_weight>0` else SELL — the adapter recomputes from the live
   delta) and `order_id = broker.submit(intent)`. Collect `{symbol, side, target_weight, order_id}`
   for every non-`"noop"` result.
6. Return `TickResult(decision_ts=t, target_weights={…}, positions_before={…from Alpaca…},
   submitted=[…])`.

`positions_before` is captured from `broker.get_positions()` at the start (Alpaca truth, for the
emitted summary).

---

## 4. CLI: `paper trade-live`

```
algua paper trade-live <name> --snapshot <id> [--start D --end D]
```
1. `load_strategy(name)`; open the registry DB; require stage `paper` (else `{ok:false}`);
   require the kill-switch not tripped (else `{ok:false}` pointing at `paper resume`).
2. `broker = _alpaca_broker_from_settings()` (raises if creds missing); `provider =
   _select_provider(demo=False, snapshot=<id>)`.
3. `try: result = run_tick(strategy, broker, provider)` — on `RiskBreach`: `kill_switch.trip(…,
   actor="system")` + audit + `{ok:false, kind, kill_switch:"tripped"}` exit 1 (the fail-closed
   path, identical to `run_paper`).
4. Persist each `submitted` order to `paper_orders` (`status="submitted"`,
   `broker_order_id`=order_id) + an audit row (`action="trade_live"`).
5. `emit({strategy, decision_ts, target_weights, positions_before, submitted})`.

Decorated `@json_errors(ValueError, LookupError, BrokerError)` so missing creds, a bad stage, and
Alpaca failures all render as `{ok:false}`.

**Persistence note:** `paper_orders` records the *submitted* live orders (no synchronous
`paper_fills` — Alpaca fills async and is the source of truth). The sim-oriented `paper show`
(derives positions from `paper_fills`) does **not** reflect live positions; live position
visibility is this tick's `positions_before` + `paper account`, with a dedicated `paper positions`
(reading Alpaca) left to slice C.

---

## 5. Error handling

- **Kill-switch tripped** → `{ok:false}`, no tick (checked before any broker call).
- **`RiskBreach`** (long-only/gross) → trip the kill-switch + audit + `{ok:false}`; nothing
  persisted.
- **`BrokerError`** (Alpaca network/non-2xx, from B1's wrapping) → `{ok:false}`. A failure
  mid-tick may leave some orders already at Alpaca; under source-of-truth that's acceptable — the
  next tick re-reads actual positions and reconciles. The failed tick persists nothing locally.
- **Warm-up not met / empty target** → normal emit with `submitted=[]` (not an error).

---

## 6. Testing

- **Unit `run_tick`** — a fake broker (records `submit` calls, returns ids; canned
  `get_positions`; counts `cancel_open_orders`) + a fake snapshot provider (bar-schema frame):
  buys from a target; exits a dropped symbol (`target_weight=0`); warm-up-not-met → `submitted=[]`;
  gross breach → `RiskBreach`; `cancel_open_orders` called once before submits.
- **Adapter** — `cancel_open_orders()` issues `DELETE /v2/orders` and raises `BrokerError` on
  non-2xx (mock `requests`).
- **CLI `trade-live`** — refuses on a non-`paper` stage and on a tripped kill-switch
  (`{ok:false}`); a fully-mocked run (broker + provider patched) emits the `TickResult` and persists
  `paper_orders`; a `RiskBreach` trips the kill-switch.
- **Live one-tick smoke (documented, NOT CI)** — `algua data ingest-bars --provider alpaca
  --symbols AAPL,MSFT,NVDA,AMZN,GOOGL --start … --end …` → `algua paper trade-live
  cross_sectional_momentum --snapshot <id>` → confirm a market order appears in the Alpaca paper
  dashboard. B2a's real acceptance test.
- **Gates** — `pytest · ruff · mypy · lint-imports` (no new import contract; `live`/`execution`
  already covered).

---

## 7. Consequences

- The same `target_weights` + `risk/limits` + `kill_switch` now drive a **real broker**, end to
  end — the research→execution seam holds across sim and live.
- Alpaca-as-source-of-truth keeps the loop simple and crash-safe: there is no local position
  ledger to drift; a re-run reconciles by reading actual positions.
- The bars seam reuses `data ingest-bars` + `--snapshot`, so the daily scheduler is two existing
  commands + cron — no new data-fetch code, and the tick stays pure/testable.
- B2b builds directly on this: auto-flatten calls the adapter's close-position path on a kill, the
  global switch wraps the gate, and the drawdown breaker adds persisted peak equity — all on top of
  a working live tick.
