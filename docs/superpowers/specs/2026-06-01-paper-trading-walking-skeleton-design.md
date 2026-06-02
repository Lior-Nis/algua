# Paper Trading — Walking Skeleton (Sub-project 5, Slice 1)

**Date:** 2026-06-01
**Status:** Accepted (pending implementation)
**Scope:** The first slice of sub-project 5 (execution + paper trading). Build the **paper loop end
to end against a local simulated broker** — replay a `paper`-stage strategy bar-by-bar through the
*same* live-loop code, generate orders, fill them deterministically, persist order-intent state +
fills + an audit trail, and reconcile. No external broker, no real-money path.

---

## 1. Context & non-goals

Sub-projects 1–4 are complete: research takes a strategy to `shortlisted`, and an agent can
transition `shortlisted → paper`. This slice gives the agent something to *run* at `paper`.

Engine model (architecture spec §3): a strategy is a pure `target_weights(view) → Series`; the
backtester runs it vectorized, and the **live/paper loop calls the same function on the latest
closed bar and diffs target weights against current positions to generate orders**. Parity is of
*semantics* (the `ExecutionContract`'s `t→t+1` rule), not PnL. This slice builds that loop.

**Decisions locked in brainstorming:**
- **Local simulated broker first** (deterministic, offline, CI-testable) — the `--demo` philosophy
  applied to execution. The real Alpaca paper adapter is a later slice.
- **Replay-mode loop** — step the same loop code over synthetic/historical bars. Wall-clock
  scheduling against the real broker comes with the Alpaca adapter.
- **Minimal core** — loop + sim broker + order-intent state + lean reconcile + audit + CLI.

**Non-goals (later slices of sub-project 5):** the Alpaca paper adapter + wall-clock scheduler;
risk limits + kill-switch; the warm-up gate; deep status/health commands; slippage/partial-fill
modeling. **Out of scope entirely:** the `paper → live` transition (sub-project 6, human + TOTP).

---

## 2. Components

New modules following the spec's `execution/` `live/` `audit/` layout. Each is small and testable.

| Module | Responsibility |
|---|---|
| `algua/execution/sim_broker.py` | `SimBroker`: holds `cash` + share `positions`, marks to market on provided bars, **fills submitted orders at the next bar's open** (full fill, no slippage). Implements the contracts `Broker` protocol (`submit`, `get_positions`) plus sim-only `mark(bar)` / `fill_pending(open_prices)` the replay loop drives. Applies **sells before buys** so freed cash funds buys. |
| `algua/live/paper_loop.py` | The bar-stepping loop. Core unit `process_bar(strategy, broker, view) -> list[OrderIntent]`; `run_paper(...)` replays it over `[start, end]`. Pure orchestration over injected `Broker` + provider. |
| `algua/execution/order_state.py` | Persist `OrderIntent`s + fills; derive positions from fills; `reconcile(derived, broker_positions) -> bool`. |
| `algua/audit/log.py` | Lean append-only audit: `append(conn, actor, action, reason, strategy)` + `read(...)`. |
| `algua/cli/paper_cmd.py` | `algua paper run <name> --demo --start D --end D [--cash N]` and `algua paper show <name>`. Emits JSON. |
| `algua/registry/db.py` (extend) | `migrate()` adds tables `paper_orders`, `paper_fills`, `audit_log`. Paper state lives in the same SQLite registry DB (single source of truth). |

The strategy's `target_weights` is reused **unchanged** — one signal definition, one execution
contract.

### Boundaries (new import-linter contracts)
- `algua.execution`, `algua.live`, `algua.audit` may import `contracts`, `registry`, `data` — but
  **not** `algua.cli`.
- `algua.backtest` stays off `algua.execution` and `algua.live` (research and execution lanes
  don't cross; mirrors the existing backtest-off-data/cli/tracking contracts).

---

## 3. Per-tick data flow

"Process one closed bar `t`":
1. View = point-in-time bars up to and including closed bar `t` (strategy universe).
2. `target_weights(view) → Series[symbol → weight]` (empty Series if insufficient history → no-op tick).
3. Determine current weights from positions marked at bar `t` close (`equity = cash + Σ shares×close`).
4. Compare target vs current weights → per symbol that needs to move, emit
   `OrderIntent(symbol, side, target_weight, decision_ts=t)`. **Weights are decided only from data
   ≤ `t`** (look-ahead-safe); share sizing happens at the fill, not here.
5. `submit()` each intent → recorded pending; persist to `paper_orders` + audit.
6. Advance to `t+1`: the broker **sizes and fills** at `t+1` **open** — `equity` marked at the open,
   `target_shares = floor(target_weight × equity / open)`, `qty = target_shares − current`, **sells
   before buys**, full fill, never negative cash. Update cash + positions; persist to `paper_fills`
   + audit.
7. Reconcile: positions derived from `paper_fills` must equal `broker.get_positions()`.

**Replay** over `[start, end]`: tick at each bar that has a successor (so every order can fill); the
final bar emits no new orders. Emit `{strategy, orders, fills, final_positions, final_equity,
reconcile_ok}`. `--cash` default $100k; long-only (weights ≥ 0). The loop **operates within `paper`
stage and never changes the registry stage**.

---

## 4. Error handling & invariants

- **Lifecycle gate.** `paper run <name>` requires the strategy at `paper` stage; otherwise a
  `{ok:false}` JSON error (via `json_errors`). The loop never transitions stages itself.
- **`t→t+1` enforcement (the safety invariant).** Orders decided on closed bar `t` fill **only** at
  `t+1` open — never same-bar. A test asserts every fill's timestamp is the bar strictly after its
  order's `decision_ts`. This mirrors the backtester's anti-look-ahead rule for the live path.
- **No-op ticks.** An empty `target_weights` (insufficient history) produces no orders; the loop
  continues.
- **Cash safety.** Size at the `t+1` open (`floor(target_weight × equity / open)`); apply sells
  before buys so freed cash funds buys; never drive cash negative (clamp the last buy if rounding
  would). Long-only.
- **Reconcile mismatch.** If derived ≠ broker-reported positions, the run reports
  `reconcile_ok=false` (always true for the sim; the check is the seam the real adapter will need).
- **Determinism.** Synthetic provider is seeded and fills at known open prices, so a replay is
  fully reproducible — the basis for the e2e tests.

---

## 5. Testing

All green in CI, offline:
- **Unit — `SimBroker`:** submit→fill-at-next-open; cash/positions update; sells-before-buys frees
  cash; mark-to-market; `get_positions`; no same-bar fill.
- **Unit — `order_state`:** persist orders/fills; derive positions from fills; `reconcile` true on
  match, false on a forced mismatch.
- **Unit — `paper_loop.process_bar`:** a tiny synthetic bar set → expected `OrderIntent`s (incl. an
  empty-weights no-op tick).
- **Unit — `audit`:** append + read round-trip.
- **e2e (`paper run`):** transition `cross_sectional_momentum` to `paper`, then `paper run --demo`
  → JSON summary has orders > 0, fills, final positions, `reconcile_ok=true`; `paper_orders` /
  `paper_fills` / `audit_log` rows persisted; `paper show` reflects them.
- **Negative e2e:** `paper run` on a non-`paper`-stage strategy → `{ok:false}` exit 1.
- **`t→t+1` e2e:** assert no fill shares a timestamp with its decision bar.
- **Gates:** `pytest · ruff · mypy · lint-imports` (incl. the new execution-boundary contracts).

---

## 6. Consequences

- The live/paper loop reuses the research `target_weights` unchanged — proving the
  "one signal definition, one execution contract" architecture across the research↔execution seam.
- All paper state (orders, fills, audit) lives in the existing registry DB, so it's queryable and
  dashboard-ready alongside lifecycle stages.
- The `Broker` protocol + reconcile seam are built now, so the Alpaca paper adapter (next slice)
  slots in behind the same interface without touching the loop.
- Deferring risk/kill-switch + warm-up to later slices keeps this slice a true walking skeleton;
  the spec notes them as the immediate next work, validated in paper before live hardening.
