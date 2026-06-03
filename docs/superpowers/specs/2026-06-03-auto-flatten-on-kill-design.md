# Auto-Flatten on Kill (Sub-project 5, Slice B2b-1)

**Date:** 2026-06-03
**Status:** Accepted (pending implementation)
**Scope:** Close the live position when a strategy is killed — a `paper flatten` command and
auto-flatten when `trade-live` trips on a `RiskBreach`. Completes the kill→flat safety loop. The
global kill-switch and the live drawdown breaker are B2b-2.

---

## 1. Context & non-goals

B2a's live tick is validated (real orders fill on Alpaca). The kill-switch currently only halts —
a tripped strategy can still be holding a live position. This slice adds the missing safety half:
**a kill flattens the live book.** The kill-switch is generic (sim strategies use it too and hold
no Alpaca position), so flatten is a separate live-broker action — a dedicated `paper flatten`
command plus auto-flatten on a live `RiskBreach` — rather than wired into the generic trip.

**Non-goals (B2b-2):** the global ("halt-all") kill-switch; the live drawdown breaker (persisted
cross-tick peak equity — the main *automatic* flatten trigger); intra-session idempotency
(`client_order_id`); the per-tick account/positions snapshot optimization. **Out of scope:** a
"rebalance to cash but keep trading" op (flatten always halts), and `paper → live` (sub-project 6).

---

## 2. Components

| Module | Responsibility |
|---|---|
| `algua/execution/alpaca_broker.py` (modify) | `close_all_positions()` — `DELETE /v2/positions?cancel_orders=true` (Alpaca liquidates all positions + cancels open orders in one call); parse the 207 multi-status (via `_read`) and raise `BrokerError` if any per-position close failed. Reuses `_delete`. |
| `algua/cli/paper_cmd.py` (modify) | `paper flatten <name>` (close + halt, fail-safe ordering); `trade-live`'s `RiskBreach` path now auto-flattens. |

**Reuses:** `kill_switch.trip`, `_alpaca_broker_from_settings`, `json_errors`, the 207 per-item
parse pattern from `cancel_open_orders`. No new import contract (`execution`/`cli` already bound).

---

## 3. `close_all_positions()` (adapter)

```
DELETE {base}/v2/positions?cancel_orders=true
```
- Alpaca liquidates every open position (market orders) and cancels open orders; returns **207**
  multi-status — a JSON list of `{symbol, status, ...}` per position.
- `results = self._read(self._delete("/v2/positions?cancel_orders=true"), "/v2/positions", ok=(200, 207))`.
- `if isinstance(results, list): failed = [r for r in results if int(r.get("status", 500)) not in (200, 204)]`
  → if `failed`, `raise BrokerError(...)`.
- **No positions → empty list → no-op success** (flatten is idempotent).

---

## 4. `paper flatten <name>` — fail-safe ordering (trip *before* close)

```
algua paper flatten <name> [--actor agent|human]
```
1. `load_strategy(name)`; open DB; require stage `paper` (else `{ok:false}`); build broker via
   `_alpaca_broker_from_settings()` (creds checked). **All validation before any state change.**
2. `kill_switch.trip(conn, name, reason="flatten", actor=actor)` + audit (`action="flatten"`).
   **Halt first** — even if the close fails, the strategy is already stopped.
3. `try: broker.close_all_positions()` —
   - on `BrokerError` → `emit({"ok": False, "strategy": name, "kill_switch": "tripped",
     "flattened": False, "error": <str>})` and `raise typer.Exit(1)` (halted-but-not-flat is
     surfaced; the human closes via the Alpaca dashboard / retries).
   - else → `emit({"strategy": name, "kill_switch": "tripped", "flattened": True})`.

Decorated `@json_errors(ValueError, LookupError, BrokerError)` so a bad stage / missing creds
surface as `{ok:false}` *before* any trip.

---

## 5. `trade-live` auto-flatten on breach

Replace the current `RiskBreach` handler so a live breach also flattens (same fail-safe order):
```
except RiskBreach as exc:
    kill_switch.trip(conn, name, reason=exc.detail, actor="system")
    audit_append(action="kill_switch_trip", reason=f"{exc.kind}: {exc.detail}", strategy=name)
    flattened = True
    try:
        broker.close_all_positions()
    except BrokerError as fexc:
        flattened = False
        audit_append(action="flatten_failed", reason=str(fexc), strategy=name)
    emit({"ok": False, "kind": exc.kind, "kill_switch": "tripped",
          "flattened": flattened, "error": exc.detail})
    raise typer.Exit(1) from exc
```
A flatten failure is audited (`flatten_failed`) but never un-trips the switch (still halted).

---

## 6. Testing

- **Adapter `close_all_positions`** — 207 all-success (assert the request path carries
  `cancel_orders=true`); 207 with a per-position failure → `BrokerError`; empty list (no positions)
  → no-op success.
- **CLI `paper flatten`** — mocked broker → trips the switch (`paper show` reports tripped) +
  `flattened:true`; non-`paper` stage → `{ok:false}`; `close_all_positions` raising `BrokerError`
  → `{ok:false, kill_switch:"tripped", flattened:false}` exit 1, switch still tripped.
- **`trade-live` breach** — monkeypatched `run_tick` raises `RiskBreach`, mocked
  `close_all_positions` → emitted `flattened:true` + switch tripped.
- **Live acceptance (manual, documented, NOT CI)** — `algua paper flatten cross_sectional_momentum`
  against the real paper account → GOOGL/NVDA/AMZN closed, account back to cash, switch tripped.
- **Gate** — `pytest · ruff · mypy · lint-imports`.

---

## 7. Consequences

- The kill→flat loop is complete for live: a manual `paper flatten` or a live `RiskBreach` both
  halt **and** close the real position. The fail-safe ordering guarantees "halted" even if the
  close call fails.
- `close_all_positions` is the exact-exit primitive B1 deferred (vs the notional-sell
  approximation), reused by B2b-2's drawdown breaker (drawdown trip → flatten).
- Flatten is idempotent (empty book → no-op), so re-running it is safe.
- B2b-2 builds on this: the drawdown breaker becomes the main *automatic* flatten trigger, and the
  global kill-switch wraps the same halt + flatten across all strategies.
