# `live trade-tick` — Real-Money Loop (Sub-project 6, live-execution slice 3)

**Date:** 2026-06-05
**Status:** Accepted (pending implementation)
**Scope:** The command that actually places real-money orders on Alpaca's live venue — one wall-clock
tick, gated by `verify_live_authorization`, driving the existing `run_tick` against `AlpacaLiveBroker`,
reusing the kill-switch / drawdown breaker / scoped-flatten safety. The highest-stakes slice.

---

## 1. Context & non-goals

Slices 1–2 built the trade-time wall (`verify_live_authorization`) and the gated `AlpacaLiveBroker`
(constructable only with a `LiveAuthorization`, only over https to the live host). Nothing constructs
the live broker yet. This slice wires them into a `live trade-tick` command that re-verifies the
human signature and trades. `run_tick`, the drawdown breaker (`check_drawdown` + `strategy_peaks`),
the per-strategy kill-switch, the global halt, and the scoped flatten are all reused unchanged.

**Non-goals (slice 4):** capital caps, max-order-size limits, a structured live-orders ledger /
dashboard table, position reconciliation against a local book. **Out of scope:** changing `run_tick`
or the paper path's behaviour; any new go-live ceremony (that's slice 1).

---

## 2. Design decisions (settled in brainstorming)

- **Re-verify cadence:** full `verify_live_authorization` (ssh-keygen + recomputed identity) once at
  the START of the tick (gates building the broker), PLUS a cheap unrevoked-row check in the
  pre-submit `should_halt` hook so a mid-tick revocation aborts before the next order — without
  re-running ssh-keygen per order. (Live trade-tick is one tick per invocation, so the full
  signature re-verify already happens every tick.)
- **Alpaca is the sole source of truth:** the tick decides off the live broker snapshot each tick;
  NO local position ledger and NO DB-vs-broker reconcile (`derived_positions=None`).
  `client_order_id` idempotency already prevents double-orders. Live orders are recorded for
  **audit only** (no `paper_orders` writes — avoids paper/live ledger mixing).
- **No extra confirmation flag:** the human's signature IS the confirmation; a CLI flag an agent
  could pass adds friction, not safety.

---

## 3. Command + gate — `algua live trade-tick`

A new `live` Typer group (`algua/cli/live_cmd.py`):
```
algua live trade-tick <name> --snapshot <id> [--start D --end D --max-drawdown F]
```
Decorated `@json_errors(ValueError, LookupError, BrokerError, LiveAuthorizationError)`. Body, in order:
1. `repo = SqliteStrategyRepository(conn)`.
2. `authorization = verify_live_authorization(conn, repo, name, ALLOWED_SIGNERS_PATH)` — THE wall:
   re-verifies the SSH signature against the trust anchor for the strategy's CURRENT artifact and
   requires `Stage.LIVE`. Raises `LiveAuthorizationError` → `{ok:false}` exit 1 otherwise.
3. refuse if `kill_switch.is_tripped(conn, name)` or `global_halt.is_engaged(conn)` (`{ok:false}`).
4. `strategy = load_strategy(name)`; `broker = _alpaca_live_broker(authorization)`;
   `provider = _select_provider(False, snapshot)`.
5. drive `run_tick(...)` with the hooks in §4; handle `TickHalted` / `RiskBreach` per §5.

`--max-drawdown` validation `0 < x <= 1` mirrors paper.

## 4. The live broker factory + hooks

`_alpaca_live_broker(authorization: LiveAuthorization) -> AlpacaLiveBroker`:
```python
s = get_settings()
if not s.alpaca_live_api_key or not s.alpaca_live_api_secret:
    raise ValueError("Alpaca LIVE credentials not configured (ALGUA_ALPACA_LIVE_API_KEY/SECRET)")
return AlpacaLiveBroker(authorization, s.alpaca_live_api_key, s.alpaca_live_api_secret,
                        base_url=s.alpaca_live_url)
```
Missing live keys → can't trade (the env wall). The token threads through the tollbooth.

`TickHooks`:
- `client_order_id_for=client_order_id` (deterministic, idempotent).
- `on_submitted` → `audit_append(action="live_order", reason=f"{side} {symbol} {order_id}", strategy=name)` — audit only, no `paper_orders` ledger.
- `peak_equity=get_peak_equity(conn, name)` — the drawdown breaker protects live for free.
- `derived_positions=None` — Alpaca is the sole source of truth; no reconcile.
- `should_halt = lambda: kill_switch.is_tripped(conn, name) or global_halt.is_engaged(conn) or not live_gate.authorization_active(conn, authorization)` — the cheap mid-tick revoke check.

New `live_gate.authorization_active(conn, authorization) -> bool`: a cheap query — does an UNREVOKED
`live_authorizations` row matching `authorization.strategy_id` + its three hashes still exist? No
ssh-keygen, no hash recompute.

## 5. Breach handling (same shape as paper, on the live broker)

- `TickHalted` (switch tripped mid-tick before submit): audit `live_trade_tick_halted`, emit
  `{ok:false, halted:true}`, exit 1. Nothing was sent.
- `RiskBreach` (drawdown / exposure): `_trip` the kill-switch + audit; **scoped** flatten —
  `broker.cancel_open_orders()` + `broker.close_positions(strategy.universe)` (a `BrokerError` here is
  audited `flatten_failed` and surfaced, never un-tripped). Account-wide `close_all_positions` stays
  reserved for `halt-all` only (Codex's note). Emit the breach payload, exit 1.

On success: `update_peak_equity` + `record_tick_snapshot` (the live equity curve, read by
`paper show`) + `audit_append(action="live_trade_tick", reason=f"{n} orders")`.

## 6. Shared scaffolding

To avoid duplicating the breach scaffolding without touching the money-critical paper command,
`live_cmd` **imports** the shared helpers `_trip` and `_breach_payload` from `paper_cmd` (one copy,
no refactor of `paper_cmd`). It also imports `run_tick`/`TickHooks`/`TickHalted`/`RiskBreach`/
`client_order_id`/`get_peak_equity`/`update_peak_equity`/`record_tick_snapshot`/`audit_append`/
`kill_switch`/`global_halt`/`live_gate`/`AlpacaLiveBroker`/`verify_live_authorization`/
`ALLOWED_SIGNERS_PATH`/`load_strategy`/`_select_provider`/`ok`/`registry_conn`/`utc`/`json_errors`.
The `live` Typer group is registered onto the root app the same way `paper` is (the CLI entry imports
`live_cmd`).

The per-order revoke check requires one addition to `run_tick`: re-check `should_halt` at the TOP of
each submit-loop iteration (today it is checked only once before the loop), raising `TickHalted` so a
mid-loop halt/revoke stops further orders. This is behaviour-preserving for paper (an extra cheap
check that only fires when already halting).

## 7. The walls, end to end

To place a live order an agent must pass ALL of: **(1)** live keys present in the env (trusted
context only); **(2)** a `LiveAuthorization` to build the broker (tollbooth); **(3)**
`verify_live_authorization` passing at tick start (the real wall — re-verifies the human signature
against the CODEOWNERS trust anchor for the current code); **(4)** the authorization still unrevoked
at each order, and not killed / globally halted. The agent has none of (1)–(3).

---

## 8. Testing (no real orders)

- **Refused without authorization** — a strategy not at `live` (or with no matching authorization)
  → `live trade-tick` exits 1 `{ok:false}`, no broker built.
- **Happy path (mocked transport)** — seed a strategy at `Stage.LIVE` + a real `live_authorizations`
  row (hermetic `ssh-keygen` key, enrolled, signed) + live keys in env; monkeypatch the broker's
  `requests` (or `run_tick`); assert the tick runs and submits to `api.alpaca.markets`, an audit
  `live_order` row per submission, and a `tick_snapshot` written.
- **Killed / globally halted** → refused `{ok:false}`.
- **Mid-tick revocation** — `authorization_active` returns False once the row is revoked; a
  `should_halt`-driven `TickHalted` aborts before further orders (unit-test `authorization_active`
  True→False on revoke).
- **RiskBreach** — a monkeypatched `run_tick` raising `RiskBreach` trips the kill-switch + calls the
  live broker's `cancel_open_orders` + `close_positions(universe)`; emits `kind` + `liquidation_submitted`.
- **Missing live keys** → `ValueError` `{ok:false}`.
- **`authorization_active`** — unit: matching unrevoked row → True; revoked / absent → False.
- **Gate** — `pytest · ruff · mypy · lint-imports` (contracts kept, 0 broken).
- **Live acceptance (manual, documented, NOT CI)** — out of scope to run; documented procedure:
  enroll, go live, run `live trade-tick` against the real account with a tiny universe, confirm a
  real order, then revoke/flatten.

---

## 9. Consequences

- The platform can now place real-money orders — but only through a command that re-verifies a human
  signature against the CODEOWNERS trust anchor every tick, with the same kill-switch / drawdown /
  flatten / halt-all safety as paper, and Alpaca as the source of truth.
- An autonomous agent cannot reach it: it lacks the live keys, the signature, and runs nowhere the
  vetted `main` go-live code executes.
- Slice 4 (operational hardening: capital caps, max order size, a live ledger/dashboard) builds on
  this; the loop and its gate are the foundation.
