# Alpaca Paper Broker Adapter (Sub-project 5, Slice B1)

**Date:** 2026-06-03
**Status:** Accepted (pending implementation)
**Scope:** A faithful `requests`-based wrapper of the Alpaca **paper-trading** REST API, implementing
the `Broker` protocol, plus a `paper account` connectivity smoke. No trading loop runs in this
slice — it de-risks the external integration in isolation, mirroring how the sim broker came before
the loop.

---

## 1. Context & non-goals

Sub-project 5 so far is fully offline: the sim-broker replay loop (`run_paper`) plus the safety
guards. Slice B brings a real broker. B1 builds just the **adapter**.

Key architectural fact: `run_paper` is synchronous/replay (`broker.fill_pending(opens, …)` fills
immediately at `t+1` open). **Alpaca is asynchronous** — you submit an order and fills happen later
on Alpaca's clock. So the Alpaca path cannot reuse `run_paper`; a wall-clock driver is needed. That
driver — and the async fill lifecycle it manages — is **B2**, not this slice.

The existing Alpaca *data* provider (`algua/data/providers/alpaca.py`) is `requests`-based (no SDK),
with keys from `settings.alpaca_api_key` / `alpaca_api_secret`. B1 mirrors that pattern for the
*trading* API.

**Non-goals (B2):** the wall-clock scheduler/loop driving this adapter; the async fill/reconcile
lifecycle; auto-flatten-on-kill (exact position close); the global kill-switch; limit orders;
order listing/cancellation.

---

## 2. Components

| Module | Responsibility |
|---|---|
| `algua/config/settings.py` (modify) | Add `alpaca_paper_url: str = "https://paper-api.alpaca.markets"` (the trading API base, distinct from the existing market-data `alpaca_data_url`). Reuse `alpaca_api_key` / `alpaca_api_secret`. |
| `algua/execution/alpaca_broker.py` (new) | `AccountState` dataclass; `BrokerError(RuntimeError)`; `AlpacaPaperBroker` (`requests`-based) implementing the `Broker` protocol + `account()`. Keys are passed to the constructor (not imported), so `execution` stays off `config`. |
| `algua/cli/paper_cmd.py` (modify) | `algua paper account` — builds the adapter from settings (raising a clear error if creds are missing) and emits equity/cash/buying-power JSON. The live auth/connectivity smoke + operational visibility. |

No new dependency — `requests` is already used by the data provider. The module lives under
`algua/execution/`, already contract-bound off `algua.cli`; no new import contract needed.

### `AlpacaPaperBroker` surface

```
AlpacaPaperBroker(api_key: str, api_secret: str, base_url: str)
  account() -> AccountState           # equity, cash, buying_power
  get_positions() -> pd.Series        # qty by symbol (Broker protocol)
  submit(intent: OrderIntent) -> str  # sizes from target_weight; returns Alpaca order id or "noop"
```

Auth headers on every request: `APCA-API-KEY-ID: <api_key>`, `APCA-API-SECRET-KEY: <api_secret>`.

---

## 3. REST mapping & `submit()` sizing

- **`account()`** → `GET {base}/v2/account` → parse `equity`, `cash`, `buying_power` (Alpaca returns
  them as strings) into floats → `AccountState`.
- **`get_positions()`** → `GET {base}/v2/positions` → `pd.Series({symbol: float(qty)})`; an empty
  list → an empty `float64` Series.
- **`submit(intent)`** (sizes internally, consistent with `SimBroker`):
  1. `equity = account().equity`.
  2. current market value: `GET {base}/v2/positions/{symbol}` → `float(market_value)`; a `404`
     means no open position → `0.0`.
  3. `target_$ = intent.target_weight * equity`; `delta_$ = target_$ - current_market_value`.
  4. if `abs(delta_$) < 1.0` (min notional) → return `"noop"` (place no order).
  5. `side = "buy" if delta_$ > 0 else "sell"`; `POST {base}/v2/orders` with
     `{"symbol": intent.symbol, "notional": str(round(abs(delta_$), 2)), "side": side,
     "type": "market", "time_in_force": "day"}` → return the response `id`.

A full exit (`target_weight = 0` → notional sell ≈ market value) is *approximate* against
price drift between the read and the fill; exact close (Alpaca's close-position endpoint) is a B2
auto-flatten concern. Acceptable for B1.

---

## 4. Error handling

- Any non-2xx Alpaca response → `raise BrokerError(f"alpaca {status}: {body}")`.
- Missing credentials: a CLI factory (`_alpaca_broker_from_settings`) checks
  `settings.alpaca_api_key` / `alpaca_api_secret` and raises
  `ValueError("Alpaca paper credentials not configured; set ALGUA_ALPACA_API_KEY and ALGUA_ALPACA_API_SECRET")`
  before constructing the adapter. `paper account` is decorated `@json_errors(ValueError, BrokerError)`
  so both surface as `{ok:false}` exit 1.
- The adapter performs HTTP via module-level `requests` calls so tests can monkeypatch them; it does
  not swallow errors silently.

---

## 5. Testing

- **Offline unit (mock `requests`)** — `account()` parses equity/cash/buying_power; `get_positions()`
  parses a positions list and an empty list; `submit()` buy-delta (POSTs the right notional + `buy`),
  sell-delta (`sell`), and below-threshold → `"noop"` (no POST); a non-2xx response → `BrokerError`.
- **CLI** — `paper account` with creds unset → `{ok:false}` exit 1; with the broker/requests mocked →
  emits `equity`/`cash`/`buying_power`.
- **Documented live smoke (NOT in CI)** — `ALGUA_ALPACA_API_KEY=… ALGUA_ALPACA_API_SECRET=…
  uv run algua paper account` against a real Alpaca paper account; optionally a one-off `submit` of a
  tiny notional order, verified in the Alpaca dashboard.
- **Gates** — `pytest · ruff · mypy · lint-imports`; the existing `execution`-off-`cli` contract
  already covers the new module.

---

## 6. Consequences

- The Alpaca paper broker now sits behind the same `Broker` protocol as `SimBroker`, so B2's
  wall-clock loop can target either. Sizing-from-weight lives in the broker (as in `SimBroker`),
  keeping the protocol surface "submit by intent".
- Credentials stay in settings/env (never in code), and the adapter is constructed with injected
  keys, so `algua/execution` keeps no dependency on `algua/config`.
- The async fill model is explicitly **not** handled here: `submit` returns an order id, and
  `get_positions` reflects whatever Alpaca has filled so far. Wiring that into a correct
  reconciling loop is B2.
