# Alpaca Paper Broker Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `requests`-based `AlpacaPaperBroker` behind the `Broker` protocol (account / get_positions / submit-by-target-weight as notional market orders) plus a `paper account` connectivity smoke.

**Architecture:** Mirrors the existing `requests`-based Alpaca *data* provider (no SDK, no new dep). The broker sizes `target_weight → notional delta` internally (consistent with `SimBroker`) and POSTs notional market orders to the Alpaca **paper-trading** REST. Keys are injected by the CLI (broker stays off `config`). No trading loop runs in this slice — async fills, auto-flatten, and the global switch are B2.

**Tech Stack:** Python 3.12, `requests`, pandas, pydantic-settings, Typer, pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-03-alpaca-paper-broker-adapter-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `algua/config/settings.py` (modify) | `alpaca_paper_url` setting (trading API base). |
| `algua/execution/alpaca_broker.py` (new) | `AccountState`, `BrokerError`, `AlpacaPaperBroker`. |
| `algua/cli/paper_cmd.py` (modify) | `_alpaca_broker_from_settings` + `paper account` command. |

---

### Task 1: `alpaca_paper_url` setting

**Files:** Modify `algua/config/settings.py`; Test `tests/test_config.py`.

- [ ] **Step 1: Add the failing test** — append to `tests/test_config.py`:

```python
def test_alpaca_paper_url_default():
    from algua.config.settings import get_settings

    assert get_settings().alpaca_paper_url == "https://paper-api.alpaca.markets"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_config.py -q` → FAIL.

- [ ] **Step 3: Implement** — in `algua/config/settings.py`, add after the `alpaca_data_url` line:

```python
    alpaca_paper_url: str = "https://paper-api.alpaca.markets"
```

- [ ] **Step 4: Run** `uv run pytest tests/test_config.py -q` → PASS.

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_config.py -q`

```bash
git add algua/config/settings.py tests/test_config.py
git commit -m "feat(broker): alpaca_paper_url setting (trading API base)"
```

---

### Task 2: AlpacaPaperBroker

**Files:** Create `algua/execution/alpaca_broker.py`; Test `tests/test_alpaca_broker.py`.

- [ ] **Step 1: Add the failing test** — create `tests/test_alpaca_broker.py`:

```python
from datetime import UTC, datetime

import pytest

from algua.contracts.types import OrderIntent, Side
from algua.execution import alpaca_broker as ab
from algua.execution.alpaca_broker import AlpacaPaperBroker, BrokerError

T0 = datetime(2023, 1, 2, tzinfo=UTC)


class _FakeResp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class _FakeRequests:
    """Routes GET by URL suffix; records POST bodies."""

    def __init__(self, routes, post_resp=None):
        self.routes = routes
        self.post_resp = post_resp
        self.posted = []

    def get(self, url, headers=None, timeout=None):
        for suffix, resp in self.routes.items():
            if url.endswith(suffix):
                return resp
        return _FakeResp(404, text="not found")

    def post(self, url, headers=None, json=None, timeout=None):
        self.posted.append(json)
        return self.post_resp


def _broker():
    return AlpacaPaperBroker(api_key="k", api_secret="s")


def test_account_parses_floats(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "50000",
                                        "buying_power": "150000"})}))
    acct = _broker().account()
    assert (acct.equity, acct.cash, acct.buying_power) == (100000.0, 50000.0, 150000.0)


def test_get_positions_parses_and_empty(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/positions": _FakeResp(200, [{"symbol": "AAA", "qty": "10"},
                                          {"symbol": "BBB", "qty": "5"}])}))
    pos = _broker().get_positions()
    assert pos["AAA"] == 10.0 and pos["BBB"] == 5.0

    monkeypatch.setattr(ab, "requests", _FakeRequests({"/v2/positions": _FakeResp(200, [])}))
    assert _broker().get_positions().empty


def test_submit_buy_delta_posts_notional(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "0", "buying_power": "0"}),
         "/v2/positions/AAA": _FakeResp(404, text="no position")},
        post_resp=_FakeResp(201, {"id": "order-1"}),
    )
    monkeypatch.setattr(ab, "requests", fake)
    oid = _broker().submit(OrderIntent("AAA", Side.BUY, 0.5, T0))
    assert oid == "order-1"
    assert fake.posted[0] == {"symbol": "AAA", "notional": "50000.0", "side": "buy",
                              "type": "market", "time_in_force": "day"}


def test_submit_sell_delta(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "0", "buying_power": "0"}),
         "/v2/positions/AAA": _FakeResp(200, {"market_value": "60000"})},
        post_resp=_FakeResp(201, {"id": "order-2"}),
    )
    monkeypatch.setattr(ab, "requests", fake)
    _broker().submit(OrderIntent("AAA", Side.SELL, 0.5, T0))
    assert fake.posted[0]["side"] == "sell" and fake.posted[0]["notional"] == "10000.0"


def test_submit_noop_below_threshold(monkeypatch):
    fake = _FakeRequests(
        {"/v2/account": _FakeResp(200, {"equity": "100000", "cash": "0", "buying_power": "0"}),
         "/v2/positions/AAA": _FakeResp(200, {"market_value": "50000"})},
        post_resp=_FakeResp(201, {"id": "unused"}),
    )
    monkeypatch.setattr(ab, "requests", fake)
    assert _broker().submit(OrderIntent("AAA", Side.BUY, 0.5, T0)) == "noop"
    assert fake.posted == []


def test_non_2xx_raises_broker_error(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequests(
        {"/v2/account": _FakeResp(403, text="forbidden")}))
    with pytest.raises(BrokerError):
        _broker().account()
```

- [ ] **Step 2: Run** `uv run pytest tests/test_alpaca_broker.py -q` → FAIL.

- [ ] **Step 3: Implement** — create `algua/execution/alpaca_broker.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

from algua.contracts.types import OrderIntent

_TIMEOUT = 30
_MIN_NOTIONAL = 1.0
_DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"


class BrokerError(RuntimeError):
    """A non-2xx response (or other failure) from the Alpaca trading API."""


@dataclass(frozen=True)
class AccountState:
    equity: float
    cash: float
    buying_power: float


class AlpacaPaperBroker:
    """requests-based wrapper of the Alpaca paper-trading REST. Implements the Broker protocol
    (get_positions, submit) + account(). Sizes target_weight -> notional market orders internally,
    consistent with SimBroker. Async by nature: submit() returns an order id; fills land later and
    show up in get_positions()."""

    def __init__(self, api_key: str, api_secret: str, base_url: str = _DEFAULT_BASE_URL) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret}

    def account(self) -> AccountState:
        resp = requests.get(f"{self.base_url}/v2/account", headers=self._headers(), timeout=_TIMEOUT)
        if resp.status_code != 200:
            raise BrokerError(f"alpaca {resp.status_code}: {resp.text}")
        data = resp.json()
        return AccountState(
            equity=float(data["equity"]),
            cash=float(data["cash"]),
            buying_power=float(data["buying_power"]),
        )

    def get_positions(self) -> pd.Series:
        resp = requests.get(f"{self.base_url}/v2/positions", headers=self._headers(),
                            timeout=_TIMEOUT)
        if resp.status_code != 200:
            raise BrokerError(f"alpaca {resp.status_code}: {resp.text}")
        return pd.Series({row["symbol"]: float(row["qty"]) for row in resp.json()}, dtype="float64")

    def _market_value(self, symbol: str) -> float:
        resp = requests.get(f"{self.base_url}/v2/positions/{symbol}", headers=self._headers(),
                            timeout=_TIMEOUT)
        if resp.status_code == 404:
            return 0.0  # no open position
        if resp.status_code != 200:
            raise BrokerError(f"alpaca {resp.status_code}: {resp.text}")
        return float(resp.json()["market_value"])

    def submit(self, intent: OrderIntent) -> str:
        equity = self.account().equity
        delta = intent.target_weight * equity - self._market_value(intent.symbol)
        if abs(delta) < _MIN_NOTIONAL:
            return "noop"
        side = "buy" if delta > 0 else "sell"
        resp = requests.post(
            f"{self.base_url}/v2/orders",
            headers=self._headers(),
            json={"symbol": intent.symbol, "notional": str(round(abs(delta), 2)),
                  "side": side, "type": "market", "time_in_force": "day"},
            timeout=_TIMEOUT,
        )
        if resp.status_code not in (200, 201):
            raise BrokerError(f"alpaca {resp.status_code}: {resp.text}")
        return str(resp.json()["id"])
```

- [ ] **Step 4: Run** `uv run pytest tests/test_alpaca_broker.py -q` → PASS (6 passed).

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_alpaca_broker.py -q`

```bash
git add algua/execution/alpaca_broker.py tests/test_alpaca_broker.py
git commit -m "feat(broker): AlpacaPaperBroker (account/positions/submit notional orders)"
```

---

### Task 3: `paper account` CLI + full gate

**Files:** Modify `algua/cli/paper_cmd.py`; Test `tests/test_cli_paper.py`.

- [ ] **Step 1: Add failing tests** — append to `tests/test_cli_paper.py`:

```python
from algua.execution.alpaca_broker import AccountState


def test_paper_account_missing_creds_errors(monkeypatch):
    monkeypatch.delenv("ALGUA_ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALGUA_ALPACA_API_SECRET", raising=False)
    result = runner.invoke(app, ["paper", "account"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_paper_account_emits_balances(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    monkeypatch.setattr(
        "algua.cli.paper_cmd.AlpacaPaperBroker.account",
        lambda self: AccountState(equity=100000.0, cash=50000.0, buying_power=150000.0),
    )
    result = runner.invoke(app, ["paper", "account"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["equity"] == 100000.0 and payload["cash"] == 50000.0
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_paper.py -q` → FAIL.

- [ ] **Step 3: Implement** — in `algua/cli/paper_cmd.py`:

(a) Add the import (after `from algua.config.settings import get_settings`):

```python
from algua.execution.alpaca_broker import AlpacaPaperBroker, BrokerError
```

(b) Add a factory helper near the top (after the `app.add_typer(...)` line):

```python
def _alpaca_broker_from_settings() -> AlpacaPaperBroker:
    s = get_settings()
    if not s.alpaca_api_key or not s.alpaca_api_secret:
        raise ValueError(
            "Alpaca paper credentials not configured; set ALGUA_ALPACA_API_KEY "
            "and ALGUA_ALPACA_API_SECRET"
        )
    return AlpacaPaperBroker(api_key=s.alpaca_api_key, api_secret=s.alpaca_api_secret,
                             base_url=s.alpaca_paper_url)
```

(c) Add the command at the end of the file:

```python
@paper_app.command("account")
@json_errors(ValueError, BrokerError)
def account() -> None:
    """Show the Alpaca paper account (equity/cash/buying-power) — a connectivity smoke."""
    broker = _alpaca_broker_from_settings()
    acct = broker.account()
    emit({"equity": acct.equity, "cash": acct.cash, "buying_power": acct.buying_power})
```

- [ ] **Step 4: Run** `uv run pytest tests/test_cli_paper.py -q` → PASS.

- [ ] **Step 5: Verify command registered** — `uv run algua paper account --help` shows the command; `ALGUA_ALPACA_API_KEY= uv run algua paper account` (creds unset) emits `{"ok": false, ...}` exit 1.

- [ ] **Step 6: Full gate + commit** — `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all pass; `lint-imports` stays `10 kept, 0 broken` — the new module is under `algua.execution`, already off `cli`).

```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(broker): `algua paper account` Alpaca connectivity smoke"
```

---

## Self-review notes

- **Spec coverage:** `alpaca_paper_url` (§2 → Task 1); `AccountState`/`BrokerError`/`AlpacaPaperBroker` with account/get_positions/submit + REST mapping + notional sizing + 404→0 + non-2xx→BrokerError (§2,§3,§4 → Task 2); `paper account` + missing-creds factory (§2,§4 → Task 3); offline mock tests + CLI tests (§5 → Tasks 2,3); documented live smoke is in the spec (§5), not code. No new import contract needed (§2).
- **Type consistency:** `AlpacaPaperBroker(api_key, api_secret, base_url)`, `account() -> AccountState(equity, cash, buying_power)`, `get_positions() -> pd.Series`, `submit(intent) -> str` are identical across Tasks 2 and 3; the CLI mocks `AlpacaPaperBroker.account` (Task 3) matching Task 2's signature.
- **No placeholders:** every code step is complete.
- **Note on `submit` notional formatting:** `str(round(abs(delta), 2))` yields e.g. `"50000.0"` / `"10000.0"`; the tests assert these exact strings, so keep the `round(...,2)` + `str(...)` as written.
