from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pandas as pd
import requests
from requests import RequestException

from algua.contracts.types import OrderIntent

_TIMEOUT = 30
_MIN_NOTIONAL = 1.0
_DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"


class BrokerError(RuntimeError):
    """Any failure talking to the Alpaca trading API — network error, non-2xx status,
    or a malformed/unexpected response. Callers (the CLI, the future loop) catch this so a
    broker hiccup never escapes as a raw traceback."""


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

    def _get(self, path: str) -> requests.Response:
        try:
            return requests.get(f"{self.base_url}{path}", headers=self._headers(), timeout=_TIMEOUT)
        except RequestException as exc:
            raise BrokerError(f"alpaca GET {path} failed: {exc}") from exc

    def _post(self, path: str, body: dict[str, Any]) -> requests.Response:
        try:
            return requests.post(f"{self.base_url}{path}", headers=self._headers(),
                                 json=body, timeout=_TIMEOUT)
        except RequestException as exc:
            raise BrokerError(f"alpaca POST {path} failed: {exc}") from exc

    def _delete(self, path: str) -> requests.Response:
        try:
            return requests.delete(f"{self.base_url}{path}", headers=self._headers(),
                                   timeout=_TIMEOUT)
        except RequestException as exc:
            raise BrokerError(f"alpaca DELETE {path} failed: {exc}") from exc

    @staticmethod
    def _read(resp: requests.Response, path: str, ok: tuple[int, ...] = (200,)) -> Any:
        if resp.status_code not in ok:
            raise BrokerError(f"alpaca {resp.status_code} on {path}: {resp.text}")
        try:
            return resp.json()
        except ValueError as exc:
            raise BrokerError(f"alpaca malformed JSON on {path}: {exc}") from exc

    @staticmethod
    def _num(data: dict[str, Any], key: str, path: str) -> float:
        try:
            return float(data[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise BrokerError(f"alpaca {path}: bad or missing field {key!r}") from exc

    def account(self) -> AccountState:
        data = self._read(self._get("/v2/account"), "/v2/account")
        return AccountState(
            equity=self._num(data, "equity", "/v2/account"),
            cash=self._num(data, "cash", "/v2/account"),
            buying_power=self._num(data, "buying_power", "/v2/account"),
        )

    def get_positions(self) -> pd.Series:
        data = self._read(self._get("/v2/positions"), "/v2/positions")
        return pd.Series(
            {row["symbol"]: self._num(row, "qty", "/v2/positions") for row in data},
            dtype="float64",
        )

    def cancel_open_orders(self) -> None:
        """Cancel all open orders (DELETE /v2/orders). Alpaca returns 207 multi-status with
        per-order results; we only require an overall-success status."""
        resp = self._delete("/v2/orders")
        if resp.status_code not in (200, 207):
            raise BrokerError(f"alpaca {resp.status_code} on /v2/orders: {resp.text}")

    def _market_value(self, symbol: str) -> float:
        path = f"/v2/positions/{symbol}"
        resp = self._get(path)
        if resp.status_code == 404:
            return 0.0  # Alpaca returns 404 when there is no open position
        return self._num(self._read(resp, path), "market_value", path)

    def submit(self, intent: OrderIntent) -> str:
        equity = self.account().equity
        delta = intent.target_weight * equity - self._market_value(intent.symbol)
        if abs(delta) < _MIN_NOTIONAL:
            return "noop"
        side = "buy" if delta > 0 else "sell"
        notional = format(Decimal(str(abs(delta))).quantize(Decimal("0.01")), "f")
        data = self._read(
            self._post("/v2/orders", {"symbol": intent.symbol, "notional": notional,
                                      "side": side, "type": "market", "time_in_force": "day"}),
            "/v2/orders", ok=(200, 201),
        )
        order_id = data.get("id") if isinstance(data, dict) else None
        if not order_id:
            raise BrokerError(f"alpaca /v2/orders: response missing 'id': {data}")
        return str(order_id)
