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
        resp = requests.get(
            f"{self.base_url}/v2/account", headers=self._headers(), timeout=_TIMEOUT
        )
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
