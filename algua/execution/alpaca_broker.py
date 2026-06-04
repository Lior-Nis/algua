from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pandas as pd
import requests
from requests import RequestException

from algua.contracts.types import OrderIntent
from algua.execution.sizing import size_order

_TIMEOUT = 30
_DEFAULT_BASE_URL = "https://paper-api.alpaca.markets"


def _multistatus_failures(results: list[Any]) -> list[Any]:
    """Per-item failures in an Alpaca 207 multi-status list. A non-dict item or a non-2xx
    `status` counts as a failure (a malformed item must not pass silently)."""
    return [
        r for r in results
        if not isinstance(r, dict) or int(r.get("status", 500)) not in (200, 204)
    ]


class BrokerError(RuntimeError):
    """Any failure talking to the Alpaca trading API — network error, non-2xx status,
    or a malformed/unexpected response. Callers (the CLI, the future loop) catch this so a
    broker hiccup never escapes as a raw traceback."""


@dataclass(frozen=True)
class AccountState:
    equity: float
    cash: float
    buying_power: float


@dataclass(frozen=True)
class TickSnapshot:
    """Equity + per-symbol market value AND qty captured ONCE at tick start (1 account GET + 1
    positions GET). The equity is the fixed sizing denominator for every symbol in the tick, so it
    can't drift as earlier orders in the same tick fill (#20). `market_values`/`qtys` are keyed on
    the universe the snapshot was scoped to (0.0 when flat)."""

    equity: float
    market_values: dict[str, float]  # symbol -> current position market value (0.0 => flat)
    qtys: dict[str, float]  # symbol -> current position shares (0.0 => flat)


class AlpacaPaperBroker:
    """requests-based wrapper of the Alpaca paper-trading REST. Implements the contracts Broker
    protocol (get_positions, submit) and sizes via the shared `size_order` rule so it sizes target
    weights identically to SimBroker. The tick loop drives it through snapshot()/submit_sized() to
    keep round-trips to 1 account GET + 1 positions GET + N POSTs with a fixed sizing denominator
    (#20). Async by nature: submit returns an order id; fills land later and show up in
    get_positions()."""

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

    def _positions_raw(self) -> list[Any]:
        return self._read(self._get("/v2/positions"), "/v2/positions")

    def get_positions(self) -> pd.Series:
        rows = self._positions_raw()
        return pd.Series(
            {row["symbol"]: self._num(row, "qty", "/v2/positions") for row in rows},
            dtype="float64",
        )

    def cancel_open_orders(self) -> None:
        """Cancel all open orders (DELETE /v2/orders). Alpaca returns 207 multi-status with a
        per-order result list; raise if ANY individual cancel failed, otherwise a stale order
        could survive and the next submit would over-order."""
        results = self._read(self._delete("/v2/orders"), "/v2/orders", ok=(200, 207))
        if isinstance(results, list) and _multistatus_failures(results):
            raise BrokerError(f"alpaca failed to cancel some orders: {results}")

    def close_positions(self, symbols: list[str]) -> None:
        """Liquidate the given symbols' positions — one DELETE /v2/positions/{symbol} each, so a
        flatten is SCOPED to a single strategy's universe rather than nuking the whole account
        (DELETE /v2/positions is reserved for a future global halt). 404 (no open position) is a
        no-op; any other non-2xx raises BrokerError. Submits liquidating market orders (async
        fills). Idempotent: re-running on a flat book is a series of 404 no-ops."""
        for symbol in symbols:
            path = f"/v2/positions/{symbol}"
            resp = self._delete(path)
            if resp.status_code == 404:
                continue  # no open position for this symbol -> nothing to close
            self._read(resp, path, ok=(200,))

    def snapshot(self, universe: list[str]) -> TickSnapshot:
        """Capture equity + per-symbol market value and qty ONCE (1 account GET + 1 positions GET).
        The snapshot keys are the union of `universe` and every currently-held symbol: universe
        names absent from the book are flat (0.0), and held names outside the universe are included
        so a dropped position can still be exited. Sizing a symbol that is neither in the universe
        nor held is rejected by submit_sized, so a typo can't be misread as flat (#29)."""
        equity = self.account().equity
        rows = self._positions_raw()
        open_values = {r["symbol"]: self._num(r, "market_value", "/v2/positions") for r in rows}
        open_qtys = {r["symbol"]: self._num(r, "qty", "/v2/positions") for r in rows}
        symbols = set(universe) | set(open_qtys)
        return TickSnapshot(
            equity=equity,
            market_values={sym: open_values.get(sym, 0.0) for sym in symbols},
            qtys={sym: open_qtys.get(sym, 0.0) for sym in symbols},
        )

    def submit_sized(self, intent: OrderIntent, snap: TickSnapshot) -> str:
        """Size ONE intent against the tick snapshot (shared `size_order`) and POST it. The symbol
        MUST be in the snapshot's universe — an unknown symbol raises rather than silently sizing a
        full target-weight buy against a phantom flat position (#29). Returns the order id, or
        "noop" if the delta is below the minimum notional."""
        if intent.symbol not in snap.market_values:
            raise BrokerError(
                f"alpaca submit: {intent.symbol!r} is not in the strategy universe "
                f"{sorted(snap.market_values)} — refusing to size an unknown symbol"
            )
        sized = size_order(symbol=intent.symbol, target_weight=intent.target_weight,
                           equity=snap.equity,
                           current_market_value=snap.market_values[intent.symbol])
        if sized.is_noop:
            return "noop"
        side = "buy" if sized.delta_notional > 0 else "sell"
        notional = format(Decimal(str(abs(sized.delta_notional))).quantize(Decimal("0.01")), "f")
        data = self._read(
            self._post("/v2/orders", {"symbol": intent.symbol, "notional": notional,
                                      "side": side, "type": "market", "time_in_force": "day"}),
            "/v2/orders", ok=(200, 201),
        )
        order_id = data.get("id") if isinstance(data, dict) else None
        if not order_id:
            raise BrokerError(f"alpaca /v2/orders: response missing 'id': {data}")
        return str(order_id)

    def submit(self, intent: OrderIntent) -> str:
        """Broker-protocol single-symbol submit: snapshot scoped to this one symbol, then size +
        POST. The tick loop instead snapshots the whole universe ONCE and calls submit_sized per
        symbol, so this self-snapshotting path is reserved for single-order callers."""
        return self.submit_sized(intent, self.snapshot([intent.symbol]))
