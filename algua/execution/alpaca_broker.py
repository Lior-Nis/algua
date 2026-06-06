from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from requests import RequestException

from algua.contracts.types import LiveAuthorization, OrderIntent
from algua.execution.sizing import size_order

_TIMEOUT = 30  # seconds: per-request connect+read timeout for every Alpaca HTTP call
# Default endpoints for the two Alpaca venues.
_PAPER_DEFAULT_URL = "https://paper-api.alpaca.markets"
_LIVE_DEFAULT_URL = "https://api.alpaca.markets"

# Retry policy for transient broker failures (#24). Reads are always safe to retry; submits are
# safe ONLY because they carry a deterministic client_order_id, so a retried POST that already
# landed is de-duplicated by Alpaca rather than double-filling.
_RETRYABLE_STATUS = (429, 500, 502, 503, 504)
_MAX_RETRIES = 3  # total attempts = 1 + retries on a retryable failure
_BACKOFF_BASE = 0.5  # seconds; sleep grows _BACKOFF_BASE * 2**attempt between attempts


def _coerce_status(value: Any) -> int:
    """Best-effort int from a 207 item's `status` field. A missing/non-int value is treated as a
    failure status (500) rather than raising — a malformed item must count as a failure, not crash
    the whole cancel with an uncaught ValueError (#22)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 500


def _multistatus_failures(results: list[Any]) -> list[Any]:
    """Per-item failures in an Alpaca 207 multi-status list. A non-dict item or a non-2xx
    `status` counts as a failure (a malformed item must not pass silently)."""
    return [
        r for r in results
        if not isinstance(r, dict) or _coerce_status(r.get("status", 500)) not in (200, 204)
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


class _AlpacaBroker:
    """requests-based wrapper of the Alpaca REST API. Implements the contracts Broker protocol
    (get_positions, submit) and sizes via the shared `size_order` rule so it sizes target weights
    identically to SimBroker. The tick loop drives it through snapshot()/submit_sized() to keep
    round-trips to 1 account GET + 1 positions GET + N POSTs with a fixed sizing denominator (#20).
    Async by nature: submit returns an order id; fills land later and show up in get_positions().

    Subclasses declare `_ALLOWED_HOSTS` to restrict which Alpaca endpoint may be used. The base
    class carries an empty set and therefore cannot be constructed against ANY host."""

    _ALLOWED_HOSTS: frozenset[str] = frozenset()

    def __init__(self, api_key: str, api_secret: str, base_url: str) -> None:
        host = urlparse(base_url).hostname
        if host not in self._ALLOWED_HOSTS:
            raise BrokerError(
                f"refusing Alpaca endpoint {base_url!r} (host {host!r}); "
                f"permitted hosts: {sorted(self._ALLOWED_HOSTS)}"
            )
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret}

    def _request(self, method: str, path: str, *, body: dict[str, Any] | None = None,
                 retryable_status: tuple[int, ...] = ()) -> requests.Response:
        """One Alpaca HTTP call with bounded exponential backoff (#24). Retries on a transport
        error (timeout/reset) and on the caller-supplied `retryable_status` set (429/5xx). Reads
        pass the full retryable set; submits do too, but ONLY because they carry a deterministic
        client_order_id so a retried POST that already landed is de-duplicated by Alpaca rather
        than double-filling. Non-retryable statuses are returned as-is for the caller to inspect."""
        url = f"{self.base_url}{path}"
        last_exc: RequestException | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                if method == "GET":
                    resp = requests.get(url, headers=self._headers(), timeout=_TIMEOUT)
                elif method == "POST":
                    resp = requests.post(url, headers=self._headers(), json=body, timeout=_TIMEOUT)
                else:  # DELETE
                    resp = requests.delete(url, headers=self._headers(), timeout=_TIMEOUT)
            except RequestException as exc:
                last_exc = exc
            else:
                if resp.status_code not in retryable_status or attempt == _MAX_RETRIES - 1:
                    return resp
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE * (2 ** attempt))
        # Exhausted retries on a transport error (a retryable-status response would have returned
        # on the final attempt above).
        raise BrokerError(
            f"alpaca {method} {path} failed after {_MAX_RETRIES} attempts: {last_exc}"
        )

    def _get(self, path: str) -> requests.Response:
        return self._request("GET", path, retryable_status=_RETRYABLE_STATUS)

    def _post(self, path: str, body: dict[str, Any]) -> requests.Response:
        return self._request("POST", path, body=body, retryable_status=_RETRYABLE_STATUS)

    def _delete(self, path: str) -> requests.Response:
        return self._request("DELETE", path, retryable_status=_RETRYABLE_STATUS)

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
        resp = self._delete("/v2/orders")
        results = self._read(resp, "/v2/orders", ok=(200, 207))
        if resp.status_code == 207:
            # A 207 MUST carry a per-order result list. A non-list body (e.g. an error object) is
            # malformed: raise rather than treat it as "all cancelled", which would let stale
            # orders survive and the next submit over-order (#22).
            if not isinstance(results, list):
                raise BrokerError(f"alpaca 207 on /v2/orders with non-list body: {results!r}")
            if _multistatus_failures(results):
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

    def close_all_positions(self) -> None:
        """Liquidate the ENTIRE account: DELETE /v2/positions?cancel_orders=true — Alpaca cancels
        all open orders then market-closes every position, returning a 207 multi-status; raise if
        any per-position close failed. Account-wide — used ONLY by the global halt (per-strategy
        flatten uses close_positions(universe)). Empty account -> empty list -> no-op."""
        results = self._read(
            self._delete("/v2/positions?cancel_orders=true"), "/v2/positions", ok=(200, 207)
        )
        if not isinstance(results, list):
            # A panic flatten must not call an unexpected (non-list) body a success and let the
            # operator believe the account is flat.
            raise BrokerError(f"alpaca /v2/positions: expected a list, got {results!r}")
        if _multistatus_failures(results):
            raise BrokerError(f"alpaca failed to close some positions: {results}")

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

    def submit_sized(self, intent: OrderIntent, snap: TickSnapshot,
                     client_order_id: str | None = None) -> str:
        """Size ONE intent against the tick snapshot (shared `size_order`) and POST it. The symbol
        MUST be in the snapshot's universe — an unknown symbol raises rather than silently sizing a
        full target-weight buy against a phantom flat position (#29). Returns the order id, or
        "noop" if the delta is below the minimum notional.

        `client_order_id`, when given, is sent as Alpaca's `client_order_id`: it makes the submit
        idempotent so a retried POST (after a transient 429/5xx/timeout) that already landed is
        de-duplicated by Alpaca rather than double-filling, and lets a re-run of the same tick
        reconcile against orders it already placed (#18, #24)."""
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
        body: dict[str, Any] = {"symbol": intent.symbol, "notional": notional,
                                "side": side, "type": "market", "time_in_force": "day"}
        if client_order_id is not None:
            body["client_order_id"] = client_order_id
        data = self._read(self._post("/v2/orders", body), "/v2/orders", ok=(200, 201))
        order_id = data.get("id") if isinstance(data, dict) else None
        if not order_id:
            raise BrokerError(f"alpaca /v2/orders: response missing 'id': {data}")
        return str(order_id)

    def submit(self, intent: OrderIntent, client_order_id: str | None = None) -> str:
        """Broker-protocol single-symbol submit: snapshot scoped to this one symbol, then size +
        POST. The tick loop instead snapshots the whole universe ONCE and calls submit_sized per
        symbol, so this self-snapshotting path is reserved for single-order callers."""
        return self.submit_sized(intent, self.snapshot([intent.symbol]), client_order_id)


class AlpacaPaperBroker(_AlpacaBroker):
    """The Alpaca PAPER venue. Hard-refuses any non-paper host (the platform invariant)."""

    _ALLOWED_HOSTS = frozenset({"paper-api.alpaca.markets"})

    def __init__(self, api_key: str, api_secret: str, base_url: str = _PAPER_DEFAULT_URL) -> None:
        super().__init__(api_key, api_secret, base_url)


class AlpacaLiveBroker(_AlpacaBroker):
    """The Alpaca LIVE (real-money) venue. Constructable ONLY with a verified LiveAuthorization
    (the construction tollbooth) AND against the live host — so a live broker cannot exist without
    a passed go-live gate. The authorization is kept for audit/the loop, never used for REST."""

    _ALLOWED_HOSTS = frozenset({"api.alpaca.markets"})

    def __init__(self, authorization: LiveAuthorization, api_key: str, api_secret: str,
                 base_url: str = _LIVE_DEFAULT_URL) -> None:
        if not isinstance(authorization, LiveAuthorization):
            raise BrokerError("AlpacaLiveBroker requires a verified LiveAuthorization")
        super().__init__(api_key, api_secret, base_url)
        self.authorization = authorization
