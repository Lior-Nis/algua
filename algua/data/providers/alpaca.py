from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

from algua.data.contracts import BarProvider, BarRequest, ProviderBars
from algua.data.providers.errors import ProviderError

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 0.5


def _alpaca_timeframe(timeframe: str) -> str:
    """Map algua's canonical timeframe to Alpaca's bars-API format (e.g. '1d' -> '1Day').

    Alpaca rejects '1d' with HTTP 400 'invalid timeframe'; its daily timeframe is '1Day'.
    Unknown values pass through (Alpaca will reject anything it doesn't recognise)."""
    return {"1d": "1Day", "1day": "1Day"}.get(timeframe.lower(), timeframe)


class AlpacaBarProvider(BarProvider):
    name = "alpaca"

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        base_url: str = "https://data.alpaca.markets/v2",
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")

    def get_bars(self, request: BarRequest) -> ProviderBars:
        raw_payload = self._fetch_bars(request, adjustment="raw")
        adjusted_payload = self._fetch_bars(request, adjustment="all")
        raw_frame = _normalize_alpaca(raw_payload)
        adjusted_frame = _normalize_alpaca(adjusted_payload)[["ts", "symbol", "close"]].rename(
            columns={"close": "adj_close"}
        )
        frame = raw_frame.merge(adjusted_frame, on=["ts", "symbol"], how="inner")
        if frame.empty:
            raise ProviderError("provider returned no overlapping raw/adjusted bars")
        return ProviderBars(
            frame=frame,
            source_metadata={
                "api": "alpaca",
                "base_url": self.base_url,
                "timeframe": request.timeframe,
                "adjustment": "raw+all",
            },
        )

    def _fetch_bars(self, request: BarRequest, *, adjustment: str) -> dict[str, Any]:
        """Fetch one adjustment view, retrying transient 429/5xx with backoff.

        All transport faults (HTTP errors, connection/timeout failures) are wrapped in
        ProviderError so the CLI's @json_errors renders them on stdout rather than
        letting a raw requests traceback escape the JSON contract.
        """
        for attempt in range(1, MAX_ATTEMPTS + 1):
            last_attempt = attempt == MAX_ATTEMPTS
            try:
                response = requests.get(
                    f"{self.base_url}/stocks/bars",
                    headers={
                        "APCA-API-KEY-ID": self.api_key,
                        "APCA-API-SECRET-KEY": self.api_secret,
                    },
                    params={
                        "symbols": ",".join(request.symbols),
                        "timeframe": _alpaca_timeframe(request.timeframe),
                        "start": request.start,
                        "end": request.end,
                        "adjustment": adjustment,
                    },
                    timeout=30,
                )
            except requests.RequestException as exc:
                if last_attempt:
                    raise ProviderError(
                        f"alpaca request failed after {MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc
                time.sleep(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
                continue

            status = getattr(response, "status_code", None)
            if status in RETRYABLE_STATUS and not last_attempt:
                time.sleep(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
                continue

            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise ProviderError(f"alpaca returned HTTP {status}: {exc}") from exc

            payload = response.json()
            if not isinstance(payload, dict):
                raise ProviderError("provider returned a non-object response")
            return payload

        raise AssertionError("unreachable: retry loop always returns or raises")


def _normalize_alpaca(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for symbol, bars in payload.get("bars", {}).items():
        for bar in bars:
            rows.append(
                {
                    "ts": bar.get("t"),
                    "symbol": symbol,
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v"),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ProviderError("provider returned no bars")
    return frame.sort_values(["symbol", "ts"]).reset_index(drop=True)
