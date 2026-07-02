from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

from algua.contracts.net import require_https_allowlisted_host
from algua.data.contracts import BarProvider, BarRequest, ProviderBars
from algua.data.providers.errors import ProviderError
from algua.data.timeframes import is_intraday

# The Alpaca market-data host. Its credentials (APCA-API-KEY-ID / APCA-API-SECRET-KEY) are the
# same account-scoped broker secrets used to place orders, so the data path enforces the SAME
# https + host wall the trading path does (issue #394) before any request attaches them.
_ALLOWED_HOSTS = frozenset({"data.alpaca.markets"})

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
        # Guard BEFORE storing the URL: credentials must never dial a plaintext or non-Alpaca
        # host (#394). Re-raised as ProviderError to keep the CLI's JSON error contract.
        try:
            require_https_allowlisted_host(base_url, _ALLOWED_HOSTS)
        except ValueError as exc:
            raise ProviderError(str(exc)) from exc
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

        # Match raw vs adjusted on the ORIGINAL provider timestamps, BEFORE any daily flooring, so a
        # genuine raw/adjusted anchor drift (different `t` for the same session) still trips this
        # integrity check rather than being masked once both are floored to UTC midnight (#262).
        raw_keys = set(map(tuple, raw_frame[["ts", "symbol"]].itertuples(index=False)))
        adjusted_keys = set(map(tuple, adjusted_frame[["ts", "symbol"]].itertuples(index=False)))
        if raw_keys != adjusted_keys:
            unmatched = sorted(
                f"{symbol}@{ts}" for ts, symbol in raw_keys.symmetric_difference(adjusted_keys)
            )
            raise ProviderError(
                "raw and adjusted bar key sets differ; refusing partial snapshot. "
                "unmatched (ts, symbol): " + ", ".join(unmatched)
            )

        frame = raw_frame.merge(adjusted_frame, on=["ts", "symbol"], how="inner")
        if frame.empty:
            raise ProviderError("provider returned no overlapping raw/adjusted bars")
        frame = _canonicalize_daily_ts(frame, request.timeframe)
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
                    # Never chase a redirect: requests re-sends the APCA credential headers on a
                    # cross-host 3xx, which would leak them to the redirect target (#394).
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                if last_attempt:
                    raise ProviderError(
                        f"alpaca request failed after {MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc
                time.sleep(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
                continue

            status = getattr(response, "status_code", None)
            if status is not None and 300 <= status <= 399:
                raise ProviderError(
                    f"alpaca returned an unexpected redirect (HTTP {status}); refusing to "
                    "forward credentials to the redirect target"
                )
            if status in RETRYABLE_STATUS and not last_attempt:
                time.sleep(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
                continue

            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise ProviderError(f"alpaca returned HTTP {status}: {exc}") from exc

            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderError(
                    f"alpaca returned a malformed JSON body (HTTP {status}): {exc}"
                ) from exc
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


def _canonicalize_daily_ts(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Floor a daily ('1d') Alpaca frame's `ts` to the canonical UTC-midnight session date.

    Alpaca stamps a daily bar at the session-start UTC instant (e.g. …T05:00:00Z), not UTC
    midnight; the frozen bar-schema pins daily timestamps to the session date at UTC midnight and
    the ingest rail now fails closed on a non-midnight 1d bar (#262). yfinance already lands on
    midnight. Intraday frames are returned untouched (their bars are not clock-aligned by contract).

    Flooring runs AFTER the raw/adjusted key match + merge, so it can only collapse rows within one
    already-reconciled frame; we still reject any resulting duplicate `(ts, symbol)` defensively."""
    if is_intraday(timeframe):
        return frame
    floored = frame.copy()
    floored["ts"] = pd.to_datetime(floored["ts"], errors="raise", utc=True).dt.normalize()
    if floored[["ts", "symbol"]].duplicated().any():
        dups = floored.loc[floored[["ts", "symbol"]].duplicated(keep=False), ["ts", "symbol"]]
        offenders = sorted({f"{r.symbol}@{r.ts.date()}" for r in dups.itertuples(index=False)})
        raise ProviderError(
            "daily bars collapse to duplicate (UTC-midnight date, symbol) after canonicalization; "
            "refusing ambiguous snapshot: " + ", ".join(offenders)
        )
    return floored.sort_values(["symbol", "ts"]).reset_index(drop=True)
