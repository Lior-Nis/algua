from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from algua.contracts.net import require_https_allowlisted_host
from algua.data.contracts import BarProvider, BarRequest, ProviderBars
from algua.data.providers.errors import ProviderError
from algua.data.timeframes import is_intraday
from algua.primitives.retry import RetriesExhausted, call_with_backoff

# The Alpaca market-data host. Its credentials (APCA-API-KEY-ID / APCA-API-SECRET-KEY) are the
# same account-scoped broker secrets used to place orders, so the data path enforces the SAME
# https + host wall the trading path does (issue #394) before any request attaches them.
_ALLOWED_HOSTS = frozenset({"data.alpaca.markets"})

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 0.5

#: Bars per page to ask Alpaca for. Its API caps a page at 10000 and DEFAULTS TO 1000, handing back
#: a `next_page_token` for the remainder. Asking for the maximum minimises round trips; it does not
#: remove the need to follow the token (see `_fetch_bars`).
PAGE_LIMIT = 10000
#: Hard bound on pages followed for a single request, so a server that kept returning a token could
#: never spin forever. At PAGE_LIMIT that is 10M bars -- orders of magnitude past any real window,
#: so hitting it means something is wrong and the request fails closed rather than returning a
#: quietly partial frame.
MAX_PAGES = 1000


#: algua's canonical timeframe tokens (`data/timeframes.KNOWN`) in Alpaca's bars-API spelling.
#: Alpaca rejects every one of algua's tokens with HTTP 400 'invalid timeframe', so an unmapped
#: entry is not a soft fallback -- it is a dead timeframe. Covering only '1d' meant the INTRADAY
#: tokens were silently unreachable through this provider even though the free IEX feed serves
#: hourly and minute bars back to 2016.
_ALPACA_TIMEFRAMES: dict[str, str] = {
    "1d": "1Day", "1day": "1Day",
    "1m": "1Min", "5m": "5Min", "30m": "30Min", "1h": "1Hour",
}


def _alpaca_timeframe(timeframe: str) -> str:
    """Map algua's canonical timeframe to Alpaca's bars-API format (e.g. '1d' -> '1Day').

    Unknown values pass through, and Alpaca rejects anything it does not recognise -- which is the
    intended behaviour for a token algua itself would have refused at `validate_timeframe`."""
    return _ALPACA_TIMEFRAMES.get(timeframe.lower(), timeframe)


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
        """Fetch one adjustment view IN FULL, following Alpaca's pagination.

        Alpaca caps a bars page at `PAGE_LIMIT` and returns `next_page_token` for the remainder.
        Ignoring that token does not error -- it returns a short frame that looks complete, which
        is the worst possible failure for a data lane: a ten-year daily request came back as
        2016-01-04..2019-12-20 and nothing downstream could tell. Intraday made it acute (a page
        is ~2.5 days of minute bars), which is how it surfaced.

        All transport faults (HTTP errors, connection/timeout failures) are wrapped in
        ProviderError so the CLI's @json_errors renders them on stdout rather than
        letting a raw requests traceback escape the JSON contract.
        """
        merged: dict[str, list[Any]] = {}
        page_token: str | None = None
        for _ in range(MAX_PAGES):
            payload = self._fetch_page(request, adjustment=adjustment, page_token=page_token)
            for symbol, bars in (payload.get("bars") or {}).items():
                merged.setdefault(symbol, []).extend(bars or [])
            token = payload.get("next_page_token")
            if not token:
                return {"bars": merged}
            page_token = str(token)
        raise ProviderError(
            f"alpaca paginated past {MAX_PAGES} pages for {request.timeframe} "
            f"{request.start}..{request.end}; refusing a possibly-partial snapshot"
        )

    def _fetch_page(
        self, request: BarRequest, *, adjustment: str, page_token: str | None,
    ) -> dict[str, Any]:
        """One page of one adjustment view, retrying transient 429/5xx with backoff."""

        def _send() -> requests.Response:
            params: dict[str, Any] = {
                "symbols": ",".join(request.symbols),
                "timeframe": _alpaca_timeframe(request.timeframe),
                "start": request.start,
                "end": request.end,
                "adjustment": adjustment,
                "limit": PAGE_LIMIT,
            }
            if page_token:
                params["page_token"] = page_token
            return requests.get(
                f"{self.base_url}/stocks/bars",
                headers={
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.api_secret,
                },
                params=params,
                timeout=30,
                # Never chase a redirect: requests re-sends the APCA credential headers on a
                # cross-host 3xx, which would leak them to the redirect target (#394).
                allow_redirects=False,
            )

        try:
            response = call_with_backoff(
                _send, attempts=MAX_ATTEMPTS, backoff_base=BACKOFF_BASE_SECONDS,
                retryable_exceptions=(requests.RequestException,),
                retry_result=lambda r: getattr(r, "status_code", None) in RETRYABLE_STATUS,
            )
        except RetriesExhausted as exc:
            raise ProviderError(
                f"alpaca request failed after {MAX_ATTEMPTS} attempts: {exc.last_exception}"
            ) from exc

        status = getattr(response, "status_code", None)
        if status is not None and 300 <= status <= 399:
            raise ProviderError(
                f"alpaca returned an unexpected redirect (HTTP {status}); refusing to "
                "forward credentials to the redirect target"
            )

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
