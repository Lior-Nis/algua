from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from algua.data.contracts import BarProvider, BarRequest, ProviderBars


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
            raise ValueError("provider returned no overlapping raw/adjusted bars")
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
        response = requests.get(
            f"{self.base_url}/stocks/bars",
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
            },
            params={
                "symbols": ",".join(request.symbols),
                "timeframe": request.timeframe,
                "start": request.start,
                "end": request.end,
                "adjustment": adjustment,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("provider returned a non-object response")
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
        raise ValueError("provider returned no bars")
    return frame.sort_values(["symbol", "ts"]).reset_index(drop=True)
