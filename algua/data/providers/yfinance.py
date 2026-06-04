from __future__ import annotations

import pandas as pd

from algua.data.contracts import BarProvider, BarRequest, ProviderBars


class YFinanceBarProvider(BarProvider):
    name = "yfinance"

    def get_bars(self, request: BarRequest) -> ProviderBars:
        import yfinance as yf

        if request.adjustment not in {"none", "raw"}:
            raise ValueError(
                "yfinance bars require adjustment='none' so raw close and adj_close are both "
                "available for the frozen bar schema"
            )
        raw = yf.download(
            tickers=list(request.symbols),
            start=request.start,
            end=request.end,
            interval=request.timeframe,
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=False,
        )
        frame = _normalize_yfinance(raw, request.symbols)
        return ProviderBars(
            frame=frame,
            source_metadata={
                "library": "yfinance",
                "timeframe": request.timeframe,
                "adjustment": "none",
            },
        )


def _normalize_yfinance(raw: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("provider returned no bars")

    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        for symbol in symbols:
            if symbol not in raw.columns.get_level_values(0):
                continue
            part = raw[symbol].copy()
            part["symbol"] = symbol
            frames.append(part.reset_index())
    else:
        part = raw.copy()
        part["symbol"] = symbols[0]
        frames.append(part.reset_index())

    if not frames:
        raise ValueError("provider returned no bars for requested symbols")

    out = pd.concat(frames, ignore_index=True)
    out.columns = [_column_name(c) for c in out.columns]
    rename = {
        "date": "ts",
        "datetime": "ts",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "adj_close",
        "volume": "volume",
        "symbol": "symbol",
    }
    out = out.rename(columns={c: rename.get(c, c) for c in out.columns})
    out["ts"] = _to_utc(out["ts"])
    cols = [
        c
        for c in ("ts", "symbol", "open", "high", "low", "close", "adj_close", "volume")
        if c in out
    ]
    return out[cols].sort_values(["symbol", "ts"]).reset_index(drop=True)


def _to_utc(ts: pd.Series) -> pd.Series:
    """Normalize yfinance timestamps to tz-aware UTC for the bar schema.

    yfinance daily bars carry a naive `Date` index — we treat those naive timestamps as UTC
    session dates and localize them explicitly. Intraday bars carry a tz-aware (exchange-local)
    index, which we convert to the same instant in UTC. Normalizing here keeps the schema's
    tz-aware requirement intact instead of silently localizing at the validation boundary.
    """
    parsed = pd.to_datetime(ts, errors="raise")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize("UTC")
    return parsed.dt.tz_convert("UTC")


def _column_name(value: object) -> str:
    return str(value).strip().lower().replace("_", " ")
