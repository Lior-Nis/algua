from __future__ import annotations

import pandas as pd

from algua.data.contracts import BarProvider, BarRequest, ProviderBars
from algua.data.providers.errors import ProviderError

REQUIRED_COLUMNS = ("ts", "symbol", "open", "high", "low", "close", "adj_close", "volume")


class YFinanceBarProvider(BarProvider):
    name = "yfinance"

    def get_bars(self, request: BarRequest) -> ProviderBars:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ProviderError(
                "yfinance is not installed; install it to fetch bars from this provider"
            ) from exc

        if request.adjustment not in {"none", "raw"}:
            raise ValueError(
                "yfinance bars require adjustment='none' so raw close and adj_close are both "
                "available for the frozen bar schema"
            )
        try:
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
        except Exception as exc:  # noqa: BLE001 - any library/network fault must stay in the JSON contract
            raise ProviderError(f"yfinance download failed: {exc}") from exc
        frame = _normalize_yfinance(raw, request.symbols)
        source_metadata = {
            "library": "yfinance",
            "timeframe": request.timeframe,
            "adjustment": "none",
        }
        return ProviderBars(frame=frame, source_metadata=source_metadata)


def _normalize_yfinance(raw: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
    """Normalize a yfinance frame to the bar schema.

    Raises ``ProviderError`` if any requested symbol is absent from the response so a
    partial frame can never be persisted under a manifest claiming full coverage.
    """
    if raw.empty:
        raise ProviderError("provider returned no bars")

    frames = []
    present = []
    if isinstance(raw.columns, pd.MultiIndex):
        available = set(raw.columns.get_level_values(0))
        for symbol in symbols:
            if symbol not in available:
                continue
            part = raw[symbol].copy()
            part["symbol"] = symbol
            frames.append(part.reset_index())
            present.append(symbol)
    else:
        part = raw.copy()
        part["symbol"] = symbols[0]
        frames.append(part.reset_index())
        present.append(symbols[0])

    if not frames:
        raise ProviderError(
            f"provider returned no bars for any requested symbol: {', '.join(symbols)}"
        )

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

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing_cols:
        raise ProviderError(
            "yfinance returned bars missing required columns: "
            + ", ".join(missing_cols)
            + f" (got: {', '.join(out.columns)})"
        )

    missing_symbols = sorted(set(symbols) - set(present))
    if missing_symbols:
        raise ProviderError(
            "yfinance returned no bars for requested symbols: " + ", ".join(missing_symbols)
        )

    out = out[list(REQUIRED_COLUMNS)].sort_values(["symbol", "ts"]).reset_index(drop=True)
    return out


def _column_name(value: object) -> str:
    return str(value).strip().lower().replace("_", " ")
