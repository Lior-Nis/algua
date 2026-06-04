from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class BarRequest:
    """An ingestion request for raw vendor bars.

    `start` is inclusive. The canonical window convention is half-open `[start, end)` — matching
    the serving seam (`StoreBackedProvider`, see `docs/contracts/bar-schema.md`). Vendors differ at
    the transport layer: yfinance treats `end` as exclusive (matches the convention); Alpaca treats
    `end` as inclusive (so a raw Alpaca pull may include the `end` bar). Adapters do not currently
    re-clip to enforce half-open at ingestion; the look-ahead-safe boundary is enforced where it
    matters — on the serving read path.
    """

    symbols: tuple[str, ...]
    start: str
    end: str
    timeframe: str = "1d"
    adjustment: str = "none"


@dataclass(frozen=True)
class ProviderBars:
    frame: pd.DataFrame
    source_metadata: dict[str, str] = field(default_factory=dict)


class BarProvider(Protocol):
    """Ingestion seam: fetch raw vendor bars for a `BarRequest`.

    Distinct from the serving `DataProvider` (`algua.contracts.types`), which reads pinned
    snapshots for the backtest/live engines with a `get_bars(symbols, start, end, timeframe)`
    signature. The two are intentionally not substitutable; see `algua/data/providers/__init__.py`.
    """

    name: str

    def get_bars(self, request: BarRequest) -> ProviderBars: ...
