from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass(frozen=True)
class ImportRequest:
    """A request to import local vendor bar files (the `BarImporter` seam).

    Distinct from `BarRequest` (network fetch by symbol/date): the source here is local files the
    operator already downloaded. `raw_dir` supplies unadjusted OHLC; `adjusted_dir` supplies the
    vendor-adjusted close used as `adj_close`. `adjustment` is the operator-declared flavor of that
    adjusted file (we never infer it). `symbols`, if set, restricts the import to that subset.
    """

    raw_dir: Path
    adjusted_dir: Path
    timeframe: str = "1d"
    as_of: str | None = None
    adjustment: str = "split_div"
    symbols: tuple[str, ...] | None = None


class BarImporter(Protocol):
    """Ingestion seam for local vendor files: yield one normalized `ProviderBars` per symbol.

    Yielding per symbol (rather than returning one giant frame) is what bounds RAM for a multi-GB
    import. Each yielded frame has the same column shape as a `BarProvider` frame
    (`ts, symbol, open, high, low, close, adj_close, volume`) so both seams converge at
    `algua.data.schema.to_bar_schema`. `vendor_label` is the provenance vendor name stamped into
    the snapshot (e.g. 'firstratedata').
    """

    name: str
    vendor_label: str

    def import_bars(self, request: ImportRequest) -> Iterator[ProviderBars]: ...
