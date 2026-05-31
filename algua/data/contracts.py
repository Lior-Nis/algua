from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class BarRequest:
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
    name: str

    def get_bars(self, request: BarRequest) -> ProviderBars: ...
