"""Historical data provider adapters and their construction registry.

These are **ingestion** providers (the `BarProvider` seam in `algua.data.contracts`):
`get_bars(request: BarRequest) -> ProviderBars`, used by `algua data ingest-bars` to fetch and
snapshot raw vendor bars. This is a distinct seam from the **serving** `DataProvider`
(`algua.contracts.types`), whose `get_bars(symbols, start, end, timeframe) -> DataFrame` feeds the
backtest/live engines off a pinned snapshot.

Construction lives here (not in the CLI) so adding a provider is open-for-extension: register a
name->factory and `get_provider` picks it up — no if/elif ladder to edit.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from algua.data.contracts import BarProvider
from algua.data.providers.alpaca import AlpacaBarProvider
from algua.data.providers.yfinance import YFinanceBarProvider

if TYPE_CHECKING:
    from algua.config.settings import Settings

ProviderFactory = Callable[["Settings"], BarProvider]


def _build_yfinance(_settings: Settings) -> BarProvider:
    return YFinanceBarProvider()


def _build_alpaca(settings: Settings) -> BarProvider:
    if settings.alpaca_api_key is None or settings.alpaca_api_secret is None:
        raise ValueError(
            "alpaca provider requires ALGUA_ALPACA_API_KEY and ALGUA_ALPACA_API_SECRET"
        )
    return AlpacaBarProvider(
        api_key=settings.alpaca_api_key,
        api_secret=settings.alpaca_api_secret,
        base_url=settings.alpaca_data_url,
    )


_REGISTRY: dict[str, ProviderFactory] = {
    "yfinance": _build_yfinance,
    "alpaca": _build_alpaca,
}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a name->factory so `get_provider(name, settings)` can build it."""
    _REGISTRY[name] = factory


def get_provider(name: str, settings: Settings) -> BarProvider:
    """Construct an ingestion `BarProvider` by name from settings.

    Raises ValueError for an unknown name or missing provider credentials.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unsupported bar provider: {name}") from None
    return factory(settings)
