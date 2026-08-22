"""Settings-honouring construction of the trading calendar (spec §5 item 2).

One place reads ``settings.exchange`` and turns it into a ``MarketCalendar``, so every operational
path agrees on which exchange it is running. Before this seam, ``settings.exchange`` was read by
exactly one site -- a ``doctor`` probe that does no work with it -- while the go-live certificate,
forward gate, fleet-health gate, operator session gate and live mark-staleness all ran XNYS
regardless of configuration.

Lives here rather than in ``market_calendar.py`` so that module stays a config-free leaf: the pure
backtest lane imports it (``backtest/_sample.py``) and must not acquire a settings dependency.

NOT a registry: unlike the broker and tracker seams there is one implementation and the selector is
a plain exchange code, so a name->factory table would add indirection without extensibility.
``MarketCalendar._get_calendar`` is already ``@cache``d per code, so this adds no caching either.
"""

from __future__ import annotations

from algua.calendar.market_calendar import MarketCalendar
from algua.config.settings import get_settings


def get_calendar(code: str | None = None) -> MarketCalendar:
    """The configured trading calendar. ``code`` overrides ``settings.exchange`` (for tests).

    Reads settings per call, never at import -- an import-time read would bind the value before a
    test's ``monkeypatch.setenv`` could take effect, which is exactly how Stage 5b silently disarmed
    a go-live guard.
    """
    return MarketCalendar(code if code is not None else get_settings().exchange)
