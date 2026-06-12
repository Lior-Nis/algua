from __future__ import annotations

from datetime import date
from functools import cache

import exchange_calendars as xcals
import pandas as pd


# xcals.get_calendar is heavy (parses ~20 years of session data on first call per code).
# Cache by exchange code so repeated MarketCalendar("XNYS") constructions reuse one object.
@cache
def _get_calendar(code: str) -> xcals.ExchangeCalendar:
    return xcals.get_calendar(code)


class MarketCalendar:
    """Thin wrapper over exchange_calendars. Both backtest and live depend on this."""

    def __init__(self, code: str = "XNYS") -> None:
        self.code = code
        self._cal = _get_calendar(code)

    def is_session(self, day: date) -> bool:
        return bool(self._cal.is_session(pd.Timestamp(day)))

    def next_session(self, day: date) -> date:
        """The earliest trading session strictly after `day` (works for any day)."""
        ts = pd.Timestamp(day)
        anchor = self._cal.date_to_session(ts, direction="previous")
        return self._cal.next_session(anchor).date()

    def previous_session(self, day: date) -> date:
        """The latest trading session strictly before `day` (works for any day)."""
        ts = pd.Timestamp(day)
        anchor = self._cal.date_to_session(ts, direction="next")
        return self._cal.previous_session(anchor).date()

    def sessions_in_range(self, start: date, end: date) -> list[date]:
        idx = self._cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        return [ts.date() for ts in idx]

    def session_on_or_before(self, day: date) -> date:
        """The trading session containing `day`, or the latest session before it."""
        return self._cal.date_to_session(pd.Timestamp(day), direction="previous").date()

    def sessions_between(self, a: date, b: date) -> int:
        """Signed count of trading sessions from session-of(a) to session-of(b): 0 when both
        map to the same session, positive when b is later. Non-session days map backward
        (session_on_or_before) first."""
        sa, sb = self.session_on_or_before(a), self.session_on_or_before(b)
        lo, hi = (sa, sb) if sb >= sa else (sb, sa)
        n = len(self._cal.sessions_in_range(pd.Timestamp(lo), pd.Timestamp(hi))) - 1
        return n if sb >= sa else -n
