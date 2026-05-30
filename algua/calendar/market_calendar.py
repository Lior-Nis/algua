from __future__ import annotations

from datetime import date

import exchange_calendars as xcals
import pandas as pd


class MarketCalendar:
    """Thin wrapper over exchange_calendars. Both backtest and live depend on this."""

    def __init__(self, code: str = "XNYS") -> None:
        self.code = code
        self._cal = xcals.get_calendar(code)

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

    def session_closes(self, start: date, end: date) -> list[pd.Timestamp]:
        """Session close timestamps (tz-aware UTC) for each session in [start, end].

        Per the bar schema, a daily bar's timestamp is the session close (when the bar became
        known); this is DST- and half-day-aware (e.g. early closes return 17:00 UTC).
        """
        sessions = self._cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        return [self._cal.session_close(s) for s in sessions]
