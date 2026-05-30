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
        # next_session() requires a valid session; use date_to_session(direction='next')
        # which accepts any date (including holidays/weekends) and returns the next session.
        return self._cal.date_to_session(pd.Timestamp(day), direction="next").date()

    def previous_session(self, day: date) -> date:
        # previous_session() requires a valid session; use date_to_session(direction='previous')
        # which accepts any date and returns the most recent prior session.
        return self._cal.date_to_session(pd.Timestamp(day), direction="previous").date()

    def sessions_in_range(self, start: date, end: date) -> list[date]:
        idx = self._cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        return [ts.date() for ts in idx]
