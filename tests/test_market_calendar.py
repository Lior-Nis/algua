from __future__ import annotations

from datetime import date

from algua.calendar.market_calendar import MarketCalendar


def test_session_on_or_before():
    cal = MarketCalendar()
    assert cal.session_on_or_before(date(2024, 7, 5)) == date(2024, 7, 5)
    assert cal.session_on_or_before(date(2024, 7, 4)) == date(2024, 7, 3)
    assert cal.session_on_or_before(date(2024, 7, 7)) == date(2024, 7, 5)


def test_sessions_between():
    cal = MarketCalendar()
    assert cal.sessions_between(date(2024, 7, 3), date(2024, 7, 3)) == 0
    assert cal.sessions_between(date(2024, 7, 3), date(2024, 7, 5)) == 1   # holiday skipped
    assert cal.sessions_between(date(2024, 7, 3), date(2024, 7, 8)) == 2   # weekend skipped
    assert cal.sessions_between(date(2024, 7, 8), date(2024, 7, 3)) == -2  # signed
    assert cal.sessions_between(date(2024, 7, 6), date(2024, 7, 7)) == 0   # both map to Jul 5
