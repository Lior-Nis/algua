from __future__ import annotations

from datetime import UTC, date, datetime

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


def test_session_of_instant_maps_in_exchange_time():
    cal = MarketCalendar()
    # 2023-06-01T00:30:00+00:00 = 2023-05-31 20:30 America/New_York (after the May 31 close):
    # belongs to the MAY 31 session, NOT its UTC date (June 1).
    after_close = datetime(2023, 6, 1, 0, 30, tzinfo=UTC)
    assert after_close.date() == date(2023, 6, 1)                         # UTC date rolls forward
    assert cal.session_of_instant(after_close) == date(2023, 5, 31)      # session is May 31
    # a during-session instant maps to its own session; a weekend instant maps backward.
    assert cal.session_of_instant(datetime(2023, 6, 1, 15, 0, tzinfo=UTC)) == date(2023, 6, 1)
    assert cal.session_of_instant(datetime(2023, 6, 10, 12, 0, tzinfo=UTC)) == date(2023, 6, 9)
    # a naive datetime is treated as UTC.
    assert cal.session_of_instant(datetime(2023, 6, 1, 0, 30)) == date(2023, 5, 31)


def test_sessions_between_instants_counts_completed_sessions():
    cal = MarketCalendar()
    tick = datetime(2023, 6, 1, 0, 30, tzinfo=UTC)   # May 31 session (after close)
    now = datetime(2023, 6, 8, 20, 0, tzinfo=UTC)    # Jun 8 session
    # Jun 1,2,5,6,7,8 = 6 completed sessions since the May 31 session.
    assert cal.sessions_between_instants(tick, now) == 6
    # ... vs the UTC-date rounding which would undercount by one.
    assert cal.sessions_between(tick.date(), now.date()) == 5
