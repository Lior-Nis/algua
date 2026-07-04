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


def test_sessions_stale_latest_bar_prior_session():
    """Latest bar is from the session immediately before now's session -> returns 1."""
    cal = MarketCalendar()
    # 2024-07-02 is a Tuesday (session), 2024-07-03 is a Wednesday (session)
    latest_bar = datetime(2024, 7, 2, 0, 0, tzinfo=UTC)  # session date 2024-07-02
    now = datetime(2024, 7, 3, 15, 0, tzinfo=UTC)        # during 2024-07-03 session
    assert cal.sessions_stale(latest_bar, now) == 1


def test_sessions_stale_same_session():
    """Latest bar's session == now's session -> returns 0."""
    cal = MarketCalendar()
    # 2024-07-03 is a Wednesday (session)
    # UTC midnight 2024-07-03T00:00:00+00:00 = 2024-07-02 20:00 ET (still in Jul 2 session)
    latest_bar = datetime(2024, 7, 3, 0, 0, tzinfo=UTC)     # session date 2024-07-02
    now = datetime(2024, 7, 3, 20, 0, tzinfo=UTC)           # session date 2024-07-02
    assert cal.sessions_stale(latest_bar, now) == 0


def test_sessions_stale_bar_years_before():
    """A bar from years before now -> a large positive session count."""
    cal = MarketCalendar()
    # 2023-12-29 is a Friday (session)
    latest_bar = datetime(2023, 12, 29, 0, 0, tzinfo=UTC)  # session date 2023-12-29
    now = datetime(2026, 6, 15, 15, 0, tzinfo=UTC)         # a date in 2026
    staleness = cal.sessions_stale(latest_bar, now)
    assert staleness > 500  # roughly 2.5 years of trading sessions


def test_sessions_stale_bar_ahead_of_now():
    """A bar dated after now (clock skew) -> NEGATIVE, NOT clamped to 0, so the caller
    fails closed as future_dated."""
    cal = MarketCalendar()
    # 2024-07-05 is a Friday (session)
    latest_bar = datetime(2024, 7, 5, 0, 0, tzinfo=UTC)    # session date 2024-07-05
    now = datetime(2024, 7, 3, 15, 0, tzinfo=UTC)          # before the bar
    # sessions_between(2024-07-05, 2024-07-03) = -1, returned as-is (unclamped)
    assert cal.sessions_stale(latest_bar, now) == -1


def test_sessions_stale_post_close_before_utc_midnight():
    """A `now` after the exchange close but on the same UTC date as the bar's session
    UTC-midnight stamp -> staleness <= 1 (boundary case)."""
    cal = MarketCalendar()
    # 2023-06-01 is a Thursday (session); close is 16:00 ET = 20:00 UTC.
    latest_bar = datetime(2023, 6, 1, 0, 0, tzinfo=UTC)   # session date 2023-06-01
    now = datetime(2023, 6, 1, 20, 30, tzinfo=UTC)        # 16:30 ET, after close, same UTC date
    assert cal.sessions_stale(latest_bar, now) <= 1


def test_sessions_stale_weekend_now():
    """`now` on a weekend maps back to the Friday session == the bar's session -> 0."""
    cal = MarketCalendar()
    # 2024-07-05 is a Friday (session); 2024-07-06 is a Saturday.
    latest_bar = datetime(2024, 7, 5, 0, 0, tzinfo=UTC)   # session date 2024-07-05
    now = datetime(2024, 7, 6, 15, 0, tzinfo=UTC)         # Saturday -> maps back to Jul 5
    assert cal.sessions_stale(latest_bar, now) == 0
