from datetime import date

from algua.calendar.market_calendar import MarketCalendar


def test_holiday_is_not_a_session():
    cal = MarketCalendar("XNYS")
    assert cal.is_session(date(2025, 7, 4)) is False   # Independence Day
    assert cal.is_session(date(2025, 1, 1)) is False   # New Year's Day


def test_trading_day_is_a_session():
    cal = MarketCalendar("XNYS")
    assert cal.is_session(date(2025, 7, 3)) is True


def test_next_and_previous_session_skip_holiday():
    cal = MarketCalendar("XNYS")
    assert cal.next_session(date(2025, 7, 4)) == date(2025, 7, 7)      # Monday
    assert cal.previous_session(date(2025, 7, 4)) == date(2025, 7, 3)


def test_sessions_in_range_count():
    cal = MarketCalendar("XNYS")
    sessions = cal.sessions_in_range(date(2025, 7, 1), date(2025, 7, 7))
    # Jul 1,2,3 trading; Jul 4 holiday; Jul 5,6 weekend; Jul 7 trading
    assert sessions == [date(2025, 7, 1), date(2025, 7, 2),
                        date(2025, 7, 3), date(2025, 7, 7)]


def test_next_session_on_trading_day_is_strictly_after():
    cal = MarketCalendar("XNYS")
    # 2025-07-07 is a trading day; the next session must be the FOLLOWING day, not itself
    assert cal.next_session(date(2025, 7, 7)) == date(2025, 7, 8)


def test_previous_session_on_trading_day_is_strictly_before():
    cal = MarketCalendar("XNYS")
    # 2025-07-08 is a trading day; the previous session must be 2025-07-07, not itself
    assert cal.previous_session(date(2025, 7, 8)) == date(2025, 7, 7)
