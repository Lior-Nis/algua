from __future__ import annotations

from datetime import date, datetime
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

    def session_close(self, session: date) -> datetime:
        """The UTC close instant of a trading `session`. xcals returns a UTC tz-aware Timestamp;
        e.g. session_close(2023-06-01) == 2023-06-01 20:00:00+00:00."""
        return self._cal.session_close(pd.Timestamp(session)).to_pydatetime()

    def bounds(self) -> tuple[datetime, datetime]:
        """The calendar's precomputed (first_minute, last_minute) horizon, UTC tz-aware. Used to
        tell a before-first-session instant (benign bring-up) apart from a past-the-horizon
        instant (a calendar that has RUN OUT and needs a refresh) — both raise ``MinuteOutOfBounds``
        from the underlying calendar, but only the latter is an operational anomaly."""
        return (
            self._cal.first_minute.to_pydatetime(),
            self._cal.last_minute.to_pydatetime(),
        )

    def session_of_instant(self, instant: datetime) -> date:
        """The exchange session an INSTANT belongs to (the session on-or-before it), mapped in
        EXCHANGE time — NOT the instant's UTC calendar date. An after-close tick such as
        2023-06-01T00:30:00+00:00 (= 2023-05-31 20:30 America/New_York, after the May 31 close)
        belongs to the MAY 31 session, though its UTC date is June 1. Using the UTC date would
        undercount elapsed sessions and let a dead loop read fresh. Aware datetimes are honoured;
        a naive datetime is treated as UTC. Instants before the calendar's first minute raise
        ``MinuteOutOfBounds`` (propagated — the caller decides how to fail closed)."""
        ts = pd.Timestamp(instant)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return self._cal.minute_to_session(ts, direction="previous").date()

    def sessions_between_instants(self, a: datetime, b: datetime) -> int:
        """Signed count of trading sessions from session-of-instant(a) to session-of-instant(b),
        each mapped in EXCHANGE time (see :meth:`session_of_instant`). This is the staleness metric
        for a heartbeat/liveness check: it counts COMPLETED market sessions since the last tick
        instant without the UTC-date rounding error of :meth:`sessions_between`."""
        return self.sessions_between(self.session_of_instant(a), self.session_of_instant(b))

    def sessions_stale(self, latest_bar: datetime, now: datetime) -> int:
        """Completed exchange sessions between the latest decided daily bar and `now`, a
        mark-freshness liveness metric. Daily bars are stamped at the session's UTC midnight
        (#262), which session_of_instant would misattribute to the PRIOR session, so map the bar
        by its session DATE (session date at UTC midnight, #262) and `now` in exchange time via
        session_of_instant. 0 == the freshest closed session; grows as the feed goes stale.

        The result is NOT clamped. A NEGATIVE result means the bar maps to a session AFTER `now`
        (clock skew or bad data) and is returned as-is so the caller fails closed as
        `future_dated`. ``session_of_instant`` may raise ``MinuteOutOfBounds`` for an out-of-range
        instant; that is propagated for the caller to convert into a fail-closed ``RiskBreach``."""
        return self.sessions_between(latest_bar.date(), self.session_of_instant(now))
