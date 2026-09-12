"""Test helper: the decision timestamp a real tick records for "the freshest closed session".

A live/paper tick stores ``decision_ts`` as the decided DAILY BAR's timestamp — the session's
date at UTC midnight (#262) — never as a wall-clock instant. ``sessions_stale`` maps that bar by
its DATE, so a fixture that writes ``decision_ts=datetime.now(UTC)`` is future-dated from the
exchange's point of view for the first ~13.5 hours of every UTC weekday (00:00 UTC until the
09:30 ET open the UTC date is one session AHEAD of ``session_of_instant(now)``), reads as
``n < 0`` and fails closed to ``stale`` (#632). Fixtures use this helper instead.
"""
from __future__ import annotations

from datetime import UTC, datetime, time

from algua.calendar.market_calendar import MarketCalendar


def fresh_decision_ts(now: datetime | None = None, *, exchange: str = "XNYS") -> str:
    """ISO timestamp of the freshest closed session's daily bar as of ``now`` (UTC midnight of
    ``session_of_instant(now)``), i.e. ``sessions_stale(...) == 0`` at ``now``."""
    now = now or datetime.now(UTC)
    session = MarketCalendar(exchange).session_of_instant(now)
    return datetime.combine(session, time(), tzinfo=UTC).isoformat()
