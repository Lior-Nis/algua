"""Tests for the calendar factory (algua/calendar/factory.py, spec §5 item 2)."""

from __future__ import annotations

import exchange_calendars as xcals
import pytest

from algua.calendar import factory
from algua.calendar.market_calendar import MarketCalendar


def test_default_config_yields_xnys_calendar(monkeypatch):
    monkeypatch.delenv("ALGUA_EXCHANGE", raising=False)
    calendar = factory.get_calendar()
    assert isinstance(calendar, MarketCalendar)
    assert calendar.code == "XNYS"


def test_honours_settings_exchange(monkeypatch):
    monkeypatch.setenv("ALGUA_EXCHANGE", "XLON")
    calendar = factory.get_calendar()
    assert calendar.code == "XLON"


def test_settings_read_per_call_not_at_import(monkeypatch):
    # Regression guard for the Stage 5b failure mode: importing the factory must not bind
    # settings.exchange before a test's monkeypatch.setenv can take effect.
    monkeypatch.delenv("ALGUA_EXCHANGE", raising=False)
    assert factory.get_calendar().code == "XNYS"

    monkeypatch.setenv("ALGUA_EXCHANGE", "XLON")
    assert factory.get_calendar().code == "XLON"


def test_invalid_exchange_code_fails_rather_than_falling_back(monkeypatch):
    monkeypatch.setenv("ALGUA_EXCHANGE", "NOT_A_REAL_EXCHANGE")
    with pytest.raises(xcals.errors.InvalidCalendarName):
        factory.get_calendar()
