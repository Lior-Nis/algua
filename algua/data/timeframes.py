# algua/data/timeframes.py
"""Canonical timeframe vocabulary for the data layer (issue #151).

Single source of truth for the closed set of bar timeframes the system accepts, plus the
daily-vs-intraday classification the FirstRate importer keys its tz-localization off. Pure
(stdlib only) so any data-layer module can import it without a cycle.
"""
from __future__ import annotations

DAILY = "1d"
# FirstRate's actual intraday offerings. Extend deliberately if a new vendor needs another token.
INTRADAY: frozenset[str] = frozenset({"1m", "5m", "30m", "1h"})
KNOWN: frozenset[str] = frozenset({DAILY, *INTRADAY})


def validate_timeframe(timeframe: str) -> str:
    """Return `timeframe` if it is a known token; raise `ValueError` otherwise."""
    if timeframe not in KNOWN:
        raise ValueError(
            f"unknown timeframe {timeframe!r}; expected one of {sorted(KNOWN)}"
        )
    return timeframe


def is_intraday(timeframe: str) -> bool:
    """True for an intraday timeframe, False for daily. Validates first (raises on unknown)."""
    return validate_timeframe(timeframe) in INTRADAY
