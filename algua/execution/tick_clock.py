from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd

from algua.execution.errors import BrokerError


def tick_clock(clock: Callable[[], str]) -> tuple[str, str]:
    """``(tick_ts, clock_source)`` for the evidence-tick stamp: the venue's clock normalized to a
    UTC ISO timestamp (``clock_source="broker"``), or the local clock (``clock_source="local"``)
    when the venue's is unusable. ValueError/TypeError cover a malformed or tz-naive venue
    timestamp — that is a clock failure too, and it must never kill the tick record after orders
    already went out. Shared by the paper and live lanes so their stamping semantics cannot drift.
    """
    try:
        return pd.Timestamp(clock()).tz_convert("UTC").isoformat(), "broker"
    except (BrokerError, ValueError, TypeError):
        return datetime.now(UTC).isoformat(), "local"
