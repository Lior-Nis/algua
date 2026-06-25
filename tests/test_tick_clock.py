import pandas as pd

from algua.execution.alpaca_broker import BrokerError
from algua.execution.tick_clock import tick_clock


def test_tick_clock_uses_broker_clock_when_valid():
    ts, source = tick_clock(lambda: "2023-06-01T14:30:00+00:00")
    assert source == "broker"
    assert ts == pd.Timestamp("2023-06-01T14:30:00+00:00").tz_convert("UTC").isoformat()


def test_tick_clock_falls_back_on_broker_error():
    def _boom():
        raise BrokerError("clock down")
    ts, source = tick_clock(_boom)
    assert source == "local"
    assert ts.endswith("+00:00") or "T" in ts


def test_tick_clock_falls_back_on_naive_timestamp():
    # tz-naive -> tz_convert raises TypeError -> local fallback
    ts, source = tick_clock(lambda: "2023-06-01T14:30:00")
    assert source == "local"
