"""Task-3: paginated, fail-closed paper venue ingest helper (broker-time cursor).

Tests confirm:
  - first call defaults to _FAR_PAST and advances cursor to broker.clock()
  - transport failure propagates and leaves cursor untouched (fail-closed)
"""
import sqlite3

import pytest

from algua.cli.paper_cmd import _ingest_paper_venue
from algua.execution.live_ledger import LedgerKind, fill_cursor, paper_believed_positions
from algua.registry.db import migrate

_FAR_PAST = "1970-01-01T00:00:00Z"


class FakeBroker:
    def __init__(self, windows: dict, clock: str = "2026-01-02T00:00:00Z"):
        # windows: dict cursor -> list[activity] OR Exception instance to raise
        self._windows = windows
        self._clock = clock

    def clock(self) -> str:
        return self._clock

    def account_activities_window(self, after: str, until: str) -> list[dict]:
        out = self._windows.get(after)
        if isinstance(out, Exception):
            raise out
        return out or []


def _conn(tmp_path):
    c = sqlite3.connect(tmp_path / "r.db")
    c.row_factory = sqlite3.Row
    migrate(c)
    return c


def _fill(aid: str, sym: str, qty: float, side: str, oid: str) -> dict:
    return {
        "id": aid,
        "activity_type": "FILL",
        "side": side,
        "qty": abs(qty),
        "price": 10.0,
        "symbol": sym,
        "order_id": oid,
        "transaction_time": "2026-01-01T12:00:00Z",
    }


def test_ingest_uses_far_past_first_then_advances_cursor(tmp_path):
    c = _conn(tmp_path)
    c.execute(
        "INSERT INTO paper_venue_orders(strategy,symbol,side,client_order_id,broker_order_id,"
        "strategy_id,status,submitted_ts) VALUES ('s','AAA','buy','c','o1',1,'submitted','t')"
    )
    c.commit()
    broker = FakeBroker({_FAR_PAST: [_fill("a1", "AAA", 5, "buy", "o1")]})
    _ingest_paper_venue(c, broker)
    assert paper_believed_positions(c, "s") == {"AAA": 5.0}
    assert fill_cursor(c, LedgerKind.PAPER) == "2026-01-02T00:00:00Z"  # = until (broker clock)


def test_ingest_fails_closed_on_transport_error(tmp_path):
    c = _conn(tmp_path)
    broker = FakeBroker({_FAR_PAST: RuntimeError("503")})
    with pytest.raises(RuntimeError):
        _ingest_paper_venue(c, broker)
    assert fill_cursor(c, LedgerKind.PAPER) is None  # cursor must NOT advance on failure
