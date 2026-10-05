"""Story 2.2 §3.2: the paper venue sync. ``ingest_paper_venue`` (the paper operator's) advances the
shared fill cursor to its ``until``; ``ingest_paper_venue_keep_cursor`` (the exit drain's) fetches
the same ``(cursor, until]`` window and re-stores the cursor it read, so it never moves it."""
from __future__ import annotations

import pytest

from algua.execution import venue_sync
from algua.execution.live_ledger import LedgerKind, believed_positions, record_paper_venue_order
from algua.registry.db import connect, migrate
from tests._exit_drain import FakePaperVenue

CURSOR = "2026-03-02T14:00:00.123456789+00:00"
UNTIL = "2026-03-02T16:00:00+00:00"


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "reg.db")
    migrate(c)
    record_paper_venue_order(c, "s1", "AAPL", "buy", 100.0, "coid-1", strategy_id=1)
    with c:
        c.execute("UPDATE paper_venue_orders SET broker_order_id='boid-1'")
    return c


def _cursor(conn) -> str | None:
    row = conn.execute(
        "SELECT cursor FROM paper_venue_fill_cursor WHERE name='activities'").fetchone()
    return None if row is None else row["cursor"]


def _venue() -> FakePaperVenue:
    venue = FakePaperVenue()
    venue.feed.append({"id": "act-1", "activity_type": "FILL", "side": "buy", "qty": "2",
                       "price": "100", "symbol": "AAPL", "order_id": "boid-1",
                       "transaction_time": "2026-03-02T15:00:00+00:00"})
    return venue


def test_keep_cursor_ingest_fetches_the_window_and_keeps_the_cursor_bytes(conn):
    with conn:
        conn.execute("INSERT INTO paper_venue_fill_cursor(name, cursor) VALUES ('activities', ?)",
                     (CURSOR,))
    venue = _venue()
    venue_sync.ingest_paper_venue_keep_cursor(conn, venue, UNTIL)
    assert venue.calls == [("activities", CURSOR, UNTIL)]
    assert believed_positions(conn, "s1", LedgerKind.PAPER) == {"AAPL": 2.0}
    assert _cursor(conn) == CURSOR
    venue_sync.ingest_paper_venue_keep_cursor(conn, venue, UNTIL)  # re-fetch is idempotent
    assert believed_positions(conn, "s1", LedgerKind.PAPER) == {"AAPL": 2.0}
    assert _cursor(conn) == CURSOR


def test_keep_cursor_ingest_without_a_cursor_row_stores_the_far_past_sentinel(conn):
    venue = _venue()
    venue_sync.ingest_paper_venue_keep_cursor(conn, venue, UNTIL)
    assert venue.calls == [("activities", venue_sync._PAPER_CURSOR_FAR_PAST, UNTIL)]
    assert _cursor(conn) == venue_sync._PAPER_CURSOR_FAR_PAST


def test_the_operators_ingest_still_advances_the_cursor_to_until(conn):
    with conn:
        conn.execute("INSERT INTO paper_venue_fill_cursor(name, cursor) VALUES ('activities', ?)",
                     (CURSOR,))
    venue_sync.ingest_paper_venue(conn, _venue(), UNTIL)
    assert _cursor(conn) == UNTIL
