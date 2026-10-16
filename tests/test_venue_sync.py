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
    assert venue.calls == [("activities", "2026-03-02T13:30:00.123456+00:00", UNTIL)]
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


# --- #691: a late-published activity must not be skipped by the time cursor ----------------------

def _seed_cursor(conn, value: str) -> None:
    with conn:
        conn.execute("INSERT INTO paper_venue_fill_cursor(name, cursor) VALUES ('activities', ?)",
                     (value,))


def test_ingest_rereads_a_lookback_window_before_the_cursor(conn):
    """The window opens ``PAPER_INGEST_LOOKBACK`` before the stored cursor, so an activity timed
    before the cursor but published after the previous read is still fetched (dedup by
    activity_id makes the overlap free)."""
    _seed_cursor(conn, CURSOR)
    venue = _venue()
    venue_sync.ingest_paper_venue(conn, venue, UNTIL)
    ((_, after, until),) = venue.calls
    assert until == UNTIL
    assert after == "2026-03-02T13:30:00.123456+00:00"  # CURSOR - 30 min
    assert _cursor(conn) == UNTIL


def test_late_published_activity_is_ingested_on_the_next_sync(conn):
    venue = FakePaperVenue()
    _seed_cursor(conn, "2026-03-02T15:00:00+00:00")
    venue_sync.ingest_paper_venue(conn, venue, "2026-03-02T15:10:00+00:00")  # feed empty
    assert believed_positions(conn, "s1", LedgerKind.PAPER) == {}
    # The venue now publishes a fill timed BEFORE the cursor that was just persisted.
    venue.feed.append({"id": "late-1", "activity_type": "FILL", "side": "buy", "qty": "2",
                       "price": "100", "symbol": "AAPL", "order_id": "boid-1",
                       "transaction_time": "2026-03-02T15:05:00+00:00"})
    venue_sync.ingest_paper_venue(conn, venue, "2026-03-02T15:20:00+00:00")
    assert believed_positions(conn, "s1", LedgerKind.PAPER) == {"AAPL": 2.0}
    venue_sync.ingest_paper_venue(conn, venue, "2026-03-02T15:30:00+00:00")  # overlap re-dedups
    assert believed_positions(conn, "s1", LedgerKind.PAPER) == {"AAPL": 2.0}


def test_far_past_sentinel_is_not_shifted(conn):
    venue = _venue()
    venue_sync.ingest_paper_venue(conn, venue, UNTIL)
    assert venue.calls == [("activities", venue_sync._PAPER_CURSOR_FAR_PAST, UNTIL)]


def test_unparseable_cursor_fails_closed(conn):
    _seed_cursor(conn, "not-a-time")
    with pytest.raises(ValueError):
        venue_sync.ingest_paper_venue(conn, _venue(), UNTIL)
