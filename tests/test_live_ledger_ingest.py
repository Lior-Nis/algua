import pytest

from algua.execution import live_ledger as L
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "i.db")
    migrate(conn)
    return conn


def _fill_act(aid, order_id, symbol, side, qty, price):
    return {"id": aid, "activity_type": "FILL", "order_id": order_id, "symbol": symbol,
            "side": side, "qty": str(qty), "price": str(price),
            "transaction_time": "2026-06-06T00:00:00Z"}


def test_ingest_signs_qty_and_attributes_by_order(tmp_path):
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")
    L.backfill_broker_order_id(conn, "coid-1", "order-1")
    L.ingest_activities(conn, [_fill_act("act-1", "order-1", "AAA", "buy", 10, 100.0),
                               _fill_act("act-2", "order-1", "AAA", "sell", 4, 110.0)])
    assert L.believed_positions(conn, "s1") == {"AAA": 6.0}  # +10 then -4, attributed to s1


def test_ingest_is_idempotent_on_activity_id(tmp_path):
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")
    L.backfill_broker_order_id(conn, "coid-1", "order-1")
    acts = [_fill_act("act-1", "order-1", "AAA", "buy", 10, 100.0)]
    L.ingest_activities(conn, acts)
    L.ingest_activities(conn, acts)  # replay (overlap window) — must not double-count
    assert L.believed_positions(conn, "s1") == {"AAA": 10.0}
    assert conn.execute("SELECT COUNT(*) FROM live_fills").fetchone()[0] == 1


def test_ingest_records_cash_activity_to_live_activities(tmp_path):
    conn = _conn(tmp_path)
    L.ingest_activities(conn, [{"id": "div-1", "activity_type": "DIV", "symbol": "AAA",
                                "net_amount": "12.50", "date": "2026-06-06"}])
    row = conn.execute("SELECT type, amount FROM live_activities WHERE activity_id='div-1'"
                       ).fetchone()
    assert row["type"] == "DIV" and row["amount"] == 12.50
    assert conn.execute("SELECT COUNT(*) FROM live_fills").fetchone()[0] == 0


def test_cursor_advances_to_latest_id(tmp_path):
    conn = _conn(tmp_path)
    L.ingest_activities(conn, [{"id": "z-9", "activity_type": "DIV", "net_amount": "1",
                                "date": "d"}])
    assert L.fill_cursor(conn) == "z-9"


def test_late_attribution_backfills_orphan_fill(tmp_path):
    # a fill that arrives BEFORE its order mapping exists is recorded with strategy NULL,
    # then attributed once record_live_order + backfill land (codex HIGH #1)
    conn = _conn(tmp_path)
    L.ingest_activities(conn, [_fill_act("act-1", "order-7", "AAA", "buy", 10, 100.0)])
    assert L.believed_positions(conn, "s1") == {}  # not yet attributed
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-7")
    L.backfill_broker_order_id(conn, "coid-7", "order-7")
    assert L.believed_positions(conn, "s1") == {"AAA": 10.0}  # back-attributed


def test_late_attribution_via_repull(tmp_path):
    # the overlap-window re-pull also re-attributes (COALESCE on conflict)
    conn = _conn(tmp_path)
    L.ingest_activities(conn, [_fill_act("act-1", "order-7", "AAA", "buy", 10, 100.0)])
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-7")
    L.backfill_broker_order_id(conn, "coid-7", "order-7")  # (already attributes here)
    L.ingest_activities(conn, [_fill_act("act-1", "order-7", "AAA", "buy", 10, 100.0)])  # replay
    assert L.believed_positions(conn, "s1") == {"AAA": 10.0}  # still 10, not doubled


def _quarantine_ids(conn):
    return [r["activity_id"]
            for r in conn.execute("SELECT activity_id FROM live_activity_quarantine ORDER BY id")]


def test_malformed_activity_is_quarantined_not_raised(tmp_path):
    # A malformed activity must NOT wedge the loop: it is dead-lettered, not raised (#250).
    conn = _conn(tmp_path)
    L.ingest_activities(conn, [{"id": "x", "activity_type": "FILL", "order_id": "o",
                                "symbol": "AAA", "side": "hold", "qty": "1", "price": "1",
                                "transaction_time": "t"}])
    assert _quarantine_ids(conn) == ["x"]
    assert conn.execute("SELECT COUNT(*) FROM live_fills").fetchone()[0] == 0
    # the cursor still advanced PAST the poison item, so the next pull won't re-fetch it forever
    assert L.fill_cursor(conn) == "x"


def test_poison_does_not_drop_good_activities_in_same_batch(tmp_path):
    # good fill + a poison fill (bad side) + a poison fill (missing qty -> KeyError) + good div,
    # interleaved: the good ones land, both poisons are quarantined, cursor = max id over the batch.
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")
    L.backfill_broker_order_id(conn, "coid-1", "order-1")
    L.ingest_activities(conn, [
        _fill_act("a-1", "order-1", "AAA", "buy", 10, 100.0),
        {"id": "a-2", "activity_type": "FILL", "order_id": "order-1", "symbol": "AAA",
         "side": "hold", "qty": "1", "price": "1", "transaction_time": "t"},  # bad side
        {"id": "a-3", "activity_type": "FILL", "order_id": "order-1", "symbol": "AAA",
         "side": "buy", "price": "1", "transaction_time": "t"},  # missing qty -> KeyError
        {"id": "a-4", "activity_type": "DIV", "symbol": "AAA", "net_amount": "5", "date": "d"},
    ])
    assert L.believed_positions(conn, "s1") == {"AAA": 10.0}  # only the good fill counts
    assert _quarantine_ids(conn) == ["a-2", "a-3"]
    assert conn.execute(
        "SELECT amount FROM live_activities WHERE activity_id='a-4'").fetchone()["amount"] == 5.0
    assert L.fill_cursor(conn) == "a-4"  # advanced to the max id, past the poisons


def test_quarantine_dedups_on_replay(tmp_path):
    conn = _conn(tmp_path)
    poison = [{"id": "p-1", "activity_type": "FILL", "order_id": "o", "symbol": "AAA",
               "side": "hold", "qty": "1", "price": "1", "transaction_time": "t"}]
    L.ingest_activities(conn, poison)
    L.ingest_activities(conn, poison)  # overlap re-pull must not double-quarantine
    assert _quarantine_ids(conn) == ["p-1"]


def test_infra_error_rolls_back_whole_batch(tmp_path, monkeypatch):
    # A non-shape error (e.g. a DB failure) is NOT quarantined: it propagates and rolls back the
    # whole batch, preserving the old all-or-nothing fail-closed behavior.
    import sqlite3

    conn = _conn(tmp_path)
    calls = {"n": 0}
    real = L._ingest_one_activity

    def _boom(c, act, aid):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("disk I/O error")
        return real(c, act, aid)

    monkeypatch.setattr(L, "_ingest_one_activity", _boom)
    with pytest.raises(sqlite3.OperationalError):
        L.ingest_activities(conn, [{"id": "div-1", "activity_type": "DIV", "net_amount": "1",
                                    "date": "d"},
                                   {"id": "div-2", "activity_type": "DIV", "net_amount": "1",
                                    "date": "d"}])
    # first item's insert was rolled back with the batch; cursor never advanced
    assert conn.execute("SELECT COUNT(*) FROM live_activities").fetchone()[0] == 0
    assert L.fill_cursor(conn) is None
