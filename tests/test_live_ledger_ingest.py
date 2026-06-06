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
