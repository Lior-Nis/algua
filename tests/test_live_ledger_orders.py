from algua.execution import live_ledger as L
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "l.db")
    migrate(conn)
    return conn


def _fill(conn, activity_id, strategy, symbol, qty, price, boid="b1"):
    conn.execute(
        "INSERT INTO live_fills"
        "(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (activity_id, boid, strategy, symbol, qty, price, "2026-06-06T00:00:00+00:00"),
    )
    conn.commit()


def test_record_order_is_idempotent_on_client_id(tmp_path):
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")  # retry, same client id
    n = conn.execute("SELECT COUNT(*) FROM live_orders").fetchone()[0]
    assert n == 1


def test_backfill_broker_order_id(tmp_path):
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")
    L.backfill_broker_order_id(conn, "coid-1", "broker-9")
    row = conn.execute("SELECT broker_order_id FROM live_orders WHERE client_order_id='coid-1'"
                       ).fetchone()
    assert row["broker_order_id"] == "broker-9"


def test_believed_positions_sums_signed_fills(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0, 100.0)
    _fill(conn, "a2", "s1", "AAA", -4.0, 110.0)
    _fill(conn, "a3", "s1", "BBB", 5.0, 50.0)
    assert L.believed_positions(conn, "s1") == {"AAA": 6.0, "BBB": 5.0}


def test_strategy_nav(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0, 100.0)  # long 10 @100
    # allocation 10_000; mark 105 -> unrealized 50, realized 0 -> NAV 10_050
    nav = L.strategy_nav(conn, "s1", allocation=10_000.0, marks={"AAA": 105.0})
    assert nav == 10_050.0
