from algua.execution import live_ledger as L
from algua.execution import live_reconcile as R
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


def test_backfill_back_attributes_fill_ingested_before_order_mapping(tmp_path):
    # #166 gap 1: a FILL can land (via the activities feed) BEFORE its order's broker_order_id
    # mapping exists — e.g. a submit timed out after Alpaca accepted it, so the fill arrives while
    # live_orders.broker_order_id is still NULL. The fill is then ingested with strategy=NULL
    # (no live_orders row maps the broker order id), and the strategy must attach once the mapping
    # lands via backfill_broker_order_id — never orphaned. Reconcile must then show zero drift.
    conn = _conn(tmp_path)
    # 1. The fill arrives first, attributed to no strategy (the mapping does not exist yet).
    L.ingest_activities(conn, [{
        "id": "act-1", "activity_type": "FILL", "side": "buy", "qty": "10", "price": "100",
        "order_id": "broker-9", "symbol": "AAA", "transaction_time": "2026-06-06T00:00:00+00:00",
    }])
    pre = conn.execute(
        "SELECT strategy FROM live_fills WHERE broker_order_id='broker-9'"
    ).fetchone()
    assert pre["strategy"] is None                    # orphaned: no strategy maps it yet
    assert L.believed_positions(conn, "s1") == {}     # s1's books don't see the orphan fill

    # 2. The order record lands late (submit recorded its client_order_id), then the broker id
    #    is backfilled — which must back-attribute the already-ingested fill to s1.
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")
    L.backfill_broker_order_id(conn, "coid-1", "broker-9")

    post = conn.execute(
        "SELECT strategy FROM live_fills WHERE broker_order_id='broker-9'"
    ).fetchone()
    assert post["strategy"] == "s1"                    # attribution attached, fill not orphaned
    assert L.believed_positions(conn, "s1") == {"AAA": 10.0}

    # 3. Reconcile against the broker's matching net shows zero drift (clean, no halt).
    result = R.reconcile(conn, broker_net={"AAA": 10.0}, cycle=1)
    assert result.clean is True and result.halt is False and result.mismatches == []


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


def test_owned_open_order_ids_filters_to_strategy(tmp_path):
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "c1")
    L.record_live_order(conn, "s2", "BBB", "buy", 1000.0, "c2")

    class _B:
        def list_open_orders(self):
            return [{"id": "o1", "client_order_id": "c1"},   # s1's
                    {"id": "o2", "client_order_id": "c2"},   # s2's
                    {"id": "o3", "client_order_id": "unknown"}]  # not ours

    assert L.owned_open_order_ids(conn, _B(), "s1") == ["o1"]


def test_strategy_live_symbols_unions_orders_and_fills(tmp_path):
    from contextlib import closing

    from algua.execution.live_ledger import record_live_order, strategy_live_symbols
    from algua.registry.db import connect, migrate
    with closing(connect(tmp_path / "p.db")) as conn:
        migrate(conn)
        record_live_order(conn, "alpha", "AAA", "buy", None, "coid-aaa")
        # a fill in a symbol with no surviving order row (e.g. dropped from the universe)
        conn.execute(
            "INSERT INTO live_fills(activity_id, broker_order_id, strategy, symbol, qty, price, "
            "fill_ts) VALUES (?,?,?,?,?,?,?)",
            ("act-zzz", "bo-zzz", "alpha", "ZZZ", 3.0, 100.0, "2023-01-01T00:00:00Z"),
        )
        conn.commit()
        record_live_order(conn, "beta", "BBB", "buy", None, "coid-bbb")

        assert strategy_live_symbols(conn, "alpha") == {"AAA", "ZZZ"}
        assert strategy_live_symbols(conn, "beta") == {"BBB"}
