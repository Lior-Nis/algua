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
    }], L.LedgerKind.LIVE)
    pre = conn.execute(
        "SELECT strategy FROM live_fills WHERE broker_order_id='broker-9'"
    ).fetchone()
    assert pre["strategy"] is None                    # orphaned: no strategy maps it yet
    assert L.believed_positions(conn, "s1", L.LedgerKind.LIVE) == {}  # s1 doesn't see the orphan

    # 2. The order record lands late (submit recorded its client_order_id), then the broker id
    #    is backfilled — which must back-attribute the already-ingested fill to s1.
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")
    L.backfill_broker_order_id(conn, "coid-1", "broker-9")

    post = conn.execute(
        "SELECT strategy FROM live_fills WHERE broker_order_id='broker-9'"
    ).fetchone()
    assert post["strategy"] == "s1"                    # attribution attached, fill not orphaned
    assert L.believed_positions(conn, "s1", L.LedgerKind.LIVE) == {"AAA": 10.0}

    # 3. Reconcile against the broker's matching net shows zero drift (clean, no halt).
    result = R.reconcile(conn, broker_net={"AAA": 10.0}, cycle=1)
    assert result.clean is True and result.halt is False and result.mismatches == []


def test_believed_positions_sums_signed_fills(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0, 100.0)
    _fill(conn, "a2", "s1", "AAA", -4.0, 110.0)
    _fill(conn, "a3", "s1", "BBB", 5.0, 50.0)
    assert L.believed_positions(conn, "s1", L.LedgerKind.LIVE) == {"AAA": 6.0, "BBB": 5.0}


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


class _RecoveryBroker:
    """Fake broker for #312 recovery: maps client_order_id -> order dict (None => 404 / not at
    venue). Records whether it was queried, to prove the DB-first guard skips the round-trip."""

    def __init__(self, orders):
        self._orders = orders
        self.queried: list[str] = []

    def get_order_by_client_order_id(self, coid):
        self.queried.append(coid)
        return self._orders.get(coid)


class _RaisingBroker:
    def get_order_by_client_order_id(self, coid):  # pragma: no cover - must never be called
        raise AssertionError("broker must not be queried when there are no stranded rows")


def test_recover_stranded_backfills_and_attributes_live(tmp_path):
    # A live order crashed after Alpaca accepted it but before backfill: row NULL and the fill
    # already ingested under the real broker id is orphaned (strategy NULL). Recovery must backfill
    # the broker id AND back-attribute the fill.
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")  # NULL broker_order_id
    L.ingest_activities(conn, [{
        "id": "act-1", "activity_type": "FILL", "side": "buy", "qty": "10", "price": "100",
        "order_id": "broker-9", "symbol": "AAA", "transaction_time": "2026-06-06T00:00:00+00:00",
    }], L.LedgerKind.LIVE)
    assert L.believed_positions(conn, "s1", L.LedgerKind.LIVE) == {}  # orphaned pre-recovery

    broker = _RecoveryBroker({"coid-1": {"id": "broker-9", "client_order_id": "coid-1",
                                         "symbol": "AAA", "status": "filled"}})
    out = L.recover_stranded_broker_order_ids(conn, broker, kind=L.LedgerKind.LIVE)

    assert out.recovered == ["coid-1"] and out.mismatched == []
    row = conn.execute(
        "SELECT broker_order_id FROM live_orders WHERE client_order_id='coid-1'").fetchone()
    assert row["broker_order_id"] == "broker-9"
    assert L.believed_positions(conn, "s1", L.LedgerKind.LIVE) == {"AAA": 10.0}


def test_recover_stranded_preserves_phantom_on_404_paper(tmp_path):
    # A pure #365 phantom (a genuine noop whose coid never reached the venue) 404s -> preserved.
    conn = _conn(tmp_path)
    L.record_paper_venue_order(conn, "s1", "AAA", "buy", None, "coid-x", strategy_id=1)
    broker = _RecoveryBroker({})  # nothing at the venue
    out = L.recover_stranded_broker_order_ids(conn, broker, kind=L.LedgerKind.PAPER)
    assert out.recovered == [] and out.mismatched == []
    assert broker.queried == ["coid-x"]  # it DID look, and found nothing
    row = conn.execute(
        "SELECT broker_order_id FROM paper_venue_orders WHERE client_order_id='coid-x'").fetchone()
    assert row["broker_order_id"] is None  # preserved, not deleted, not backfilled


def test_recover_stranded_skips_symbol_mismatch(tmp_path):
    # A coid collision surfaces as a symbol mismatch: never mis-attribute -> skip+flag, leave NULL.
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")
    broker = _RecoveryBroker({"coid-1": {"id": "broker-9", "client_order_id": "coid-1",
                                         "symbol": "BBB"}})  # WRONG symbol
    out = L.recover_stranded_broker_order_ids(conn, broker, kind=L.LedgerKind.LIVE)
    assert out.recovered == [] and out.mismatched == ["coid-1"]
    row = conn.execute(
        "SELECT broker_order_id FROM live_orders WHERE client_order_id='coid-1'").fetchone()
    assert row["broker_order_id"] is None


def test_recover_stranded_rejects_bad_coid_or_empty_id(tmp_path):
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-echo")   # returned coid differs
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-empty")  # empty id
    broker = _RecoveryBroker({
        "coid-echo": {"id": "b1", "client_order_id": "coid-OTHER", "symbol": "AAA"},
        "coid-empty": {"id": "", "client_order_id": "coid-empty", "symbol": "AAA"},
    })
    out = L.recover_stranded_broker_order_ids(conn, broker, kind=L.LedgerKind.LIVE)
    assert out.recovered == [] and sorted(out.mismatched) == ["coid-echo", "coid-empty"]


def test_recover_stranded_no_null_rows_skips_broker(tmp_path):
    # DB-first guard: with every row already backfilled, the broker is NEVER queried.
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")
    L.backfill_broker_order_id(conn, "coid-1", "broker-9")
    out = L.recover_stranded_broker_order_ids(conn, _RaisingBroker(), kind=L.LedgerKind.LIVE)
    assert out.recovered == [] and out.mismatched == []


def test_backfill_order_conditional_never_overwrites(tmp_path):
    # The shared conditional backfill sets a NULL row once, is idempotent for the same id, and
    # refuses to overwrite an already-set (different) broker id (concurrent-tick TOCTOU guard).
    conn = _conn(tmp_path)
    L.record_live_order(conn, "s1", "AAA", "buy", 1000.0, "coid-1")
    assert L._backfill_order(conn, L.LedgerKind.LIVE, "coid-1", "b1") is True
    assert L._backfill_order(conn, L.LedgerKind.LIVE, "coid-1", "b1") is True   # idempotent replay
    assert L._backfill_order(conn, L.LedgerKind.LIVE, "coid-1", "b2") is False  # no overwrite
    row = conn.execute(
        "SELECT broker_order_id FROM live_orders WHERE client_order_id='coid-1'").fetchone()
    assert row["broker_order_id"] == "b1"


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
