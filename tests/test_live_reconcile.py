from algua.execution import live_reconcile as R
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def _fill(conn, aid, strategy, symbol, qty, price="100"):
    conn.execute(
        "INSERT INTO live_fills"
        "(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (aid, "b", strategy, symbol, qty, price, "2026-06-06T00:00:00+00:00"),
    )
    conn.commit()


def test_account_expected_net_sums_all_strategies(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0)
    _fill(conn, "a2", "s2", "AAA", 5.0)   # a sibling holds the same symbol
    _fill(conn, "a3", "s1", "BBB", -3.0)
    assert R.account_expected_net(conn) == {"AAA": 15.0, "BBB": -3.0}


def test_next_cycle_is_monotonic_and_persistent(tmp_path):
    conn = _conn(tmp_path)
    assert R.next_cycle(conn) == 1
    assert R.next_cycle(conn) == 2


def test_reconcile_clean_when_books_match_broker(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0)
    res = R.reconcile(conn, broker_net={"AAA": 10.0}, cycle=1)
    assert res.clean and not res.halt and res.mismatches == []


def test_reconcile_tolerates_rounding(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0)
    res = R.reconcile(conn, broker_net={"AAA": 10.0 + 5e-7}, cycle=1)
    assert res.clean and not res.halt


def test_reconcile_pending_then_escalates_to_halt(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0)
    broker = {"AAA": 12.0}  # broker holds 2 more than the books explain
    r1 = R.reconcile(conn, broker, cycle=1)
    assert not r1.clean and not r1.halt and r1.mismatches[0]["status"] == "pending"
    r2 = R.reconcile(conn, broker, cycle=2)
    assert not r2.clean and not r2.halt  # still within grace (default 3)
    r3 = R.reconcile(conn, broker, cycle=4)  # cycle - first_seen (1) >= 3 -> unexplained
    assert r3.halt and r3.mismatches[0]["status"] == "unexplained"


def test_reconcile_clears_pending_when_it_resolves(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0)
    R.reconcile(conn, {"AAA": 12.0}, cycle=1)            # pending recorded
    R.reconcile(conn, {"AAA": 10.0}, cycle=2)            # resolves -> row cleared
    assert conn.execute("SELECT COUNT(*) FROM live_reconcile_state").fetchone()[0] == 0
