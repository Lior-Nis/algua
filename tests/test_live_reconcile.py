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


def _live_strategy(conn, name):
    conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) "
        "VALUES (?, 'live', ?, ?)",
        (name, "2026-06-06T00:00:00+00:00", "2026-06-06T00:00:00+00:00"),
    )
    conn.commit()


def test_attributed_live_net_excludes_orphan_and_non_live(tmp_path):
    conn = _conn(tmp_path)
    _live_strategy(conn, "live_a")            # a registered, currently-live strategy
    _fill(conn, "a1", "live_a", "AAA", 5.0)   # counted (attributed + live)
    _fill(conn, "a2", None, "AAA", 7.0)        # orphan (manual/external) — must be EXCLUDED
    _fill(conn, "a3", "demoted", "BBB", 9.0)   # non-live strategy (no live row) — EXCLUDED
    # account_expected_net counts everything; attributed_live_net only the live-attributed fill
    assert R.account_expected_net(conn) == {"AAA": 12.0, "BBB": 9.0}
    assert R.attributed_live_net(conn) == {"AAA": 5.0}


def test_next_cycle_is_monotonic_and_persistent(tmp_path):
    conn = _conn(tmp_path)
    assert R.next_cycle(conn) == 1
    assert R.next_cycle(conn) == 2


def test_reconcile_clean_when_books_match_broker(tmp_path):
    conn = _conn(tmp_path)
    _live_strategy(conn, "s1")
    _fill(conn, "a1", "s1", "AAA", 10.0)
    res = R.reconcile(conn, broker_net={"AAA": 10.0}, cycle=1)
    assert res.clean and not res.halt and res.mismatches == []


def test_reconcile_tolerates_rounding(tmp_path):
    conn = _conn(tmp_path)
    _live_strategy(conn, "s1")
    _fill(conn, "a1", "s1", "AAA", 10.0)
    res = R.reconcile(conn, broker_net={"AAA": 10.0 + 5e-7}, cycle=1)
    assert res.clean and not res.halt


def test_reconcile_pending_then_escalates_to_halt(tmp_path):
    conn = _conn(tmp_path)
    _live_strategy(conn, "s1")
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
    _live_strategy(conn, "s1")
    _fill(conn, "a1", "s1", "AAA", 10.0)
    R.reconcile(conn, {"AAA": 12.0}, cycle=1)            # pending recorded
    R.reconcile(conn, {"AAA": 10.0}, cycle=2)            # resolves -> row cleared
    assert conn.execute("SELECT COUNT(*) FROM live_reconcile_state").fetchone()[0] == 0


def test_reconcile_clears_pending_when_symbol_goes_flat_on_both(tmp_path):
    # a mismatch that resolves to flat on BOTH sides (symbol absent from expected and broker) must
    # still clear its pending row, else a later mismatch reads a stale first_seen_cycle (codex)
    conn = _conn(tmp_path)
    _live_strategy(conn, "s1")
    _fill(conn, "a1", "s1", "AAA", 10.0)
    R.reconcile(conn, {"AAA": 12.0}, cycle=1)            # pending recorded for AAA
    # now AAA is flat on both: remove the books' fills and the broker shows nothing
    conn.execute("DELETE FROM live_fills")
    conn.commit()
    res = R.reconcile(conn, {}, cycle=2)                 # AAA absent from expected AND broker
    assert res.clean
    assert conn.execute("SELECT COUNT(*) FROM live_reconcile_state").fetchone()[0] == 0


def test_reconcile_fails_closed_on_demoted_strategy_orphan_holding(tmp_path):
    # #451: a strategy that left live (dormant/retired) whose broker position was never flattened
    # must NOT be explained away. Its fill is no longer attributed to a currently-live strategy, so
    # attributed_live_net drops it and the un-flattened broker holding becomes an unexplained
    # residual that fails closed and escalates to halt.
    conn = _conn(tmp_path)
    _fill(conn, "a1", "demoted", "AAA", 10.0)  # 'demoted' is NOT registered as live
    r1 = R.reconcile(conn, {"AAA": 10.0}, cycle=1)
    assert not r1.clean and r1.mismatches[0]["symbol"] == "AAA"
    r2 = R.reconcile(conn, {"AAA": 10.0}, cycle=2)
    assert not r2.halt  # still within grace
    r3 = R.reconcile(conn, {"AAA": 10.0}, cycle=4)  # cycle - first_seen (1) >= 3 -> unexplained
    assert r3.halt and r3.mismatches[0]["status"] == "unexplained"
