from algua.execution.live_ledger import LedgerKind, position_pnl, strategy_cash_credit
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "s.db")
    migrate(conn)
    return conn


def _fill(conn, aid, strategy, symbol, qty, price):
    conn.execute(
        "INSERT INTO live_fills"
        "(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (aid, "b", strategy, symbol, qty, price, "2026-06-06T00:00:00+00:00"),
    )
    conn.commit()


def _activity(conn, aid, type_, symbol, amount):
    conn.execute(
        "INSERT INTO live_activities(activity_id, type, symbol, amount, ts, raw) "
        "VALUES (?,?,?,?,?,?)",
        (aid, type_, symbol, amount, "2026-06-06T00:00:00+00:00", "{}"),
    )
    conn.commit()


def test_cash_credit_single_holder_gets_full_amount(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "f1", "s1", "AAA", 10.0, 100.0)          # only s1 traded AAA
    _activity(conn, "d1", "DIV", "AAA", 25.0)
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 25.0


def test_cash_credit_splits_pro_rata_by_gross_volume(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "f1", "s1", "AAA", 10.0, 100.0)          # s1 gross vol 10
    _fill(conn, "f2", "s2", "AAA", -30.0, 100.0)         # s2 gross vol 30 (abs)
    _activity(conn, "d1", "DIV", "AAA", 40.0)            # 40 split 10:30
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 10.0
    assert strategy_cash_credit(conn, "s2", LedgerKind.LIVE) == 30.0


def test_cash_credit_excludes_symbolless_and_untraded(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "f1", "s1", "AAA", 10.0, 100.0)          # s1 only traded AAA
    _activity(conn, "i1", "INT", None, 5.0)              # symbol-less account cash -> excluded
    _activity(conn, "d1", "DIV", "BBB", 7.0)             # symbol s1 never traded -> excluded
    assert strategy_cash_credit(conn, "s1", LedgerKind.LIVE) == 0.0


def test_long_then_partial_close_realizes():
    # buy 10@100, buy 10@110 -> avg 105, qty 20; sell 5@120 -> realized 5*(120-105)=75
    fills = [(10.0, 100.0), (10.0, 110.0), (-5.0, 120.0)]
    r = position_pnl(fills, mark=120.0)
    assert r.qty == 15.0
    assert r.avg_cost == 105.0
    assert r.realized == 75.0
    assert r.unrealized == 15.0 * (120.0 - 105.0)


def test_flip_long_to_short():
    # buy 10@100; sell 15@120 -> close 10 (realize 10*(120-100)=200), open short 5@120
    fills = [(10.0, 100.0), (-15.0, 120.0)]
    r = position_pnl(fills, mark=130.0)
    assert r.qty == -5.0
    assert r.avg_cost == 120.0
    assert r.realized == 200.0
    # short unrealized = (avg-mark)*|qty| = (120-130)*5 = -50; (mark-avg)*qty == (130-120)*-5 == -50
    assert r.unrealized == -50.0


def test_short_then_cover():
    # sell 10@100 (short); buy 4@90 -> realize 4*(100-90)=40 covering short
    fills = [(-10.0, 100.0), (4.0, 90.0)]
    r = position_pnl(fills, mark=95.0)
    assert r.qty == -6.0
    assert r.avg_cost == 100.0
    assert r.realized == 40.0


def test_flat_is_zero():
    r = position_pnl([], mark=100.0)
    assert r.qty == 0.0 and r.realized == 0.0 and r.unrealized == 0.0
