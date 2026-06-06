from algua.execution.live_ledger import position_pnl


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
