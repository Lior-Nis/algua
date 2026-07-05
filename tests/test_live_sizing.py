import pandas as pd
import pytest

from algua.execution import live_sizing as S
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


def _paper_fill(conn, aid, strategy, symbol, qty, price):
    conn.execute(
        "INSERT INTO paper_venue_fills"
        "(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (aid, "b", strategy, symbol, qty, price, "2026-06-06T00:00:00+00:00"),
    )
    conn.commit()


def _paper_activity(conn, aid, type_, symbol, amount):
    conn.execute(
        "INSERT INTO paper_venue_activities(activity_id, type, symbol, amount, ts, raw) "
        "VALUES (?,?,?,?,?,?)",
        (aid, type_, symbol, amount, "2026-06-06T00:00:00+00:00", "{}"),
    )
    conn.commit()


def _bars(close_by_symbol):
    rows = []
    for sym, closes in close_by_symbol.items():
        for i, c in enumerate(closes):
            rows.append({"timestamp": pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(days=i),
                         "symbol": sym, "open": c, "high": c, "low": c, "close": c, "volume": 1.0})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def test_sizing_equity_is_allocation_when_nav_above(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0, 100.0)         # long 10 @100
    bars = _bars({"AAA": [100.0, 110.0]})                # mark 110 -> unrealized +100 -> NAV 10_100
    snap, nav = S.build_live_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                             universe=["AAA"])
    assert nav == 10_100.0
    assert snap.equity == 10_000.0                       # min(allocation, NAV) = allocation
    assert snap.qtys["AAA"] == 10.0
    assert snap.market_values["AAA"] == 10.0 * 110.0


def test_sizing_equity_derisks_when_nav_below_allocation(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0, 100.0)
    bars = _bars({"AAA": [100.0, 50.0]})                 # mark 50 -> unrealized -500 -> NAV 9_500
    snap, nav = S.build_live_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                             universe=["AAA"])
    assert nav == 9_500.0
    assert snap.equity == 9_500.0                        # min(allocation, NAV) = NAV (de-risked)


def test_dividend_cash_credits_nav(tmp_path):
    # #437: a broker-paid dividend must credit the strategy's virtual NAV, matching the total-return
    # adj_close the backtest reinvests on — otherwise NAV understates and the drawdown breaker trips
    # on a phantom drawdown.
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0, 100.0)          # long 10 @100
    _activity(conn, "d1", "DIV", "AAA", 200.0)           # $200 dividend on AAA

    # mark AT cost -> unrealized 0 -> NAV = allocation + dividend; equity caps at allocation.
    bars = _bars({"AAA": [100.0, 100.0]})
    snap, nav = S.build_live_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                             universe=["AAA"])
    assert nav == 10_000.0 + 200.0
    assert snap.equity == 10_000.0                        # min(allocation, NAV) = allocation

    # mark BELOW cost -> NAV = allocation + unrealized_loss + dividend; the dividend cushions the
    # drawdown basis. Unrealized = 10*(90-100) = -100, so nav = 10_000 - 100 + 200 = 10_100 (>
    # allocation still, so equity stays at allocation) — but crucially nav is 200 higher than the
    # 9_900 it would be WITHOUT the credit.
    bars = _bars({"AAA": [100.0, 90.0]})
    snap, nav = S.build_live_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                             universe=["AAA"])
    nav_without_dividend = 10_000.0 + 10.0 * (90.0 - 100.0)   # 9_900
    assert nav == nav_without_dividend + 200.0               # 10_100 — dividend cushions the DD
    assert nav > nav_without_dividend
    assert snap.equity == min(10_000.0, nav)                 # 10_000


def test_paper_lane_dividend_cash_credits_nav(tmp_path):
    # #437 (GATE-2 finding #3): the dividend credit is kind-parameterized, but only the LIVE lane
    # was exercised. This proves the PAPER lane too — build_paper_sizing_snapshot must read the
    # paper_venue_* tables and credit the paper strategy's virtual NAV identically, so a paper
    # book's drawdown breaker doesn't trip on the phantom dividend drawdown the live lane guards.
    conn = _conn(tmp_path)
    _paper_fill(conn, "a1", "s1", "AAA", 10.0, 100.0)     # long 10 @100 in the PAPER venue
    _paper_activity(conn, "d1", "DIV", "AAA", 200.0)      # $200 paper-venue dividend on AAA
    bars = _bars({"AAA": [100.0, 100.0]})                 # mark AT cost -> unrealized 0
    snap, nav = S.build_paper_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                              universe=["AAA"])
    assert nav == 10_000.0 + 200.0                        # NAV = allocation + paper dividend credit
    assert snap.equity == 10_000.0                        # min(allocation, NAV) = allocation

    # a LIVE-lane dividend must NOT leak into the paper NAV (separate tables): booking a live DIV
    # leaves the paper snapshot unchanged.
    _activity(conn, "d2", "DIV", "AAA", 999.0)
    snap2, nav2 = S.build_paper_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                                universe=["AAA"])
    assert nav2 == nav                                    # unchanged: live DIV is a different lane


def test_universe_symbol_with_no_position_is_flat(tmp_path):
    conn = _conn(tmp_path)
    bars = _bars({"AAA": [100.0], "BBB": [50.0]})
    snap, nav = S.build_live_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                             universe=["AAA", "BBB"])
    assert nav == 10_000.0
    assert snap.qtys == {"AAA": 0.0, "BBB": 0.0}
    assert snap.market_values == {"AAA": 0.0, "BBB": 0.0}


def test_held_symbol_missing_mark_fails_closed(tmp_path):
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "ZZZ", 5.0, 10.0)            # held ZZZ, but no bar for ZZZ
    bars = _bars({"AAA": [100.0]})
    with pytest.raises(S.LiveSizingError, match="mark"):
        S.build_live_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars, universe=["AAA"])


def test_nav_collapse_returns_non_positive_equity_snapshot(tmp_path):
    # NAV driven negative by a large unrealized loss against a small allocation. build_sizing_
    # snapshot does NOT raise here (#452): it RETURNS the non-positive-equity snapshot so run_tick's
    # single uniform guard `if not (snap.equity > 0.0): raise RiskBreach('non_positive_equity')`
    # fires on BOTH snapshot sources (ledger AND broker.snapshot) and routes a wiped book to trip +
    # FLATTEN, not a silent LiveSizingError skip. (The CLI-lane flatten is proved in test_cli_live.)
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0, 100.0)        # long 10 @100
    bars = _bars({"AAA": [100.0, 1.0]})                # mark 1 -> unrealized -990 -> NAV 100-990<0
    snap, nav = S.build_live_sizing_snapshot(conn, "s1", allocation=100.0, bars=bars,
                                             universe=["AAA"])
    assert nav == 100.0 + 10.0 * (1.0 - 100.0)         # 100 - 990 = -890
    assert snap.equity == nav                          # min(allocation, NAV) = NAV, and it is <= 0
    assert not (snap.equity > 0.0)                     # the exact predicate run_tick's guard tests
