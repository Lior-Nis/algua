import pandas as pd
import pytest

from algua.execution import live_sizing as S
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "s.db")
    migrate(conn)
    return conn


def _paper_fill(conn, aid, strategy, symbol, qty, price):
    conn.execute(
        "INSERT INTO paper_venue_fills"
        "(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (aid, "b", strategy, symbol, qty, price, "2026-06-06T00:00:00+00:00"),
    )
    conn.commit()


def _live_fill(conn, aid, strategy, symbol, qty, price):
    conn.execute(
        "INSERT INTO live_fills"
        "(activity_id, broker_order_id, strategy, symbol, qty, price, fill_ts)"
        " VALUES (?,?,?,?,?,?,?)",
        (aid, "b", strategy, symbol, qty, price, "2026-06-06T00:00:00+00:00"),
    )
    conn.commit()


def _bars(close_by_symbol):
    rows = []
    for sym, closes in close_by_symbol.items():
        for i, c in enumerate(closes):
            rows.append({"timestamp": pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(days=i),
                         "symbol": sym, "open": c, "high": c, "low": c, "close": c, "volume": 1.0})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def test_paper_nav_and_equity_above_allocation(tmp_path):
    conn = _conn(tmp_path)
    _paper_fill(conn, "a1", "s1", "AAA", 10.0, 100.0)        # long 10 @100
    bars = _bars({"AAA": [100.0, 110.0]})                    # mark 110 -> +100 -> NAV 10_100
    snap, nav = S.build_paper_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                              universe=["AAA"])
    assert nav == 10_100.0
    assert snap.equity == 10_000.0                           # min(allocation, NAV)
    assert snap.qtys["AAA"] == 10.0
    assert snap.market_values["AAA"] == 10.0 * 110.0


def test_paper_equity_derisks_below_allocation(tmp_path):
    conn = _conn(tmp_path)
    _paper_fill(conn, "a1", "s1", "AAA", 10.0, 100.0)
    bars = _bars({"AAA": [100.0, 50.0]})                     # mark 50 -> -500 -> NAV 9_500
    snap, nav = S.build_paper_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                              universe=["AAA"])
    assert nav == 9_500.0
    assert snap.equity == 9_500.0


def test_paper_held_symbol_missing_mark_fails_closed(tmp_path):
    conn = _conn(tmp_path)
    _paper_fill(conn, "a1", "s1", "ZZZ", 5.0, 10.0)          # held ZZZ, no bar -> fail closed
    bars = _bars({"AAA": [100.0]})
    with pytest.raises(S.LiveSizingError, match="mark"):
        S.build_paper_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars, universe=["AAA"])


def test_paper_nav_collapse_returns_non_positive_equity_snapshot(tmp_path):
    # Shared build_sizing_snapshot no longer raises on a NAV wipe (#452): it RETURNS the
    # non-positive-equity snapshot so run_tick's uniform `not (snap.equity > 0.0)` guard fires
    # RiskBreach('non_positive_equity') on the paper lane too — trip + flatten, not a silent skip.
    conn = _conn(tmp_path)
    _paper_fill(conn, "a1", "s1", "AAA", 10.0, 100.0)
    bars = _bars({"AAA": [100.0, 1.0]})                      # NAV 100 - 990 < 0
    snap, nav = S.build_paper_sizing_snapshot(conn, "s1", allocation=100.0, bars=bars,
                                              universe=["AAA"])
    assert nav == 100.0 + 10.0 * (1.0 - 100.0)              # -890
    assert snap.equity == nav
    assert not (snap.equity > 0.0)


def test_live_paper_parity_identical_fills(tmp_path):
    # Identical fills in each lane's own table must yield identical snapshot + NAV.
    conn = _conn(tmp_path)
    _live_fill(conn, "L1", "s1", "AAA", 10.0, 100.0)
    _paper_fill(conn, "P1", "s1", "AAA", 10.0, 100.0)
    bars = _bars({"AAA": [100.0, 110.0]})
    live_snap, live_nav = S.build_live_sizing_snapshot(
        conn, "s1", allocation=10_000.0, bars=bars, universe=["AAA"]
    )
    paper_snap, paper_nav = S.build_paper_sizing_snapshot(
        conn, "s1", allocation=10_000.0, bars=bars, universe=["AAA"]
    )
    assert live_nav == paper_nav
    assert live_snap == paper_snap
