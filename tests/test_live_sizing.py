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
