from datetime import UTC, datetime

import pandas as pd
import pytest

from algua.contracts.types import OrderIntent, Side
from algua.execution import alpaca_broker as ab
from algua.execution import live_sizing as S
from algua.execution.alpaca_broker import AlpacaPaperBroker
from algua.registry.db import connect, migrate


def _conn(tmp_path):
    conn = connect(tmp_path / "s.db")
    migrate(conn)
    return conn


class _Resp:
    def __init__(self, code, payload):
        self.status_code = code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


class _RecordingRequests:
    """Records POST bodies; returns a 201 with an order id (no GETs — sizing is off the snapshot)."""

    def __init__(self):
        self.posted = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.posted.append(json)
        return _Resp(201, {"id": "o1"})


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


def test_derealized_equity_flows_into_order_sizing_downstream(tmp_path, monkeypatch):
    # #166 gap 4: the derealized sizing equity (min(allocation, NAV)) must actually reach order
    # SIZING downstream, not just appear on the snapshot. allocation=10k but a loss-making held
    # position derisks NAV to 8k -> a fresh buy must be sized off 8k, not the 10k allocation.
    conn = _conn(tmp_path)
    # Hold 100 BBB @100 (cost 10k), now marked at 80 -> unrealized -2000 -> NAV 8000.
    _fill(conn, "a1", "s1", "BBB", 100.0, 100.0)
    bars = _bars({"AAA": [100.0], "BBB": [80.0]})
    snap, nav = S.build_live_sizing_snapshot(conn, "s1", allocation=10_000.0, bars=bars,
                                             universe=["AAA", "BBB"])
    assert nav == 8_000.0 and snap.equity == 8_000.0   # snapshot is derealized

    fake = _RecordingRequests()
    monkeypatch.setattr(ab, "requests", fake)
    broker = AlpacaPaperBroker(api_key="k", api_secret="s")
    # Size a FRESH AAA buy (flat) to target weight 0.5 off the derealized snapshot.
    oid = broker.submit_sized(OrderIntent("AAA", Side.BUY, 0.5, datetime(2026, 6, 1, tzinfo=UTC)),
                              snap)
    assert oid == "o1"
    # 0.5 * 8000 (derealized) = 4000.00 — NOT 0.5 * 10000 allocation = 5000.00.
    assert fake.posted[0]["notional"] == "4000.00"


def test_nav_collapse_fails_closed(tmp_path):
    # NAV driven negative by a large unrealized loss against a small allocation -> non-positive
    # sizing denominator must fail closed, not ZeroDivision/invert weights in run_tick (codex)
    conn = _conn(tmp_path)
    _fill(conn, "a1", "s1", "AAA", 10.0, 100.0)        # long 10 @100
    bars = _bars({"AAA": [100.0, 1.0]})                # mark 1 -> unrealized -990 -> NAV 100-990<0
    with pytest.raises(S.LiveSizingError, match="non-positive"):
        S.build_live_sizing_snapshot(conn, "s1", allocation=100.0, bars=bars, universe=["AAA"])
