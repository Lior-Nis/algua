from datetime import date

import numpy as np
import pandas as pd
import pytest

from algua.backtest.delisting import (
    DelistingExitError,
    DelistingRecord,
    apply_delisting_exits,
)


def _grid(cols):  # cols: dict[str, list]; 4 daily bars
    idx = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    return pd.DataFrame(cols, index=idx)


def test_record_construction_rejects_nonpositive_price():
    with pytest.raises(ValueError, match="terminal_price"):
        DelistingRecord(date(2020, 1, 2), 0.0, "test")
    with pytest.raises(ValueError, match="terminal_price"):
        DelistingRecord(date(2020, 1, 2), float("nan"), "test")


def test_held_with_record_forces_exit_at_terminal_price():
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, np.nan, np.nan]})
    w = _grid({"A": [0.5, 0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.0, 0.0]})
    recs = {"B": [DelistingRecord(date(2020, 1, 2), 5.0, "vendor")]}
    adj_x, w_x, forced = apply_delisting_exits(adj, w, recs)
    T = adj.index[1]
    assert w_x.loc[T, "B"] == 0.0 and (w_x.loc[T:, "B"] == 0.0).all()
    assert adj_x.loc[T, "B"] == 5.0
    assert (adj_x.loc[adj_x.index > T, "B"] == 5.0).all()
    assert forced == [{"symbol": "B", "bar": T.isoformat(),
                       "terminal_price": 5.0, "source": "vendor"}]
    assert (w_x["A"] == w["A"]).all() and adj_x["A"].equals(adj["A"])


def test_held_without_record_fails_closed():
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, np.nan, np.nan]})
    w = _grid({"A": [0.5, 0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.0, 0.0]})
    with pytest.raises(DelistingExitError, match="no delisting record"):
        apply_delisting_exits(adj, w, None)


def test_held_without_record_relaxation_realizes_last_close():
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, np.nan, np.nan]})
    w = _grid({"A": [0.5, 0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.0, 0.0]})
    adj_x, w_x, forced = apply_delisting_exits(adj, w, None, assume_terminal_last_close=True)
    T = adj.index[1]
    assert (w_x.loc[T:, "B"] == 0.0).all()
    assert forced == [{"symbol": "B", "bar": T.isoformat(),
                       "terminal_price": 11.0, "source": "assumed_last_close"}]


def test_not_held_only_nan_killed_no_error():
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, np.nan, np.nan, np.nan]})
    w = _grid({"A": [1.0, 1.0, 1.0, 1.0], "B": [0.0, 0.0, 0.0, 0.0]})
    adj_x, w_x, forced = apply_delisting_exits(adj, w, None)
    assert forced == []
    T = adj.index[0]
    assert (adj_x.loc[adj_x.index > T, "B"] == 10.0).all()


def test_integrity_bars_after_resolved_delisting_fails_closed():
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, 12, 13]})
    w = _grid({"A": [0.5] * 4, "B": [0.5] * 4})
    recs = {"B": [DelistingRecord(date(2020, 1, 2), 5.0, "vendor")]}
    with pytest.raises(DelistingExitError, match="after stated delisting"):
        apply_delisting_exits(adj, w, recs)


def test_period_ends_on_delisting_applies_terminal_price():
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, 12, 20]})
    w = _grid({"A": [0.5] * 4, "B": [0.5] * 4})
    recs = {"B": [DelistingRecord(date(2020, 1, 4), 5.0, "vendor")]}
    adj_x, w_x, forced = apply_delisting_exits(adj, w, recs)
    T = adj.index[3]
    assert adj_x.loc[T, "B"] == 5.0 and w_x.loc[T, "B"] == 0.0
    assert forced and forced[0]["bar"] == T.isoformat()


def test_record_after_panel_end_skipped():
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, np.nan, np.nan]})
    w = _grid({"A": [0.5] * 4, "B": [0.5, 0.5, 0.0, 0.0]})
    recs = {"B": [DelistingRecord(date(2021, 6, 1), 5.0, "vendor")]}
    with pytest.raises(DelistingExitError, match="no delisting record"):
        apply_delisting_exits(adj, w, recs)


def test_two_records_same_terminal_bar_ambiguous():
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, np.nan, np.nan]})
    w = _grid({"A": [0.5] * 4, "B": [0.5, 0.5, 0.0, 0.0]})
    recs = {"B": [DelistingRecord(date(2020, 1, 2), 5.0, "v1"),
                  DelistingRecord(date(2020, 1, 2), 6.0, "v2")]}
    with pytest.raises(DelistingExitError, match="ambiguous"):
        apply_delisting_exits(adj, w, recs)


def test_symbol_never_traded_skipped():
    adj = _grid({"A": [10, 11, 12, 13], "B": [np.nan, np.nan, np.nan, np.nan]})
    w = _grid({"A": [1.0] * 4, "B": [0.0] * 4})
    adj_x, w_x, forced = apply_delisting_exits(adj, w, None)
    assert forced == []


def test_vendor_date_one_day_past_last_trade_resolves_to_terminal_bar():
    # B's last TRADED bar is index 1 (Jan 2); the union panel index has a Jan 3 row only because
    # A traded then (B is NaN there). A vendor delisting_date of Jan 3 must resolve to B's own
    # terminal bar (Jan 2) and apply there — not fail closed with a misleading "no record".
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, np.nan, np.nan]})
    w = _grid({"A": [0.5] * 4, "B": [0.5, 0.5, 0.0, 0.0]})
    recs = {"B": [DelistingRecord(date(2020, 1, 3), 5.0, "vendor")]}
    adj_x, w_x, forced = apply_delisting_exits(adj, w, recs)
    T = adj.index[1]
    assert forced == [{"symbol": "B", "bar": T.isoformat(),
                       "terminal_price": 5.0, "source": "vendor"}]
    assert adj_x.loc[T, "B"] == 5.0 and (w_x.loc[T:, "B"] == 0.0).all()


def test_phantom_post_delisting_weight_is_suppressed():
    # Strategy goes flat on B at its last bar (T=index1) but targets B AGAIN after it delists
    # (index2,3). Those post-T weights must be forced to 0 (can't trade a delisted name).
    adj = _grid({"A": [10, 11, 12, 13], "B": [10, 11, np.nan, np.nan]})
    w = _grid({"A": [0.5, 0.5, 0.5, 0.5], "B": [0.0, 0.0, 0.7, 0.7]})
    adj_x, w_x, forced = apply_delisting_exits(adj, w, None)  # no record needed: not held at T
    T = adj.index[1]
    assert (w_x.loc[adj_x.index > T, "B"] == 0.0).all()   # phantom weights suppressed
    assert forced == []                                    # nothing realized (wasn't held at T)
