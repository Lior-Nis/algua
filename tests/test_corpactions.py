# tests/test_corpactions.py
import numpy as np
import pandas as pd
import pytest

from algua.data.corpactions import (
    Dividend,
    Split,
    back_adjust,
    check_adj_close_consistent,
)


def _utc(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def test_split_and_dividend_construct():
    s = Split(ex_date=_utc("2024-01-03"), ratio=2.0)
    d = Dividend(ex_date=_utc("2024-01-03"), cash=1.5)
    assert s.ratio == 2.0 and d.cash == 1.5


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("inf"), float("nan")])
def test_split_ratio_must_be_finite_positive(ratio):
    with pytest.raises(ValueError, match="ratio"):
        Split(ex_date=_utc("2024-01-03"), ratio=ratio)


@pytest.mark.parametrize("cash", [0.0, -1.0, float("inf"), float("nan")])
def test_dividend_cash_must_be_finite_positive(cash):
    with pytest.raises(ValueError, match="cash"):
        Dividend(ex_date=_utc("2024-01-03"), cash=cash)


def test_tz_naive_ex_date_rejected():
    with pytest.raises(ValueError, match="tz-aware"):
        Split(ex_date=pd.Timestamp("2024-01-03"), ratio=2.0)
    with pytest.raises(ValueError, match="tz-aware"):
        Dividend(ex_date=pd.Timestamp("2024-01-03"), cash=1.0)


def _bars(closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({"ts": ts, "close": [float(c) for c in closes]})


def _factor(raw, events) -> np.ndarray:
    return back_adjust(raw, events)["adj_factor"].to_numpy()


def _adj(raw, events) -> np.ndarray:
    return back_adjust(raw, events)["adj_close"].to_numpy()


def test_no_events_is_identity():
    raw = _bars([100, 110, 120])
    out = back_adjust(raw, [])
    assert list(out.columns) == ["ts", "adj_close", "adj_factor"]
    np.testing.assert_allclose(out["adj_factor"].to_numpy(), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(out["adj_close"].to_numpy(), [100, 110, 120])


def test_forward_2for1_split():
    raw = _bars([100, 110, 50, 55])  # ex on bar index 2 (already post-split)
    ev = [Split(ex_date=_utc("2024-01-03"), ratio=2.0)]
    np.testing.assert_allclose(_factor(raw, ev), [0.5, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(_adj(raw, ev), [50, 55, 50, 55])


def test_reverse_1for10_split_scales_history_up():
    raw = _bars([5, 6, 50, 55])  # 1:10 reverse on bar index 2
    ev = [Split(ex_date=_utc("2024-01-03"), ratio=0.1)]
    np.testing.assert_allclose(_factor(raw, ev), [10.0, 10.0, 1.0, 1.0])
    np.testing.assert_allclose(_adj(raw, ev), [50, 60, 50, 55])  # adj/raw > 1 historically


def test_single_dividend_total_return():
    raw = _bars([100, 110, 120, 130])  # ex on bar index 2; P_prev = raw close[1] = 110
    ev = [Dividend(ex_date=_utc("2024-01-03"), cash=2.0)]
    m = (110 - 2) / 110
    np.testing.assert_allclose(_factor(raw, ev), [m, m, 1.0, 1.0])


def test_two_dividends_same_ex_date_sum_cash_no_cross_term():
    raw = _bars([100, 110, 120, 130])  # P_prev = 110
    ev = [
        Dividend(ex_date=_utc("2024-01-03"), cash=2.0),
        Dividend(ex_date=_utc("2024-01-03"), cash=3.0),
    ]
    correct = (110 - 5) / 110  # NOT (1 - 2/110)(1 - 3/110)
    np.testing.assert_allclose(_factor(raw, ev)[0], correct)


def test_same_ex_date_split_and_dividend():
    raw = _bars([100, 110, 50, 55])  # P_prev = raw close[1] = 110 (pre-split units)
    ev = [
        Split(ex_date=_utc("2024-01-03"), ratio=2.0),
        Dividend(ex_date=_utc("2024-01-03"), cash=2.0),
    ]
    expected = 0.5 * (110 - 2) / 110
    np.testing.assert_allclose(_factor(raw, ev)[0], expected)


def test_p_prev_is_raw_when_split_sits_between():
    # split on bar 1, dividend on bar 3; P_prev for the dividend is the RAW close[2], not adjusted.
    raw = _bars([100, 50, 55, 60, 65])
    ev = [
        Split(ex_date=_utc("2024-01-02"), ratio=2.0),
        Dividend(ex_date=_utc("2024-01-04"), cash=2.0),
    ]
    div = (55 - 2) / 55
    np.testing.assert_allclose(_factor(raw, ev), [0.5 * div, div, div, 1.0, 1.0])


def test_event_before_first_bar_is_noop():
    raw = _bars([100, 110, 120], start="2024-02-01")
    ev = [Split(ex_date=_utc("2024-01-01"), ratio=2.0)]
    np.testing.assert_allclose(_factor(raw, ev), [1.0, 1.0, 1.0])


def test_event_after_last_bar_is_noop_anchor_preserved():
    raw = _bars([100, 110, 120])
    ev = [Split(ex_date=_utc("2030-01-01"), ratio=2.0)]
    out = back_adjust(raw, ev)
    np.testing.assert_allclose(out["adj_factor"].to_numpy(), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(out["adj_close"].to_numpy(), [100, 110, 120])  # adj == close (no-op)


def test_dividend_ex_on_non_trading_day_resolves_to_prior_bar():
    # bars Mon/Tue/Thu/Fri; dividend ex on Wed (no bar) scales the two bars strictly before it.
    ts = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"], tz="UTC")
    raw = pd.DataFrame({"ts": ts, "close": [100.0, 110.0, 120.0, 130.0]})
    ev = [Dividend(ex_date=_utc("2024-01-03"), cash=2.0)]  # P_prev = close at 01-02 = 110
    m = (110 - 2) / 110
    np.testing.assert_allclose(_factor(raw, ev), [m, m, 1.0, 1.0])


def test_empty_bars_returns_empty_typed_frame():
    raw = pd.DataFrame(
        {"ts": pd.DatetimeIndex([], tz="UTC"), "close": pd.Series([], dtype="float64")}
    )
    out = back_adjust(raw, [Split(ex_date=_utc("2024-01-03"), ratio=2.0)])
    assert list(out.columns) == ["ts", "adj_close", "adj_factor"]
    assert len(out) == 0


def test_non_midnight_ex_date_raises():
    raw = _bars([100, 110, 120])
    ev = [Split(ex_date=pd.Timestamp("2024-01-02 12:00", tz="UTC"), ratio=2.0)]
    with pytest.raises(ValueError, match="midnight"):
        back_adjust(raw, ev)


def test_dividend_ge_prior_close_raises_actionable():
    raw = _bars([100, 110, 120, 130])  # P_prev = 110
    ev = [Dividend(ex_date=_utc("2024-01-03"), cash=200.0)]
    with pytest.raises(ValueError, match="liquidating"):
        back_adjust(raw, ev)


def test_raw_frame_guards():
    ev: list = []
    with pytest.raises(ValueError, match="'ts' and 'close'"):
        back_adjust(pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=2, tz="UTC")}), ev)
    # unsorted ts
    ts = pd.DatetimeIndex(["2024-01-02", "2024-01-01"], tz="UTC")
    with pytest.raises(ValueError, match="ascending"):
        back_adjust(pd.DataFrame({"ts": ts, "close": [1.0, 2.0]}), ev)
    # duplicate ts
    ts = pd.DatetimeIndex(["2024-01-01", "2024-01-01"], tz="UTC")
    with pytest.raises(ValueError, match="ascending|unique"):
        back_adjust(pd.DataFrame({"ts": ts, "close": [1.0, 2.0]}), ev)
    # nonpositive / inf close
    with pytest.raises(ValueError, match="finite"):
        back_adjust(_bars([100, 0, 120]), ev)
    with pytest.raises(ValueError, match="finite"):
        back_adjust(_bars([100, float("inf"), 120]), ev)
    # tz-naive ts
    naive = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=2), "close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="tz-aware"):
        back_adjust(naive, ev)


def _series(closes: list[float], start: str = "2024-01-01") -> pd.Series:
    ts = pd.date_range(start, periods=len(closes), freq="D", tz="UTC")
    return pd.Series([float(c) for c in closes], index=ts)


def _vendor_adj(closes: list[float], events) -> pd.Series:
    raw = _bars(closes)
    out = back_adjust(raw, events)
    return pd.Series(out["adj_close"].to_numpy(), index=pd.DatetimeIndex(out["ts"]))


def test_validator_accepts_consistent_series():
    closes = [100, 110, 120, 130]
    ev = [Dividend(ex_date=_utc("2024-01-03"), cash=2.0)]
    check_adj_close_consistent(_series(closes), _vendor_adj(closes, ev), ev)  # no raise


def test_validator_accepts_reverse_split():
    closes = [5, 6, 50, 55]
    ev = [Split(ex_date=_utc("2024-01-03"), ratio=0.1)]
    check_adj_close_consistent(_series(closes), _vendor_adj(closes, ev), ev)  # no raise


def test_validator_rejects_globally_mis_scaled_cents_vs_dollars():
    closes = [100, 110, 120, 130]
    ev = [Dividend(ex_date=_utc("2024-01-03"), cash=2.0)]
    vendor_cents = _vendor_adj(closes, ev) * 100.0  # adj in cents, raw in dollars
    with pytest.raises(ValueError, match="anchor|mis-scaled|cents"):
        check_adj_close_consistent(_series(closes), vendor_cents, ev)


def test_validator_rejects_torn_shifted_series():
    closes = [100, 110, 120, 130]
    ev = [Split(ex_date=_utc("2024-01-03"), ratio=2.0)]
    good = _vendor_adj(closes, ev)
    shifted = pd.Series(np.roll(good.to_numpy(), 1), index=good.index)
    with pytest.raises(ValueError):
        check_adj_close_consistent(_series(closes), shifted, ev)


def test_validator_rejects_wrong_magnitude_split():
    closes = [100, 110, 60, 66]
    vendor = _vendor_adj(closes, [Split(ex_date=_utc("2024-01-03"), ratio=3.0)])
    claimed = [Split(ex_date=_utc("2024-01-03"), ratio=2.0)]
    with pytest.raises(ValueError):
        check_adj_close_consistent(_series(closes), vendor, claimed)


def test_validator_low_price_small_dividend_within_tolerance_passes():
    closes = [5.00, 5.10, 5.20, 5.30]
    ev = [Dividend(ex_date=_utc("2024-01-03"), cash=0.03)]
    vendor = _vendor_adj(closes, ev).round(2)  # vendor rounds adj to the cent
    check_adj_close_consistent(_series(closes), vendor, ev)  # no false-reject


def test_validator_input_guards():
    closes = [100, 110, 120]
    ev: list = []
    good = _series(closes)
    # mismatched index
    other = _series(closes, start="2025-01-01")
    with pytest.raises(ValueError, match="same index"):
        check_adj_close_consistent(good, other, ev)
    # NaN / inf values
    nan_series = _series([100, float("nan"), 120])
    with pytest.raises(ValueError, match="finite"):
        check_adj_close_consistent(good, nan_series, ev)
    inf_series = _series([100, float("inf"), 120])
    with pytest.raises(ValueError, match="finite"):
        check_adj_close_consistent(inf_series, good, ev)
    # tz-naive index
    naive = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="tz-aware"):
        check_adj_close_consistent(naive, naive, ev)


def test_back_adjust_rejects_non_utc_ts():
    ts = pd.date_range("2024-01-01", periods=3, freq="D", tz="US/Eastern")
    raw = pd.DataFrame({"ts": ts, "close": [100.0, 110.0, 120.0]})
    with pytest.raises(ValueError, match="UTC"):
        back_adjust(raw, [])


def test_validator_rejects_non_utc_index():
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="US/Eastern")
    s = pd.Series([100.0, 110.0, 120.0], index=idx)
    with pytest.raises(ValueError, match="UTC"):
        check_adj_close_consistent(s, s, [])


def test_validator_accepts_hand_built_split_series():
    # Independent ground truth (NOT produced by back_adjust): a 2:1 split on bar index 2 means
    # pre-split bars are expressed at half price; post-split bars are unchanged.
    closes = [100, 110, 50, 55]
    ev = [Split(ex_date=_utc("2024-01-03"), ratio=2.0)]
    raw = _series(closes)
    vendor = pd.Series([50.0, 55.0, 50.0, 55.0], index=raw.index)
    check_adj_close_consistent(raw, vendor, ev)  # no raise
