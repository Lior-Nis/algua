"""Tests for #48: _coerce_values widens mixed int/float lists to float."""
from algua.backtest.sweep import _coerce_values


def test_all_ints_stay_int():
    assert _coerce_values([10, 20, 30]) == [10, 20, 30]
    assert all(type(v) is int for v in _coerce_values([10, 20, 30]))


def test_all_floats_stay_float():
    result = _coerce_values([1.0, 2.5, 3.0])
    assert all(type(v) is float for v in result)


def test_mixed_int_float_widens_to_float():
    # "10,10.5" parses to [int(10), float(10.5)] — must widen all to float
    result = _coerce_values([10, 10.5])
    assert all(type(v) is float for v in result)
    assert result == [10.0, 10.5]


def test_mixed_with_strings_untouched():
    # strings — no numeric widening applied
    result = _coerce_values(["fast", "slow"])
    assert result == ["fast", "slow"]


def test_single_int_stays_int():
    result = _coerce_values([5])
    assert result == [5]
    assert type(result[0]) is int


def test_single_float_stays_float():
    result = _coerce_values([5.0])
    assert result == [5.0]
    assert type(result[0]) is float
