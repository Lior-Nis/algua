import pytest
from algua.backtest.engine import BacktestError
from algua.backtest.sweep import _combos


def test_cartesian_product():
    combos = _combos({"a": [1, 2], "b": [3, 4, 5]})
    assert len(combos) == 6
    assert {"a": 1, "b": 3} in combos
    assert {"a": 2, "b": 5} in combos


def test_single_param():
    assert _combos({"a": [1, 2, 3]}) == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_too_many_combos_raises():
    with pytest.raises(BacktestError):
        _combos({"a": list(range(15)), "b": list(range(15))})  # 225 > 200
