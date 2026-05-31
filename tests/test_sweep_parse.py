import pytest

from algua.backtest.sweep import _parse_grid


def test_parses_ints_floats_strs():
    grid = _parse_grid(["lookback=20,40,60", "rate=0.1,0.2", "mode=fast,slow"])
    assert grid == {"lookback": [20, 40, 60], "rate": [0.1, 0.2], "mode": ["fast", "slow"]}


def test_empty_list_raises():
    with pytest.raises(ValueError):
        _parse_grid([])


def test_missing_equals_raises():
    with pytest.raises(ValueError):
        _parse_grid(["lookback"])


def test_empty_key_or_values_raises():
    with pytest.raises(ValueError):
        _parse_grid(["=1,2"])
    with pytest.raises(ValueError):
        _parse_grid(["k="])
