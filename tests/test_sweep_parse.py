import pytest

from algua.backtest.sweep import parse_grid


def test_parses_ints_floats_strs():
    grid = parse_grid(["lookback=20,40,60", "rate=0.1,0.2", "mode=fast,slow"])
    assert grid == {"lookback": [20, 40, 60], "rate": [0.1, 0.2], "mode": ["fast", "slow"]}


def test_empty_list_raises():
    with pytest.raises(ValueError):
        parse_grid([])


def test_missing_equals_raises():
    with pytest.raises(ValueError):
        parse_grid(["lookback"])


def test_empty_key_or_values_raises():
    with pytest.raises(ValueError):
        parse_grid(["=1,2"])
    with pytest.raises(ValueError):
        parse_grid(["k="])


def test_duplicate_key_raises():
    with pytest.raises(ValueError):
        parse_grid(["lookback=20,40", "lookback=60"])


@pytest.mark.parametrize("bad", ["inf", "-inf", "nan", "1e400"])
def test_non_finite_grid_value_rejected(bad):
    """A non-finite numeric grid value fails fast with a clear message at parse time,
    rather than coercing through to an opaque config_hash JSON-serialization error (#258)."""
    with pytest.raises(ValueError, match="non-finite grid value"):
        parse_grid([f"lookback=20,{bad}"])
