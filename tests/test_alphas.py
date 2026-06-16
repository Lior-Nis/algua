import pandas as pd
import pytest

from algua.features.alphas import xs_trailing_return
from algua.features.catalogue import load_all_factors


def _view(rows):
    # long bar-schema frame: timestamp index + symbol/adj_close columns
    df = pd.DataFrame(rows, columns=["timestamp", "symbol", "adj_close"])
    return df.set_index("timestamp")


def test_xs_trailing_return_is_trailing_return_per_symbol():
    view = _view([
        ("2023-01-01", "AAA", 100.0), ("2023-01-01", "BBB", 100.0),
        ("2023-01-02", "AAA", 110.0), ("2023-01-02", "BBB", 90.0),
    ])
    scores = xs_trailing_return(view, {"lookback": 1})
    assert scores["AAA"] == pytest.approx(0.10)
    assert scores["BBB"] == pytest.approx(-0.10)


def test_xs_trailing_return_empty_before_enough_history():
    view = _view([("2023-01-01", "AAA", 100.0)])
    assert xs_trailing_return(view, {"lookback": 5}).empty


def test_xs_trailing_return_is_catalogued_standalone():
    specs = load_all_factors()
    assert "xs_trailing_return" in specs
    assert specs["xs_trailing_return"].standalone is True
