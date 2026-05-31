import pytest

from algua.backtest.engine import BacktestError
from algua.backtest.walkforward import _segment_bounds


def test_even_split():
    windows, holdout = _segment_bounds(100, windows=4, holdout_frac=0.2)
    assert holdout == (80, 100)
    assert windows == [(0, 20), (20, 40), (40, 60), (60, 80)]


def test_remainder_goes_to_last_window():
    windows, holdout = _segment_bounds(102, windows=4, holdout_frac=0.2)
    assert holdout == (82, 102)
    assert windows[-1] == (60, 82)
    assert windows[0] == (0, 20)


def test_full_coverage_no_overlap():
    windows, holdout = _segment_bounds(97, windows=3, holdout_frac=0.25)
    covered = []
    for s, e in windows:
        covered.extend(range(s, e))
    covered.extend(range(holdout[0], holdout[1]))
    assert covered == list(range(97))


def test_invalid_windows():
    with pytest.raises(ValueError):
        _segment_bounds(100, windows=1, holdout_frac=0.2)


def test_invalid_holdout_frac():
    with pytest.raises(ValueError):
        _segment_bounds(100, windows=4, holdout_frac=1.0)


def test_too_few_bars_raises_backtest_error():
    with pytest.raises(BacktestError):
        _segment_bounds(20, windows=4, holdout_frac=0.2)  # train=16, base=4 < _MIN_WINDOW_BARS(5)


def test_zero_bar_holdout_rejected():
    # holdout_frac so small it rounds to 0 holdout bars must be rejected
    with pytest.raises(BacktestError):
        _segment_bounds(100, windows=4, holdout_frac=0.005)  # int(100*0.005)=0
