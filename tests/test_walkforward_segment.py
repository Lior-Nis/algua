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


def test_embargo_carves_gap_from_train_side(): # noqa
    # holdout unchanged; the last `embargo` train bars are purged out of the windows (#345).
    windows, holdout = _segment_bounds(100, windows=4, holdout_frac=0.2, embargo=10)
    assert holdout == (80, 100)                 # holdout interval byte-identical to embargo=0
    assert windows[-1][1] == 70                 # usable_train = 80 - 10
    assert holdout[0] - windows[-1][1] == 10    # carved gap == embargo (Lopez de Prado purge)
    # windows cover [0, 70) contiguously with no overlap into the gap or the holdout
    covered = [i for s, e in windows for i in range(s, e)]
    assert covered == list(range(70))


def test_embargo_zero_matches_no_embargo():
    assert _segment_bounds(100, 4, 0.2, embargo=0) == _segment_bounds(100, 4, 0.2)


def test_negative_embargo_rejected():
    # A negative embargo would set usable_train > train_n, overlapping windows into the holdout.
    with pytest.raises(BacktestError):
        _segment_bounds(100, windows=4, holdout_frac=0.2, embargo=-1)


def test_embargo_too_large_leaves_too_few_bars():
    # train=80, embargo=65 -> usable=15, base=3 < _MIN_WINDOW_BARS(5) -> fail closed.
    with pytest.raises(BacktestError):
        _segment_bounds(100, windows=4, holdout_frac=0.2, embargo=65)
