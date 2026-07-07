"""walk_forward(compute_holdout=False) — the PBO/CSCV window-only path (#467, task 1).

On the window-only path the holdout STATISTIC is never computed: window_metrics is BIT-IDENTICAL
to the compute_holdout=True path (identical _segment_bounds), but holdout_metrics == {},
holdout_returns is None, market_returns is None, and on_peek is NEVER called (no single-use burn).
"""

from algua.backtest._sample import SyntheticProvider
from algua.backtest.walkforward import walk_forward
from tests.test_sweep import END, START, _momentum


def _wf(compute_holdout, on_peek=None):
    return walk_forward(
        _momentum(), SyntheticProvider(seed=3), START, END,
        windows=4, holdout_frac=0.2, compute_holdout=compute_holdout, on_peek=on_peek,
    )


def test_window_metrics_bit_identical_to_holdout_path():
    full = _wf(True)
    window_only = _wf(False)
    assert window_only.window_metrics == full.window_metrics
    # stability derives from window_metrics, so it is identical too.
    assert window_only.stability == full.stability


def test_holdout_statistic_and_series_elided():
    r = _wf(False)
    assert r.holdout_metrics == {}
    assert r.holdout_returns is None
    # The benchmark series (a regime-gate input) is skipped on the window-only path.
    assert r.market_returns is None


def test_on_peek_never_called_on_window_only_path():
    calls = []
    _wf(False, on_peek=lambda cfg: calls.append(cfg))
    assert calls == []  # no single-use holdout burn


def test_on_peek_still_fires_on_default_path():
    calls = []
    _wf(True, on_peek=lambda cfg: calls.append(cfg))
    assert len(calls) == 1  # default behavior preserved


def test_default_path_still_computes_holdout():
    r = _wf(True)
    assert r.holdout_metrics != {}
    assert r.holdout_returns is not None
    assert r.market_returns is not None
