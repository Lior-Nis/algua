"""Regression + behaviour tests for SyntheticProvider's additive drift/vol params (#347)."""
from __future__ import annotations

from datetime import UTC, datetime

from algua.backtest._sample import SyntheticProvider

START = datetime(2020, 1, 1, tzinfo=UTC)
END = datetime(2020, 6, 30, tzinfo=UTC)


def test_default_output_byte_identical_to_explicit_legacy_constants():
    """The new params default to the original constants, so seed=0 output is unchanged."""
    default = SyntheticProvider(seed=0).get_bars(["AAA", "BBB"], START, END, "1d")
    explicit = SyntheticProvider(seed=0, drift=0.0005, vol=0.02).get_bars(
        ["AAA", "BBB"], START, END, "1d"
    )
    assert default.equals(explicit)


def test_higher_drift_lifts_mean_log_return():
    import numpy as np

    low = SyntheticProvider(seed=1, drift=0.0, vol=0.01).get_bars(["AAA"], START, END, "1d")
    high = SyntheticProvider(seed=1, drift=0.005, vol=0.01).get_bars(["AAA"], START, END, "1d")
    # Same seed + vol → the only difference is the drift; the high-drift path ends much higher.
    assert high["close"].iloc[-1] > low["close"].iloc[-1]
    low_ret = np.diff(np.log(low["close"].to_numpy())).mean()
    high_ret = np.diff(np.log(high["close"].to_numpy())).mean()
    assert high_ret > low_ret
