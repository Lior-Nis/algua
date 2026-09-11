"""Universe-derived regime features (overlays spec §Features). Pure, adj_close-only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algua.features.regime import (
    equal_weight_index,
    robust_zscore,
    rolling_drawdown,
    turbulence,
    wide_adj_close,
)


def _view(prices: dict[str, list[float]]) -> pd.DataFrame:
    """prices = {symbol: [adj_close per bar]} -> long bar-schema view, all symbols same length."""
    n = len(next(iter(prices.values())))
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    rows = []
    for sym, path in prices.items():
        for t, px in zip(ts, path, strict=True):
            rows.append({"timestamp": t, "symbol": sym, "open": px, "high": px, "low": px,
                         "close": px * 2.0, "adj_close": px, "volume": 1.0})
    return pd.DataFrame(rows).set_index("timestamp").sort_index()


def test_wide_adj_close_pivots_and_sorts():
    view = _view({"B": [1.0, 2.0], "A": [3.0, 4.0]})
    wide = wide_adj_close(view.iloc[::-1])  # reversed input must come back sorted
    assert list(wide.columns) == ["A", "B"]
    assert wide.index.is_monotonic_increasing
    assert wide["A"].tolist() == [3.0, 4.0]


def test_equal_weight_index_is_mean_of_member_returns_base_one():
    view = _view({"A": [100.0, 110.0, 121.0], "B": [100.0, 90.0, 99.0]})
    level = equal_weight_index(view)
    # bar 0 -> 1.0; bar 1 -> mean(+10%, -10%) = 0 -> 1.0; bar 2 -> mean(+10%, +10%) = +10% -> 1.1
    assert level.iloc[0] == pytest.approx(1.0)
    assert level.iloc[1] == pytest.approx(1.0)
    assert level.iloc[2] == pytest.approx(1.1)


def test_equal_weight_index_ignores_a_member_missing_a_bar():
    view = _view({"A": [100.0, 110.0], "B": [100.0, 120.0]})
    view = view[~((view["symbol"] == "B") & (view.index == view.index[1]))]  # drop B's 2nd bar
    level = equal_weight_index(view)
    assert level.iloc[1] == pytest.approx(1.10)  # only A contributes


def test_equal_weight_index_uses_adj_close_not_close():
    view = _view({"A": [100.0, 110.0]})
    assert equal_weight_index(view).iloc[1] == pytest.approx(1.10)  # close is 2x and unused


def test_rolling_drawdown_nan_until_window_then_from_rolling_high():
    level = pd.Series([1.0, 1.2, 1.1, 0.9, 1.0])
    dd = rolling_drawdown(level, window=3)
    assert np.isnan(dd.iloc[0]) and np.isnan(dd.iloc[1])
    assert dd.iloc[2] == pytest.approx(1.1 / 1.2 - 1.0)
    assert dd.iloc[3] == pytest.approx(0.9 / 1.2 - 1.0)
    assert dd.iloc[4] == pytest.approx(1.0 / 1.1 - 1.0)  # 1.2 has rolled out of the window


def test_robust_zscore_nan_until_full_and_nan_on_zero_mad():
    z = robust_zscore(pd.Series([1.0] * 10), window=5)
    assert z.isna().all()  # constant -> zero MAD -> NaN, never inf
    x = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 50.0])
    z = robust_zscore(x, window=5)
    assert z.iloc[:4].isna().all()
    assert z.iloc[-1] > 10.0  # the planted outlier


def test_turbulence_nan_until_window_and_spikes_on_planted_outlier():
    rng = np.random.default_rng(0)
    n, window = 40, 10
    rets = rng.normal(0.0, 0.01, size=(n, 3))
    rets[-1] = 0.10  # every symbol jumps +10 sigma on the last bar
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    view = _view({s: prices[:, i].tolist() for i, s in enumerate(["A", "B", "C"])})
    t = turbulence(view, window)
    assert t.iloc[: window + 1].isna().all()  # first row has no return + `window` prior returns
    assert t.iloc[window + 1 :].notna().all()
    assert t.iloc[-1] > 10.0 * t.iloc[window + 1 : -1].median()


def test_turbulence_last_computes_only_the_tail_identically():
    rng = np.random.default_rng(1)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=(30, 2)), axis=0)
    view = _view({"A": prices[:, 0].tolist(), "B": prices[:, 1].tolist()})
    full = turbulence(view, 5)
    tail = turbulence(view, 5, last=4)
    assert tail.iloc[:-4].isna().all()
    pd.testing.assert_series_equal(tail.iloc[-4:], full.iloc[-4:])


def test_turbulence_excludes_symbols_without_full_window_that_bar():
    rng = np.random.default_rng(2)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=(20, 2)), axis=0)
    view = _view({"A": prices[:, 0].tolist(), "B": prices[:, 1].tolist()})
    view_a_only = view[view["symbol"] == "A"]
    view_b_only = view[view["symbol"] == "B"]
    # Drop B entirely before bar 15: at bar 19, B has only 4 prior returns -> excluded -> A-only.
    # (index into B's own per-symbol view, not the interleaved combined `view`, to land on bar 15.)
    partial = pd.concat([view_a_only, view_b_only[view_b_only.index >= view_b_only.index[15]]])
    partial = partial.sort_index()
    assert turbulence(partial, 5).iloc[-1] == pytest.approx(turbulence(view_a_only, 5).iloc[-1])
