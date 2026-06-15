import pandas as pd
import pytest

from algua.backtest.factor_eval import build_factor_strategy, forward_returns, score_panel
from algua.features.catalogue import get_factor


def _bars():
    rows = []
    for i, ts in enumerate(pd.date_range("2023-01-01", periods=10, freq="D")):
        rows.append((ts, "AAA", 100.0 + i))
        rows.append((ts, "BBB", 100.0 + 2.0 * i))
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "adj_close"]).set_index("timestamp")


def _strategy():
    return build_factor_strategy(
        get_factor("xs_trailing_return"), symbols=["AAA", "BBB"],
        params={"lookback": 2}, construction="equal_weight_positive", construction_params={},
    )


def test_score_panel_is_pit_no_look_ahead():
    bars = _bars()
    full = score_panel(_strategy(), bars)
    truncated = score_panel(_strategy(), bars.loc[:full.index[5]])
    # the score at an early bar must not change when later bars are added
    pd.testing.assert_series_equal(full.loc[truncated.index[3]], truncated.loc[truncated.index[3]])


def test_forward_returns_offset_by_lag_and_horizon():
    bars = _bars()
    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    fwd = forward_returns(adj, lag=1, horizon=1)
    t0 = adj.index[0]
    # entry at t0+lag=index[1], exit at index[2]: (102/101 - 1)
    assert fwd.loc[t0, "AAA"] == pytest.approx(adj["AAA"].iloc[2] / adj["AAA"].iloc[1] - 1)
    # the last (lag+horizon) rows have no future bar -> NaN
    assert fwd.iloc[-1].isna().all()
