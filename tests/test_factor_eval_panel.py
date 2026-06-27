from datetime import date

import pandas as pd
import pytest

from algua.backtest.factor_eval import (
    build_factor_strategy,
    forward_returns,
    mask_panel_to_members,
    score_panel,
)
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


def test_mask_panel_to_members_nans_non_members_pit():
    """#261: scores for symbols not in the as-of PIT universe are NaN'd per timestamp, so the IC
    cross-section matches the survivorship-clean backtest membership (no look-ahead)."""
    idx = pd.date_range("2023-01-01", periods=3, freq="D")
    panel = pd.DataFrame({"AAA": [1.0, 2.0, 3.0], "BBB": [4.0, 5.0, 6.0]}, index=idx)
    # AAA is a member throughout; BBB only joins on 2023-01-02.
    universe_by_date = {date(2023, 1, 1): ["AAA"], date(2023, 1, 2): ["AAA", "BBB"]}
    masked = mask_panel_to_members(panel, universe_by_date)
    # Day 1: only AAA is a member -> BBB is masked out; AAA kept.
    assert masked.loc[idx[0], "AAA"] == 1.0
    assert pd.isna(masked.loc[idx[0], "BBB"])
    # Day 2+: both members -> both kept.
    assert masked.loc[idx[1], "BBB"] == 5.0
    assert masked.loc[idx[2], "AAA"] == 3.0
    # the original panel is not mutated
    assert panel.loc[idx[0], "BBB"] == 4.0
