import pandas as pd

from algua.backtest.metrics import metrics_from_returns


def test_empty_returns_are_zero():
    m = metrics_from_returns(pd.Series(dtype="float64"))
    assert m == {"total_return": 0.0, "ann_return": 0.0, "ann_volatility": 0.0,
                 "sharpe": 0.0, "max_drawdown": 0.0}


def test_zero_volatility_returns_zero_sharpe():
    m = metrics_from_returns(pd.Series([0.0, 0.0, 0.0]))
    assert m["sharpe"] == 0.0
    assert m["ann_volatility"] == 0.0


def test_total_return_and_drawdown():
    # +10%, -50%, then flat: total = 1.1 * 0.5 - 1 = -0.45; max drawdown = -0.5
    m = metrics_from_returns(pd.Series([0.1, -0.5, 0.0]))
    assert abs(m["total_return"] - (-0.45)) < 1e-9
    assert abs(m["max_drawdown"] - (-0.5)) < 1e-9


def test_positive_series_has_positive_sharpe():
    m = metrics_from_returns(pd.Series([0.01, 0.02, 0.015, 0.005]))
    assert m["sharpe"] > 0
    assert m["ann_volatility"] > 0


def test_first_bar_loss_counts_as_drawdown():
    # starting capital is the initial peak: a 50% loss on the first bar is a -0.5 drawdown
    m = metrics_from_returns(pd.Series([-0.5, 0.0]))
    assert abs(m["max_drawdown"] - (-0.5)) < 1e-9


def test_risk_free_lowers_sharpe():
    # Threading a positive risk-free rate reduces the excess return -> lower Sharpe.
    r = pd.Series([0.01, 0.02, 0.015, 0.005])
    base = metrics_from_returns(r)["sharpe"]
    with_rf = metrics_from_returns(r, risk_free=0.10)["sharpe"]
    assert with_rf < base


def test_default_sharpe_assumes_zero_risk_free():
    # The default (and historical) behaviour is zero-rf: Sharpe == ann_return / ann_vol.
    r = pd.Series([0.01, 0.02, 0.015, 0.005])
    m = metrics_from_returns(r)
    assert abs(m["sharpe"] - m["ann_return"] / m["ann_volatility"]) < 1e-12
