"""The fixed-vocabulary metric mapper: sentinel detection and sample-class naming."""
from __future__ import annotations

import pytest

from algua.registry.runs import backtest_metrics, walk_forward_metrics


class _FakeBacktest:
    def __init__(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics
        self.returns = None


def _full_metrics(**overrides: float) -> dict[str, float]:
    base = {
        "sharpe": 1.2, "sortino": 1.4, "total_return": 0.30, "max_drawdown": -0.12,
        "ann_volatility": 0.18, "cagr": 0.15, "calmar": 1.25,
    }
    return base | overrides


def test_maps_to_sample_suffixed_keys() -> None:
    out = backtest_metrics(_FakeBacktest(_full_metrics()))
    assert out["sharpe_is"] == pytest.approx(1.2)
    assert out["total_return_is"] == pytest.approx(0.30)
    assert out["ann_vol_is"] == pytest.approx(0.18)
    assert "sharpe" not in out


def test_zero_ann_vol_nulls_sharpe_and_sortino() -> None:
    """metrics_from_returns returns a 0.0 SENTINEL when ann_volatility == 0."""
    out = backtest_metrics(_FakeBacktest(_full_metrics(ann_volatility=0.0, sharpe=0.0,
                                                       sortino=0.0)))
    assert out["sharpe_is"] is None
    assert out["sortino_is"] is None
    assert out["ann_vol_is"] == pytest.approx(0.0)


def test_zero_max_drawdown_nulls_calmar() -> None:
    out = backtest_metrics(_FakeBacktest(_full_metrics(max_drawdown=0.0, calmar=0.0)))
    assert out["calmar_is"] is None
    assert out["max_drawdown_is"] == pytest.approx(0.0)


def test_a_genuine_zero_sharpe_survives() -> None:
    """A real 0.0 Sharpe with non-zero volatility is a MEASUREMENT, not a sentinel."""
    out = backtest_metrics(_FakeBacktest(_full_metrics(sharpe=0.0)))
    assert out["sharpe_is"] == pytest.approx(0.0)


class _FakeWalkForward:
    def __init__(self) -> None:
        self.stability = {
            "mean_sharpe": 0.8, "std_sharpe": 0.3,
            "min_sharpe": -0.2, "pct_positive_windows": 0.75,
        }
        self.holdout_metrics = {
            "start": "2024-01-01", "end": "2024-06-30", "n_bars": 120,
            "sharpe": 0.4, "sortino": 0.5, "total_return": 0.05,
            "max_drawdown": -0.08, "ann_volatility": 0.14,
        }


def test_walk_forward_maps_holdout_to_oos_and_windows_to_their_own_names() -> None:
    out = walk_forward_metrics(_FakeWalkForward())
    assert out["sharpe_oos"] == pytest.approx(0.4)
    assert out["total_return_oos"] == pytest.approx(0.05)
    assert out["n_obs_oos"] == 120
    assert out["mean_window_sharpe"] == pytest.approx(0.8)
    assert out["min_window_sharpe"] == pytest.approx(-0.2)
    assert out["pct_positive_windows"] == pytest.approx(0.75)
    # A walk-forward measures no in-sample full-period figure — it must not invent one.
    assert "sharpe_is" not in out
