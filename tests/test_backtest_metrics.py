import pandas as pd

from algua.backtest.metrics import (
    METRIC_FUNCTIONS,
    avg_gross_exposure,
    metrics_from_returns,
    weights_turnover,
)
from algua.backtest.result import BacktestResult


def test_turnover_counts_weight_changes():
    # t0: 100% A; t1: 100% B -> one full rotation = turnover 1.0 at t1
    w = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]})
    assert weights_turnover(w) == 1.0


def test_avg_gross_exposure():
    w = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]})
    assert avg_gross_exposure(w) == 1.0


def test_metric_registry_drives_named_pure_functions():
    # Adding a metric should mean registering a pure fn, not editing a core loop (#40).
    r = pd.Series([0.01, -0.02, 0.03])
    for name, fn in METRIC_FUNCTIONS.items():
        assert isinstance(fn(r), float)
        assert name in metrics_from_returns(r)


def test_drawdown_uses_floored_peak_definition():
    # A first-bar loss must register as drawdown (peak floored at starting capital 1.0).
    assert abs(metrics_from_returns(pd.Series([-0.3]))["max_drawdown"] - (-0.3)) < 1e-9


def test_result_to_dict_is_json_serializable():
    import json
    r = BacktestResult(
        strategy="s", metrics={"sharpe": 1.2}, config_hash="abc",
        data_source="synthetic", timeframe="1d",
        period={"start": "2024-01-01", "end": "2024-03-01"}, seed=0,
    )
    json.dumps(r.to_dict())  # must not raise
    assert r.to_dict()["metrics"]["sharpe"] == 1.2
