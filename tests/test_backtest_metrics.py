import pandas as pd

from algua.backtest.metrics import avg_gross_exposure, weights_turnover
from algua.backtest.result import BacktestResult


def test_turnover_counts_weight_changes():
    # t0: 100% A; t1: 100% B -> one full rotation = turnover 1.0 at t1
    w = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]})
    assert weights_turnover(w) == 1.0


def test_avg_gross_exposure():
    w = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]})
    assert avg_gross_exposure(w) == 1.0


def test_result_to_dict_is_json_serializable():
    import json
    r = BacktestResult(
        strategy="s", metrics={"sharpe": 1.2}, config_hash="abc",
        data_source="synthetic", timeframe="1d",
        period={"start": "2024-01-01", "end": "2024-03-01"}, seed=0,
    )
    json.dumps(r.to_dict())  # must not raise
    assert r.to_dict()["metrics"]["sharpe"] == 1.2
