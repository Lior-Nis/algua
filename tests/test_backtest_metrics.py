import pandas as pd

from algua.backtest.metrics import (
    METRIC_FUNCTIONS,
    avg_gross_exposure,
    metrics_from_returns,
    weights_turnover,
)
from algua.backtest.result import BacktestResult, config_hash
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig


def _strategy(**execution_kwargs):
    execution = ExecutionContract(rebalance_frequency="1d", **execution_kwargs)
    config = StrategyConfig(
        name="s", universe=["A"], execution=execution, params={},
        construction="top_k_equal_weight", construction_params={"top_k": 1},
    )
    # These tests only exercise config_hash; the signal/construct never run. Use an identity
    # construct (scores ARE weights) so no policy re-normalization muddies intent.
    return LoadedStrategy(
        config=config,
        signal_fn=lambda features, params: features.iloc[-1],
        construct_fn=lambda scores, view, params: scores,
    )


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


def test_config_hash_distinguishes_warmup_bars():
    # Two configs differing only in warmup_bars produce different trades, so their
    # config_hash must differ (reproducibility/provenance).
    assert config_hash(_strategy(warmup_bars=0)) != config_hash(_strategy(warmup_bars=5))


def test_config_hash_distinguishes_allow_fractional():
    # allow_fractional changes execution behavior; it must be part of the identity hash.
    assert config_hash(_strategy(allow_fractional=True)) != config_hash(
        _strategy(allow_fractional=False)
    )


def test_config_hash_stable_for_identical_execution():
    assert config_hash(_strategy(warmup_bars=3)) == config_hash(_strategy(warmup_bars=3))


def test_one_sample_returns_keep_metrics_finite():
    # A one-bar series has ddof=1 std == NaN; volatility must be coerced safe so Sharpe
    # and every metric stay finite (the "safe on ... zero-vol input" contract).
    import math

    out = metrics_from_returns(pd.Series([0.05]))
    assert all(math.isfinite(v) for v in out.values())
    assert out["ann_volatility"] == 0.0
    assert out["sharpe"] == 0.0


def test_result_to_dict_is_json_serializable():
    import json
    r = BacktestResult(
        strategy="s", metrics={"sharpe": 1.2}, config_hash="abc",
        data_source="synthetic", timeframe="1d",
        period={"start": "2024-01-01", "end": "2024-03-01"}, seed=0,
    )
    json.dumps(r.to_dict())  # must not raise
    assert r.to_dict()["metrics"]["sharpe"] == 1.2
