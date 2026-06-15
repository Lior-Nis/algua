import pytest

from algua.backtest.factor_eval import build_factor_strategy
from algua.features.catalogue import get_factor


def test_build_factor_strategy_wraps_a_standalone_factor():
    spec = get_factor("xs_trailing_return")
    strat = build_factor_strategy(
        spec,
        symbols=["AAA", "BBB"],
        params={"lookback": 5},
        construction="top_k_equal_weight",
        construction_params={"top_k": 1},
    )
    assert strat.name == "__factor__:xs_trailing_return"
    assert strat.universe == ["AAA", "BBB"]
    assert strat.signal_fn is not None
    assert strat.config.construction == "top_k_equal_weight"


def test_build_factor_strategy_rejects_non_standalone():
    spec = get_factor("momentum")  # standalone=False
    with pytest.raises(ValueError, match="not standalone-evaluable"):
        build_factor_strategy(
            spec, symbols=["AAA"], params={"lookback": 5},
            construction="equal_weight_positive", construction_params={},
        )
