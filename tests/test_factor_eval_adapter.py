import dataclasses

import pytest

from algua.backtest.factor_eval import build_factor_strategy
from algua.features.catalogue import FactorNotFound, get_factor


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


def test_build_factor_strategy_rejects_forged_standalone_spec():
    # A forged spec claiming standalone=True but pointing at a non-standalone factor must NOT slip
    # through: the standalone decision rests on the function's own immutable stamp, so the spec
    # mismatch is rejected at resolution (FactorNotFound) rather than wrapping a mis-shaped factor.
    real = get_factor("momentum")  # standalone=False
    forged = dataclasses.replace(real, standalone=True)
    with pytest.raises(FactorNotFound):
        build_factor_strategy(
            forged, symbols=["AAA"], params={"lookback": 5},
            construction="equal_weight_positive", construction_params={},
        )
