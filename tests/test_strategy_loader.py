import pytest

from algua.strategies.base import LoadedStrategy
from algua.strategies.loader import StrategyNotFound, list_strategies, load_strategy


def test_load_bundled_momentum():
    strat = load_strategy("cross_sectional_momentum")
    assert isinstance(strat, LoadedStrategy)
    assert strat.name == "cross_sectional_momentum"


def test_unknown_strategy_raises():
    with pytest.raises(StrategyNotFound):
        load_strategy("does_not_exist")


def test_strategy_missing_compute_weights_raises():
    from pathlib import Path

    import algua.strategies.examples as examples

    mod_path = Path(examples.__path__[0]) / "_tmp_no_weights_fn.py"
    mod_path.write_text(
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='_tmp_no_weights_fn', universe=['AAPL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'))\n"
    )
    try:
        with pytest.raises(StrategyNotFound, match="compute_weights"):
            load_strategy("_tmp_no_weights_fn")
    finally:
        mod_path.unlink()


def test_list_includes_bundled():
    assert "cross_sectional_momentum" in list_strategies()
