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


def test_strategy_missing_signal_raises():
    from pathlib import Path

    import algua.strategies.examples as examples

    mod_path = Path(examples.__path__[0]) / "_tmp_no_signal_fn.py"
    mod_path.write_text(
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='_tmp_no_signal_fn', universe=['AAPL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d'),\n"
        "    construction='equal_weight_positive')\n"
    )
    try:
        with pytest.raises(StrategyNotFound, match="signal"):
            load_strategy("_tmp_no_signal_fn")
    finally:
        mod_path.unlink()


def test_list_includes_bundled():
    assert "cross_sectional_momentum" in list_strategies()


def test_loader_resolves_and_binds_construction():
    from algua.strategies.loader import load_strategy
    s = load_strategy("cross_sectional_momentum")
    assert s.config.construction == "top_k_equal_weight"
    assert s.signal_fn is not None and s.signal_panel_fn is not None
    assert callable(s.construct_fn)


def test_loader_rejects_unknown_construction(tmp_path, monkeypatch):
    # A module whose CONFIG names a missing policy must fail at load.
    import textwrap
    import algua.strategies.examples as ex
    p = next(iter(ex.__path__)) + "/_tmp_bad_policy.py"
    with open(p, "w") as f:
        f.write(textwrap.dedent('''
            import pandas as pd
            from algua.contracts.types import ExecutionContract
            from algua.strategies.base import StrategyConfig
            CONFIG = StrategyConfig(name="_tmp_bad_policy", universe=["A"],
                execution=ExecutionContract(rebalance_frequency="1d"),
                construction="nope_not_real")
            def signal(view, params):
                return pd.Series(dtype="float64")
        '''))
    try:
        from algua.strategies.loader import StrategyNotFound, load_strategy
        with pytest.raises((StrategyNotFound, ValueError)):
            load_strategy("_tmp_bad_policy")
    finally:
        import os
        os.remove(p)
