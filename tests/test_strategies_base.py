import pandas as pd

from algua.contracts.types import ExecutionContract, Strategy
from algua.portfolio.construction import equal_weight_positive
from algua.strategies.base import LoadedStrategy, StrategyConfig


def _signal(view, params):
    # trivial: a unit score per configured symbol -> equal_weight_positive holds them all
    syms = params["symbols"]
    return pd.Series(1.0, index=syms)


def test_loaded_strategy_satisfies_protocol_and_exposes_config():
    cfg = StrategyConfig(
        name="demo",
        universe=["AAPL", "MSFT"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        params={"symbols": ["AAPL", "MSFT"]},
        construction="equal_weight_positive",
    )
    strat = LoadedStrategy(config=cfg, signal_fn=_signal, construct_fn=equal_weight_positive)
    assert strat.name == "demo"
    assert strat.universe == ["AAPL", "MSFT"]
    assert isinstance(strat, Strategy)  # runtime_checkable protocol
    w = strat.target_weights(pd.DataFrame())
    assert abs(w.sum() - 1.0) < 1e-9


def test_target_weights_composes_signal_then_construct():
    import pandas as pd
    from algua.contracts.types import ExecutionContract
    from algua.portfolio.construction import top_k_equal_weight
    from algua.strategies.base import LoadedStrategy, StrategyConfig

    def signal(view, params):
        return pd.Series({"A": 0.9, "B": 0.1, "C": 0.5})

    cfg = StrategyConfig(
        name="s", universe=["A", "B", "C"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        construction="top_k_equal_weight", construction_params={"top_k": 2},
    )
    loaded = LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=top_k_equal_weight)
    w = loaded.target_weights(pd.DataFrame())
    assert set(w.index) == {"A", "C"}
    assert w.to_dict() == {"A": 0.5, "C": 0.5}


def test_construct_reads_current_config_params_not_a_bound_partial():
    import pandas as pd
    from algua.contracts.types import ExecutionContract
    from algua.portfolio.construction import top_k_equal_weight
    from algua.strategies.base import LoadedStrategy, StrategyConfig

    def signal(view, params):
        return pd.Series({"A": 0.9, "B": 0.5, "C": 0.1})

    cfg = StrategyConfig(
        name="s", universe=["A", "B", "C"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        construction="top_k_equal_weight", construction_params={"top_k": 1},
    )
    loaded = LoadedStrategy(config=cfg, signal_fn=signal, construct_fn=top_k_equal_weight)
    # Rebuild with top_k=2 (what a sweep override does); behavior must follow the NEW config.
    loaded2 = LoadedStrategy(
        config=cfg.model_copy(update={"construction_params": {"top_k": 2}}),
        signal_fn=signal, construct_fn=top_k_equal_weight,
    )
    assert len(loaded.target_weights(pd.DataFrame())) == 1
    assert len(loaded2.target_weights(pd.DataFrame())) == 2
