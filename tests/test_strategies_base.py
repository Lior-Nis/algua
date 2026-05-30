import pandas as pd

from algua.contracts.types import ExecutionContract, Strategy
from algua.strategies.base import LoadedStrategy, StrategyConfig


def _tw(view, params):
    # trivial: equal weight the configured universe
    syms = params["symbols"]
    return pd.Series(1.0 / len(syms), index=syms)


def test_loaded_strategy_satisfies_protocol_and_exposes_config():
    cfg = StrategyConfig(
        name="demo",
        universe=["AAPL", "MSFT"],
        execution=ExecutionContract(rebalance_frequency="1d"),
        params={"symbols": ["AAPL", "MSFT"]},
    )
    strat = LoadedStrategy(config=cfg, fn=_tw)
    assert strat.name == "demo"
    assert strat.universe == ["AAPL", "MSFT"]
    assert isinstance(strat, Strategy)  # runtime_checkable protocol
    w = strat.target_weights(pd.DataFrame())
    assert abs(w.sum() - 1.0) < 1e-9
