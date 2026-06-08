from datetime import datetime

import pytest

from algua.contracts.types import ExecutionContract, OrderIntent, Side, Strategy


def test_execution_contract_rejects_same_bar_fill():
    with pytest.raises(ValueError):
        ExecutionContract(rebalance_frequency="1D", decision_lag_bars=0)


def test_execution_contract_defaults():
    c = ExecutionContract(rebalance_frequency="1D")
    assert c.decision_lag_bars == 1
    assert c.allow_fractional is True


def test_order_intent_fields():
    oi = OrderIntent(symbol="AAPL", side=Side.BUY, target_weight=0.1,
                     decision_ts=datetime(2025, 1, 2))
    assert oi.symbol == "AAPL"
    assert oi.side is Side.BUY


def test_strategy_protocol_runtime_check():
    class Dummy:
        name = "dummy"
        execution = ExecutionContract(rebalance_frequency="1D")

        def target_weights(self, features):  # noqa: ANN001
            return features

    assert isinstance(Dummy(), Strategy)


def test_execution_contract_warmup_bars_default_and_validation():
    import pytest

    from algua.contracts.types import ExecutionContract

    assert ExecutionContract(rebalance_frequency="1d").warmup_bars == 0
    assert ExecutionContract(rebalance_frequency="1d", warmup_bars=30).warmup_bars == 30
    with pytest.raises(ValueError, match="warmup_bars"):
        ExecutionContract(rebalance_frequency="1d", warmup_bars=-1)


def test_execution_contract_new_fields_default_to_todays_behavior():
    c = ExecutionContract(rebalance_frequency="1d")
    assert c.max_weight_per_symbol == 1.0   # no cap by default
    assert c.allow_short is False           # long-only by default


def test_execution_contract_rejects_nonpositive_per_symbol_cap():
    with pytest.raises(ValueError, match="max_weight_per_symbol must be > 0"):
        ExecutionContract(rebalance_frequency="1d", max_weight_per_symbol=0.0)
    with pytest.raises(ValueError, match="max_weight_per_symbol must be > 0"):
        ExecutionContract(rebalance_frequency="1d", max_weight_per_symbol=-0.1)
