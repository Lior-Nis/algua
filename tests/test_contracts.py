from datetime import datetime

import pytest

from algua.contracts.types import ExecutionContract, OrderIntent, Side, Strategy


def test_execution_contract_rejects_negative_lag():
    with pytest.raises(ValueError):
        ExecutionContract(rebalance_frequency="1D", decision_lag_bars=-1)


def test_execution_contract_allows_zero_lag():
    # lag=0 is permitted (same-bar fill, useful for look-ahead comparison tests)
    # but discouraged in production; default is 1.
    c = ExecutionContract(rebalance_frequency="1D", decision_lag_bars=0)
    assert c.decision_lag_bars == 0


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
