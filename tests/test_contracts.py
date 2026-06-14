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


def test_execution_contract_rejects_nonfinite_per_symbol_cap():
    import math

    import pytest as _pytest

    from algua.contracts.types import ExecutionContract
    with _pytest.raises(ValueError, match="max_weight_per_symbol must be finite"):
        ExecutionContract(rebalance_frequency="1d", max_weight_per_symbol=math.nan)
    with _pytest.raises(ValueError, match="max_weight_per_symbol must be finite"):
        ExecutionContract(rebalance_frequency="1d", max_weight_per_symbol=math.inf)


def test_execution_contract_rejects_non_bool_allow_short():
    import pytest as _pytest

    from algua.contracts.types import ExecutionContract
    with _pytest.raises(ValueError, match="allow_short must be a bool"):
        ExecutionContract(rebalance_frequency="1d", allow_short="false")  # type: ignore[arg-type]
    with _pytest.raises(ValueError, match="allow_short must be a bool"):
        ExecutionContract(rebalance_frequency="1d", allow_short=1)  # type: ignore[arg-type]


def test_fundamentals_provider_protocol_and_constants():
    from algua.contracts.types import (
        FUNDAMENTALS_AS_OF_KEY,
        FUNDAMENTALS_COLUMNS,
        FUNDAMENTALS_KNOWABLE_AT,
        FundamentalsProvider,
    )

    assert FUNDAMENTALS_COLUMNS == (
        "symbol", "fiscal_period_end", "metric", "value", "knowable_at", "source",
    )
    assert FUNDAMENTALS_AS_OF_KEY == ("symbol", "fiscal_period_end", "metric")
    assert FUNDAMENTALS_KNOWABLE_AT == "knowable_at"

    class _P:
        snapshot_id = "x"
        def get_fundamentals(self, symbols, end):
            return None

    assert isinstance(_P(), FundamentalsProvider)


def test_news_column_constants():
    from algua.contracts.types import (
        NEWS_AS_OF_KEY,
        NEWS_COLUMNS,
        NEWS_KNOWABLE_AT,
        NEWS_RETRACTED,
    )

    assert NEWS_COLUMNS == (
        "source", "article_id", "symbol", "published_at", "knowable_at",
        "headline", "url", "body", "retracted",
    )
    assert NEWS_AS_OF_KEY == ("source", "article_id", "symbol")
    assert NEWS_KNOWABLE_AT == "knowable_at"
    assert NEWS_RETRACTED == "retracted"
    assert set(NEWS_AS_OF_KEY).issubset(set(NEWS_COLUMNS))
