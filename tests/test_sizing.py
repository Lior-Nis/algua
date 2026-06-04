import pytest

from algua.execution.sizing import MIN_NOTIONAL, size_order


def test_notional_buy_delta_from_flat():
    s = size_order(symbol="AAA", target_weight=0.5, equity=100_000.0, current_market_value=0.0)
    assert s.delta_notional == 50_000.0
    assert s.is_noop is False


def test_notional_sell_delta_when_overweight():
    s = size_order(symbol="AAA", target_weight=0.5, equity=100_000.0,
                   current_market_value=60_000.0)
    assert s.delta_notional == -10_000.0
    assert s.is_noop is False


def test_noop_below_threshold():
    s = size_order(symbol="AAA", target_weight=0.5, equity=100_000.0,
                   current_market_value=50_000.0)
    assert s.is_noop is True
    assert abs(s.delta_notional) < MIN_NOTIONAL


def test_share_path_floors_to_whole_shares():
    # weight 0.5 of 10k = 5000 notional / price 100 = 50 shares, from flat
    s = size_order(symbol="AAA", target_weight=0.5, equity=10_000.0, current_market_value=0.0,
                   price=100.0, current_shares=0.0)
    assert s.delta_shares == 50.0
    assert s.is_noop is False


def test_share_path_subtracts_current_shares():
    s = size_order(symbol="AAA", target_weight=1.0, equity=10_000.0, current_market_value=5_000.0,
                   price=100.0, current_shares=50.0)
    assert s.delta_shares == 50.0  # target 100 shares - 50 held


def test_share_path_zero_share_delta_is_noop():
    # target rounds to the shares already held -> no order
    s = size_order(symbol="AAA", target_weight=1.0, equity=10_000.0, current_market_value=10_000.0,
                   price=100.0, current_shares=100.0)
    assert s.delta_shares == 0.0
    assert s.is_noop is True


def test_share_and_notional_use_same_equity_snapshot():
    # Both representations derive from the same target_weight*equity, so they agree on direction
    s = size_order(symbol="AAA", target_weight=0.3, equity=20_000.0, current_market_value=0.0,
                   price=50.0, current_shares=0.0)
    assert s.delta_notional == pytest.approx(6_000.0)
    assert s.delta_shares == 120.0  # floor(6000/50)


def test_nonpositive_equity_raises():
    with pytest.raises(ValueError, match="positive equity"):
        size_order(symbol="AAA", target_weight=0.5, equity=0.0, current_market_value=0.0)
