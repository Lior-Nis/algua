"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Generator

from configs.settings import get_settings
from domain.value_objects import Symbol, Price, Quantity, Money
from domain.entities import Position, PositionSide


@pytest.fixture(scope="session")
def test_settings():
    """Test settings configuration."""
    import os
    os.environ["ENVIRONMENT"] = "testing"
    return get_settings()


@pytest.fixture
def sample_symbol() -> Symbol:
    """Sample trading symbol."""
    return Symbol("AAPL", "NASDAQ")


@pytest.fixture
def sample_price() -> Price:
    """Sample price."""
    return Price(Decimal("150.50"))


@pytest.fixture
def sample_quantity() -> Quantity:
    """Sample quantity."""
    return Quantity(Decimal("100"))


@pytest.fixture
def sample_money() -> Money:
    """Sample money amount."""
    return Money(Decimal("10000.00"), "USD")


@pytest.fixture
def sample_position(sample_symbol, sample_quantity, sample_price) -> Position:
    """Sample trading position."""
    return Position(
        id="pos_123",
        symbol=sample_symbol,
        side=PositionSide.LONG,
        quantity=sample_quantity,
        avg_price=sample_price,
        opened_at=datetime.now()
    )


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Sample OHLCV market data."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * np.cumprod(1 + returns)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    
    # Ensure high >= open, close and low <= open, close
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data


@pytest.fixture
def sample_signals() -> pd.DataFrame:
    """Sample trading signals."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate some random signals
    np.random.seed(42)
    signals = pd.DataFrame({
        'buy': np.random.choice([True, False], len(dates), p=[0.05, 0.95]),
        'sell': np.random.choice([True, False], len(dates), p=[0.05, 0.95]),
        'position': 0
    }, index=dates)
    
    # Ensure no simultaneous buy/sell signals
    simultaneous = signals['buy'] & signals['sell']
    signals.loc[simultaneous, ['buy', 'sell']] = False
    
    return signals


@pytest.fixture
def mock_alpaca_client():
    """Mock Alpaca client for testing."""
    class MockAlpacaClient:
        def __init__(self):
            self.positions = []
            self.orders = []
            
        def get_positions(self):
            return self.positions
            
        def submit_order(self, symbol, qty, side, type="market"):
            order = {
                "id": f"order_{len(self.orders)}",
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": type,
                "status": "filled"
            }
            self.orders.append(order)
            return order
            
        def get_orders(self):
            return self.orders
    
    return MockAlpacaClient()


@pytest.fixture
def temporary_database():
    """Temporary database for testing."""
    # This would be implemented when we add actual database models
    pass


class DatabaseTransaction:
    """Context manager for database transactions in tests."""
    
    def __enter__(self):
        # Start transaction
        pass
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Rollback transaction
        pass


@pytest.fixture
def db_transaction():
    """Database transaction fixture for test isolation."""
    return DatabaseTransaction()


# Parametrized fixtures for testing different scenarios
@pytest.fixture(params=[
    ("AAPL", "NASDAQ"),
    ("GOOGL", "NASDAQ"),
    ("TSLA", "NASDAQ"),
    ("SPY", "NYSE")
])
def various_symbols(request):
    """Fixture providing various symbols for parametrized tests."""
    ticker, exchange = request.param
    return Symbol(ticker, exchange)


@pytest.fixture(params=[
    PositionSide.LONG,
    PositionSide.SHORT
])
def position_sides(request):
    """Fixture providing different position sides."""
    return request.param


@pytest.fixture(params=[
    Decimal("50.00"),
    Decimal("150.75"),
    Decimal("1000.50"),
    Decimal("0.01")
])
def various_prices(request):
    """Fixture providing various price points."""
    return Price(request.param)


# Async fixtures for testing async code
@pytest.fixture
async def async_mock_client():
    """Async mock client for testing."""
    class AsyncMockClient:
        async def fetch_data(self, symbol):
            return {"symbol": symbol, "price": 100.0}
    
    return AsyncMockClient()


# Performance testing fixtures
@pytest.fixture
def performance_data():
    """Large dataset for performance testing."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='1min')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'price': 100 + np.cumsum(np.random.randn(len(dates)) * 0.1),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    return data