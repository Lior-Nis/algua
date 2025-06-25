"""
Unit tests for the backtesting module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtesting.engine import BacktestEngine
from models.strategies import MeanReversionStrategy


class TestBacktestEngine:
    """Test cases for BacktestEngine."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic OHLCV data
        close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        high_prices = close_prices + np.random.rand(len(dates)) * 2
        low_prices = close_prices - np.random.rand(len(dates)) * 2
        open_prices = close_prices + np.random.randn(len(dates)) * 0.5
        volumes = np.random.randint(100000, 1000000, len(dates))
        
        data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)
        
        return data
    
    def test_initialization(self):
        """Test BacktestEngine initialization."""
        engine = BacktestEngine(initial_capital=100000.0)
        assert engine.initial_capital == 100000.0
        assert engine.settings is not None
    
    def test_run_backtest_dummy(self):
        """Test running a dummy backtest."""
        engine = BacktestEngine(initial_capital=100000.0)
        
        # Create minimal test data
        data = pd.DataFrame({'close': [100, 101, 102]})
        signals = pd.DataFrame({'buy': [True, False, False]})
        
        result = engine.run_backtest(
            data=data,
            strategy_signals=signals,
            strategy_name="TestStrategy"
        )
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'strategy_name' in result
        assert 'total_return' in result
        assert 'sharpe_ratio' in result
        assert result['strategy_name'] == "TestStrategy"


# TODO: Add more comprehensive tests when actual implementations are complete 