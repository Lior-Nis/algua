#!/usr/bin/env python3
"""
Test script to verify component interactions.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from domain.value_objects import Symbol, Price, Quantity
from infrastructure.interfaces import DataProviderFactory, TimeFrame
from backtesting.engine import BacktestEngine


def generate_sample_data(symbol: str = "AAPL", days: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)  # For reproducible results
    base_price = 150.0
    
    data = []
    current_price = base_price
    
    for date in dates:
        # Random walk with slight upward bias
        change = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
        current_price *= (1 + change)
        
        # Generate OHLC from close price
        daily_range = current_price * np.random.uniform(0.01, 0.03)  # 1-3% daily range
        
        high = current_price + np.random.uniform(0, daily_range * 0.7)
        low = current_price - np.random.uniform(0, daily_range * 0.3)
        open_price = low + np.random.uniform(0, high - low)
        
        volume = int(np.random.uniform(1000000, 5000000))  # Random volume
        
        data.append({
            'Timestamp': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': current_price,
            'Volume': volume
        })
    
    return pd.DataFrame(data)


def generate_simple_strategy_signals(data: pd.DataFrame, 
                                   fast_period: int = 10, 
                                   slow_period: int = 30) -> pd.DataFrame:
    """Generate simple moving average crossover signals."""
    close_prices = data['Close']
    
    # Calculate moving averages
    fast_ma = close_prices.rolling(window=fast_period).mean()
    slow_ma = close_prices.rolling(window=slow_period).mean()
    
    # Generate signals
    buy_signal = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    sell_signal = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    
    signals = pd.DataFrame({
        'buy_signal': buy_signal.fillna(False),
        'sell_signal': sell_signal.fillna(False),
        'fast_ma': fast_ma,
        'slow_ma': slow_ma
    })
    
    return signals


def test_data_provider():
    """Test the data provider interface (mock implementation since yfinance might not be available)."""
    print("Testing Data Provider Interface...")
    
    try:
        # List available providers
        providers = DataProviderFactory.list_providers()
        print(f"Available providers: {providers}")
        
        # For testing without external dependencies, we'll use mock data
        symbol = Symbol("AAPL")
        sample_data = generate_sample_data("AAPL", 100)
        
        print(f"Generated sample data for {symbol}")
        print(f"Data shape: {sample_data.shape}")
        print(f"Date range: {sample_data['Timestamp'].min()} to {sample_data['Timestamp'].max()}")
        print(f"Price range: ${sample_data['Close'].min():.2f} - ${sample_data['Close'].max():.2f}")
        
        return sample_data
        
    except Exception as e:
        print(f"Error testing data provider: {e}")
        return None


def test_backtesting_engine():
    """Test the backtesting engine."""
    print("\nTesting Backtesting Engine...")
    
    try:
        # Initialize backtesting engine
        engine = BacktestEngine(initial_capital=100000.0)
        
        # Generate sample data and signals
        data = generate_sample_data("AAPL", 100)
        signals = generate_simple_strategy_signals(data, fast_period=10, slow_period=30)
        
        print(f"Generated {len(signals)} signals")
        print(f"Buy signals: {signals['buy_signal'].sum()}")
        print(f"Sell signals: {signals['sell_signal'].sum()}")
        
        # Run backtest
        results = engine.run_backtest(
            data=data,
            strategy_signals=signals,
            strategy_name="Simple MA Crossover"
        )
        
        # Display results
        print(f"\nBacktest Results for {results['strategy_name']}:")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Value: ${results['final_portfolio_value']:,.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"Win Rate: {results['win_rate_pct']:.1f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        
        return results
        
    except Exception as e:
        print(f"Error testing backtesting engine: {e}")
        return None


def test_strategy_optimization():
    """Test strategy parameter optimization."""
    print("\nTesting Strategy Optimization...")
    
    try:
        engine = BacktestEngine(initial_capital=100000.0)
        data = generate_sample_data("AAPL", 200)
        
        # Define parameter grid for optimization
        param_grid = {
            'fast_period': [5, 10, 15, 20],
            'slow_period': [25, 30, 35, 40]
        }
        
        # Run optimization
        optimization_results = engine.run_optimization(
            data=data,
            strategy_func=generate_simple_strategy_signals,
            param_grid=param_grid,
            strategy_name="MA Crossover Optimization",
            metric="sharpe_ratio"
        )
        
        print(f"Optimization completed!")
        print(f"Total combinations tested: {optimization_results['total_combinations_tested']}")
        print(f"Best parameters: {optimization_results['best_parameters']}")
        print(f"Best Sharpe ratio: {optimization_results['best_score']:.3f}")
        
        best_results = optimization_results['best_results']
        print(f"Best strategy return: {best_results['total_return_pct']:.2f}%")
        
        return optimization_results
        
    except Exception as e:
        print(f"Error testing optimization: {e}")
        return None


def test_domain_entities():
    """Test domain entities."""
    print("\nTesting Domain Entities...")
    
    try:
        # Test Symbol
        symbol = Symbol("AAPL")
        print(f"Created symbol: {symbol}")
        
        # Test Price
        price = Price(Decimal("150.50"))
        print(f"Created price: ${price.value}")
        
        # Test Quantity
        quantity = Quantity(Decimal("100"))
        print(f"Created quantity: {quantity.value}")
        
        print("Domain entities working correctly!")
        return True
        
    except Exception as e:
        print(f"Error testing domain entities: {e}")
        return False


def main():
    """Run all component tests."""
    print("=" * 60)
    print("ALGUA COMPONENT INTEGRATION TESTS")
    print("=" * 60)
    
    # Test individual components
    test_domain_entities()
    
    data = test_data_provider()
    if data is not None:
        backtest_results = test_backtesting_engine()
        
        if backtest_results is not None:
            optimization_results = test_strategy_optimization()
    
    print("\n" + "=" * 60)
    print("COMPONENT TESTS COMPLETED")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Install required packages: yfinance, vectorbt, alpaca-trade-api")
    print("2. Configure API keys in .env file")
    print("3. Test with real market data")
    print("4. Implement more sophisticated trading strategies")
    print("5. Add risk management rules")


if __name__ == "__main__":
    main()