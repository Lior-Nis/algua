#!/usr/bin/env python3
"""
Complete end-to-end test of the Algua trading system.
"""

import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from domain.value_objects import Symbol, Price, Quantity, Money
from infrastructure.interfaces import DataProviderFactory, TimeFrame
from infrastructure.providers.simple_data_provider import SimpleDataProvider
from models.strategy_factory import StrategyFactory
from backtesting.engine import BacktestEngine
from utils.logging import get_logger

logger = get_logger(__name__)


def test_data_provider():
    """Test the data provider."""
    print("🔍 Testing Data Provider...")
    
    try:
        # Create data provider
        provider = DataProviderFactory.create("simple")
        
        # Test current price
        symbol = Symbol("AAPL")
        price = provider.get_current_price(symbol)
        print(f"✓ Current price for {symbol}: ${price.value}")
        
        # Test historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        data = provider.get_historical_data(symbol, start_date, end_date)
        print(f"✓ Retrieved {len(data)} historical data points")
        
        # Test multiple prices
        symbols = [Symbol("AAPL"), Symbol("MSFT"), Symbol("GOOGL")]
        prices = provider.get_multiple_prices(symbols)
        print(f"✓ Retrieved prices for {len(prices)} symbols")
        
        # Test symbol search
        search_results = provider.search_symbols("APP")
        print(f"✓ Found {len(search_results)} symbols matching 'APP'")
        
        # Test market status
        is_open = provider.is_market_open()
        print(f"✓ Market is {'open' if is_open else 'closed'}")
        
        return data
        
    except Exception as e:
        print(f"✗ Data provider test failed: {e}")
        return None


def test_strategy_factory():
    """Test the strategy factory."""
    print("\n📊 Testing Strategy Factory...")
    
    try:
        # List available strategies
        strategies = StrategyFactory.list_strategies()
        print(f"✓ Available strategies: {strategies}")
        
        # Get strategy info
        if 'sma_crossover' in strategies:
            info = StrategyFactory.get_strategy_info('sma_crossover')
            print(f"✓ SMA Crossover info: {info['name']} - {info['description']}")
        
        # Create strategy with default parameters
        strategy = StrategyFactory.create_strategy('sma_crossover')
        print(f"✓ Created strategy with params: {strategy.get_parameters()}")
        
        # Create strategy with custom parameters
        custom_strategy = StrategyFactory.create_strategy(
            'sma_crossover',
            fast_period=5,
            slow_period=20,
            min_volume=50000
        )
        print(f"✓ Created custom strategy with params: {custom_strategy.get_parameters()}")
        
        return strategy
        
    except Exception as e:
        print(f"✗ Strategy factory test failed: {e}")
        return None


def test_strategy_signals(strategy, data):
    """Test strategy signal generation."""
    print("\n🎯 Testing Strategy Signal Generation...")
    
    try:
        if not data:
            print("✗ No data available for signal generation")
            return None
        
        # Validate data
        if not strategy.validate_data(data):
            print("✗ Data validation failed")
            return None
        
        print(f"✓ Data validation passed for {len(data)} data points")
        
        # Generate signals
        signals = strategy.generate_signals(data)
        print(f"✓ Generated {len(signals)} signal points")
        
        # Get signal summary
        summary = strategy.get_signal_summary(signals)
        print(f"✓ Signal summary:")
        print(f"  - Total periods: {summary['total_periods']}")
        print(f"  - Buy signals: {summary['buy_signals']}")
        print(f"  - Sell signals: {summary['sell_signals']}")
        print(f"  - Signal frequency: {summary['signal_frequency']:.2f}%")
        
        if summary['first_signal_period'] is not None:
            print(f"  - First signal at period: {summary['first_signal_period']}")
        
        return signals
        
    except Exception as e:
        print(f"✗ Strategy signal test failed: {e}")
        return None


def test_backtesting(data, signals):
    """Test the backtesting engine."""
    print("\n📈 Testing Backtesting Engine...")
    
    try:
        if not data or not signals:
            print("✗ No data or signals available for backtesting")
            return None
        
        # Initialize backtesting engine
        engine = BacktestEngine(initial_capital=100000.0)
        
        # Convert data to simple format for backtesting
        backtest_data = []
        for item in data:
            backtest_data.append({
                'Open': item['Open'],
                'High': item['High'],
                'Low': item['Low'],
                'Close': item['Close'],
                'Volume': item['Volume'],
                'Timestamp': item.get('Timestamp', datetime.now())
            })
        
        # Convert signals to simple format
        backtest_signals = []
        for signal in signals:
            backtest_signals.append({
                'buy_signal': signal['buy_signal'],
                'sell_signal': signal['sell_signal']
            })
        
        # Run backtest
        results = engine.run_backtest(
            data=backtest_data,
            strategy_signals=backtest_signals,
            strategy_name="SMA Crossover Test"
        )
        
        # Display results
        print(f"✓ Backtest Results:")
        print(f"  - Strategy: {results['strategy_name']}")
        print(f"  - Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"  - Final Value: ${results['final_portfolio_value']:,.2f}")
        print(f"  - Total Return: {results['total_return_pct']:.2f}%")
        print(f"  - Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"  - Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"  - Win Rate: {results['win_rate_pct']:.1f}%")
        print(f"  - Total Trades: {results['total_trades']}")
        print(f"  - Profit Factor: {results['profit_factor']:.2f}")
        
        return results
        
    except Exception as e:
        print(f"✗ Backtesting test failed: {e}")
        return None


def test_strategy_optimization():
    """Test strategy parameter optimization."""
    print("\n🔧 Testing Strategy Optimization...")
    
    try:
        # Create simple data for optimization
        provider = DataProviderFactory.create("simple")
        symbol = Symbol("AAPL")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=200)
        
        data = provider.get_historical_data(symbol, start_date, end_date)
        
        if not data:
            print("✗ No data available for optimization")
            return None
        
        # Convert to format expected by optimization
        def strategy_signal_func(data_list, fast_period=10, slow_period=30):
            """Function that generates signals for optimization."""
            strategy = StrategyFactory.create_strategy(
                'sma_crossover',
                fast_period=fast_period,
                slow_period=slow_period
            )
            signals = strategy.generate_signals(data_list)
            return signals
        
        # Define parameter grid for optimization
        param_grid = {
            'fast_period': [5, 10, 15],
            'slow_period': [20, 30, 40]
        }
        
        # Run optimization (simplified version)
        engine = BacktestEngine()
        
        best_params = None
        best_score = float('-inf')
        
        print(f"✓ Testing {len(param_grid['fast_period']) * len(param_grid['slow_period'])} parameter combinations...")
        
        for fast in param_grid['fast_period']:
            for slow in param_grid['slow_period']:
                if fast >= slow:
                    continue
                
                try:
                    signals = strategy_signal_func(data, fast_period=fast, slow_period=slow)
                    
                    # Convert for backtesting
                    backtest_data = [{
                        'Open': item['Open'],
                        'High': item['High'],
                        'Low': item['Low'],
                        'Close': item['Close'],
                        'Volume': item['Volume'],
                        'Timestamp': item.get('Timestamp', datetime.now())
                    } for item in data]
                    
                    backtest_signals = [{
                        'buy_signal': signal['buy_signal'],
                        'sell_signal': signal['sell_signal']
                    } for signal in signals]
                    
                    results = engine.run_backtest(
                        data=backtest_data,
                        strategy_signals=backtest_signals,
                        strategy_name=f"SMA_{fast}_{slow}"
                    )
                    
                    score = results['sharpe_ratio']
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'fast_period': fast, 'slow_period': slow}
                        
                except Exception as e:
                    print(f"  Warning: Failed to test params fast={fast}, slow={slow}: {e}")
                    continue
        
        if best_params:
            print(f"✓ Optimization completed!")
            print(f"  - Best parameters: {best_params}")
            print(f"  - Best Sharpe ratio: {best_score:.3f}")
        else:
            print("✗ Optimization failed - no valid parameter combinations")
        
        return best_params
        
    except Exception as e:
        print(f"✗ Optimization test failed: {e}")
        return None


def main():
    """Run the complete end-to-end test."""
    print("=" * 80)
    print("🚀 ALGUA END-TO-END SYSTEM TEST")
    print("=" * 80)
    
    # Test data provider
    data = test_data_provider()
    
    # Test strategy factory
    strategy = test_strategy_factory()
    
    # Test strategy signal generation
    signals = None
    if strategy and data:
        signals = test_strategy_signals(strategy, data)
    
    # Test backtesting
    backtest_results = None
    if data and signals:
        backtest_results = test_backtesting(data, signals)
    
    # Test optimization
    optimization_results = test_strategy_optimization()
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)
    
    tests = {
        "Data Provider": data is not None,
        "Strategy Factory": strategy is not None,
        "Signal Generation": signals is not None,
        "Backtesting": backtest_results is not None,
        "Optimization": optimization_results is not None
    }
    
    passed = sum(tests.values())
    total = len(tests)
    
    for test_name, passed_test in tests.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"  {test_name:<20}: {status}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All systems operational! Algua is ready for trading.")
        print("\n📝 Next steps:")
        print("  1. Configure your .env file with real API keys")
        print("  2. Test with real market data using yfinance")
        print("  3. Set up Alpaca paper trading account")
        print("  4. Deploy strategies to live trading")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the errors above.")
    
    print("\n🔧 To configure the system:")
    print("  cp .env.example .env")
    print("  # Edit .env with your API keys")
    print("  python run_full_test.py")


if __name__ == "__main__":
    main()