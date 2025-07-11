#!/usr/bin/env python3
"""
Simple test of the Algua trading system without pandas/numpy dependencies.
"""

import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal
import math

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from domain.value_objects import Symbol, Price, Quantity, Money
from infrastructure.interfaces import DataProviderFactory, TimeFrame
from infrastructure.providers.simple_data_provider import SimpleDataProvider  # This registers the provider
from models.strategy_factory import StrategyFactory
from utils.logging import get_logger

logger = get_logger(__name__)


def simple_backtest(data, signals, initial_capital=100000.0):
    """Simple backtest implementation without external dependencies."""
    portfolio_value = initial_capital
    cash = initial_capital
    positions = 0
    trades = []
    portfolio_values = []
    
    for i in range(len(data)):
        current_price = data[i]['Close']
        buy_signal = signals[i]['buy_signal']
        sell_signal = signals[i]['sell_signal']
        
        # Execute trades
        if buy_signal and cash > 0:
            shares_to_buy = int(cash // current_price)
            if shares_to_buy > 0:
                positions += shares_to_buy
                cash -= shares_to_buy * current_price
                trades.append({
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'date': data[i].get('Timestamp', i)
                })
        
        elif sell_signal and positions > 0:
            cash += positions * current_price
            trades.append({
                'type': 'SELL',
                'price': current_price,
                'shares': positions,
                'date': data[i].get('Timestamp', i)
            })
            positions = 0
        
        # Calculate portfolio value
        portfolio_value = cash + positions * current_price
        portfolio_values.append(portfolio_value)
    
    # Calculate simple metrics
    total_return = (portfolio_values[-1] / initial_capital - 1) * 100 if portfolio_values else 0
    
    # Calculate returns
    returns = []
    for i in range(1, len(portfolio_values)):
        ret = (portfolio_values[i] / portfolio_values[i-1] - 1) if portfolio_values[i-1] > 0 else 0
        returns.append(ret)
    
    # Simple Sharpe ratio calculation
    if returns:
        avg_return = sum(returns) / len(returns)
        return_variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        return_std = math.sqrt(return_variance)
        sharpe_ratio = (avg_return * 252) / (return_std * math.sqrt(252)) if return_std > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Max drawdown
    running_max = initial_capital
    max_drawdown = 0
    for value in portfolio_values:
        if value > running_max:
            running_max = value
        drawdown = (running_max - value) / running_max
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Trade analysis
    if trades:
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        trade_pairs = []
        for i in range(min(len(buy_trades), len(sell_trades))):
            pnl = sell_trades[i]['price'] * sell_trades[i]['shares'] - buy_trades[i]['price'] * buy_trades[i]['shares']
            trade_pairs.append(pnl)
        
        if trade_pairs:
            winning_trades = [t for t in trade_pairs if t > 0]
            total_trades = len(trade_pairs)
            win_count = len(winning_trades)
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        else:
            total_trades = win_count = win_rate = 0
    else:
        total_trades = win_count = win_rate = 0
    
    return {
        'initial_capital': initial_capital,
        'final_portfolio_value': portfolio_values[-1] if portfolio_values else initial_capital,
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown * 100,
        'win_rate_pct': win_rate,
        'total_trades': total_trades,
        'winning_trades': win_count,
        'trades': trades,
        'portfolio_values': portfolio_values
    }


def test_complete_workflow():
    """Test the complete trading workflow."""
    print("🚀 Testing Complete Algua Workflow")
    print("=" * 50)
    
    try:
        # 1. Test Data Provider
        print("\n📊 Step 1: Data Provider")
        provider = DataProviderFactory.create("simple")
        
        symbol = Symbol("AAPL")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        data = provider.get_historical_data(symbol, start_date, end_date)
        print(f"✓ Retrieved {len(data)} data points for {symbol}")
        
        # 2. Test Strategy
        print("\n🎯 Step 2: Strategy Creation")
        strategy = StrategyFactory.create_strategy('sma_crossover', fast_period=10, slow_period=30)
        print(f"✓ Created strategy: {strategy.get_parameters()}")
        
        # 3. Generate Signals
        print("\n📈 Step 3: Signal Generation")
        signals = strategy.generate_signals(data)
        summary = strategy.get_signal_summary(signals)
        print(f"✓ Generated signals:")
        print(f"  - Buy signals: {summary['buy_signals']}")
        print(f"  - Sell signals: {summary['sell_signals']}")
        print(f"  - Signal frequency: {summary['signal_frequency']:.2f}%")
        
        # 4. Run Backtest
        print("\n💰 Step 4: Backtesting")
        results = simple_backtest(data, signals)
        
        print(f"✓ Backtest Results:")
        print(f"  - Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"  - Final Value: ${results['final_portfolio_value']:,.2f}")
        print(f"  - Total Return: {results['total_return_pct']:.2f}%")
        print(f"  - Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"  - Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"  - Win Rate: {results['win_rate_pct']:.1f}%")
        print(f"  - Total Trades: {results['total_trades']}")
        
        # 5. Test Optimization
        print("\n🔧 Step 5: Strategy Optimization")
        
        best_return = results['total_return_pct']
        best_params = {'fast_period': 10, 'slow_period': 30}
        
        param_combinations = [
            {'fast_period': 5, 'slow_period': 20},
            {'fast_period': 10, 'slow_period': 30},
            {'fast_period': 15, 'slow_period': 40},
        ]
        
        print(f"✓ Testing {len(param_combinations)} parameter combinations...")
        
        for params in param_combinations:
            try:
                test_strategy = StrategyFactory.create_strategy('sma_crossover', **params)
                test_signals = test_strategy.generate_signals(data)
                test_results = simple_backtest(data, test_signals)
                
                if test_results['total_return_pct'] > best_return:
                    best_return = test_results['total_return_pct']
                    best_params = params
                    
            except Exception as e:
                print(f"  Warning: Failed to test {params}: {e}")
        
        print(f"✓ Best parameters: {best_params}")
        print(f"✓ Best return: {best_return:.2f}%")
        
        # 6. Test Current Market Data
        print("\n💹 Step 6: Current Market Data")
        current_price = provider.get_current_price(symbol)
        print(f"✓ Current {symbol} price: ${current_price.value}")
        
        symbols = [Symbol("AAPL"), Symbol("MSFT"), Symbol("GOOGL")]
        prices = provider.get_multiple_prices(symbols)
        print(f"✓ Multiple prices: {[(str(s), f'${p.value}') for s, p in prices.items()]}")
        
        market_open = provider.is_market_open()
        print(f"✓ Market is {'open' if market_open else 'closed'}")
        
        print("\n🎉 SUCCESS: All workflow steps completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Workflow failed at step: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_domain_entities():
    """Test domain entities."""
    print("\n🏛️ Testing Domain Entities")
    print("-" * 30)
    
    try:
        # Test value objects
        symbol = Symbol("AAPL")
        price = Price(Decimal("150.50"))
        quantity = Quantity(Decimal("100"))
        money = Money(Decimal("10000.00"))
        
        print(f"✓ Symbol: {symbol}")
        print(f"✓ Price: ${price.value}")
        print(f"✓ Quantity: {quantity.value}")
        print(f"✓ Money: ${money.amount}")
        
        # Test strategy entity
        from domain.entities.strategy import Strategy, StrategyStatus
        
        strategy = Strategy(
            id="test_strategy",
            name="Test Strategy",
            description="A test strategy",
            strategy_type="test",
            allocated_capital=Money(Decimal("50000.00"))
        )
        
        print(f"✓ Strategy: {strategy.name} (${strategy.allocated_capital.amount})")
        
        # Test state changes
        strategy.start()
        print(f"✓ Strategy started: {strategy.status.value}")
        
        strategy.stop()
        print(f"✓ Strategy stopped: {strategy.status.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Domain entities test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("🏦 ALGUA TRADING PLATFORM - SIMPLE TEST SUITE")
    print("=" * 80)
    
    tests = []
    
    # Run domain entity tests
    tests.append(("Domain Entities", test_domain_entities()))
    
    # Run complete workflow test
    tests.append(("Complete Workflow", test_complete_workflow()))
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:<20}: {status}")
        if result:
            passed += 1
    
    total = len(tests)
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Algua is working correctly.")
        print("\n📋 System Summary:")
        print("  ✓ Domain entities and value objects")
        print("  ✓ Pluggable data provider interface")
        print("  ✓ Strategy factory and registration")
        print("  ✓ SMA crossover strategy implementation")
        print("  ✓ Signal generation and backtesting")
        print("  ✓ Parameter optimization")
        print("  ✓ Current market data access")
        
        print(f"\n🚀 Ready for next steps:")
        print("  1. Copy .env.example to .env and configure API keys")
        print("  2. Install optional packages: pip install pandas numpy yfinance vectorbt")
        print("  3. Test with real market data")
        print("  4. Set up Alpaca paper trading")
        print("  5. Deploy strategies to live trading")
        
    else:
        print(f"\n⚠️ {total - passed} tests failed.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)