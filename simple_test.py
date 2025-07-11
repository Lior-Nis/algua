#!/usr/bin/env python3
"""
Simple test script to verify basic component functionality without external dependencies.
"""

import sys
import os
from decimal import Decimal
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_domain_entities():
    """Test domain entities without external dependencies."""
    print("Testing Domain Entities...")
    
    try:
        from domain.value_objects import Symbol, Price, Quantity, Money
        
        # Test Symbol
        symbol = Symbol("AAPL")
        print(f"✓ Created symbol: {symbol}")
        
        # Test Price
        price = Price(Decimal("150.50"))
        print(f"✓ Created price: ${price.value}")
        
        # Test Quantity
        quantity = Quantity(Decimal("100"))
        print(f"✓ Created quantity: {quantity.value}")
        
        # Test Money
        money = Money(Decimal("10000.00"))
        print(f"✓ Created money: ${money.amount}")
        
        print("✓ Domain entities working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing domain entities: {e}")
        return False


def test_interfaces():
    """Test interface definitions."""
    print("\nTesting Interface Definitions...")
    
    try:
        from infrastructure.interfaces import (
            MarketDataProvider, BrokerInterface, TimeFrame, OrderType, 
            OrderSide, OrderStatus, DataProviderFactory, BrokerFactory
        )
        
        # Test enums
        print(f"✓ TimeFrame enum: {list(TimeFrame)[:3]}...")
        print(f"✓ OrderType enum: {list(OrderType)}")
        print(f"✓ OrderSide enum: {list(OrderSide)}")
        print(f"✓ OrderStatus enum: {list(OrderStatus)}")
        
        # Test factories
        providers = DataProviderFactory.list_providers()
        brokers = BrokerFactory.list_brokers()
        print(f"✓ Available data providers: {providers}")
        print(f"✓ Available brokers: {brokers}")
        
        print("✓ Interface definitions working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing interfaces: {e}")
        return False


def test_configuration():
    """Test configuration system."""
    print("\nTesting Configuration System...")
    
    try:
        from configs.settings import get_settings, get_config_class
        
        # Test configuration class detection
        config_class = get_config_class()
        print(f"✓ Configuration class: {config_class.__name__}")
        
        # Note: We can't fully test settings without environment setup
        print("✓ Configuration system accessible!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing configuration: {e}")
        return False


def test_dependency_injection():
    """Test dependency injection container."""
    print("\nTesting Dependency Injection...")
    
    try:
        from infrastructure.container import get_container, Container
        
        container = get_container()
        print(f"✓ Container instance: {type(container).__name__}")
        
        # Test registering a simple service
        container.register("test_service", "test_value")
        retrieved = container.get("test_service")
        
        if retrieved == "test_value":
            print("✓ Service registration and retrieval working!")
        else:
            print("✗ Service registration failed")
            return False
        
        print("✓ Dependency injection working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing dependency injection: {e}")
        return False


def test_logging():
    """Test logging system."""
    print("\nTesting Logging System...")
    
    try:
        from utils.logging import get_logger
        
        logger = get_logger("test_logger")
        logger.info("Test log message")
        print("✓ Logger created and working!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing logging: {e}")
        return False


def test_strategy_entity():
    """Test strategy entity."""
    print("\nTesting Strategy Entity...")
    
    try:
        from domain.entities.strategy import Strategy, StrategyStatus, StrategyPerformance
        from domain.value_objects import Money
        
        # Create a strategy
        strategy = Strategy(
            id="test_strategy_1",
            name="Test Strategy",
            description="A test trading strategy",
            strategy_type="test",
            allocated_capital=Money(Decimal("50000.00"))
        )
        
        print(f"✓ Created strategy: {strategy.name}")
        print(f"✓ Strategy ID: {strategy.id}")
        print(f"✓ Allocated capital: ${strategy.allocated_capital.amount}")
        print(f"✓ Status: {strategy.status.value}")
        print(f"✓ Is active: {strategy.is_active}")
        
        # Test strategy state changes
        strategy.start()
        print(f"✓ Strategy started, status: {strategy.status.value}")
        
        strategy.pause()
        print(f"✓ Strategy paused, status: {strategy.status.value}")
        
        strategy.resume()
        print(f"✓ Strategy resumed, status: {strategy.status.value}")
        
        strategy.stop()
        print(f"✓ Strategy stopped, status: {strategy.status.value}")
        
        print("✓ Strategy entity working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing strategy entity: {e}")
        return False


def test_portfolio_entity():
    """Test portfolio entity."""
    print("\nTesting Portfolio Entity...")
    
    try:
        from domain.entities.portfolio import Portfolio
        from domain.value_objects import Money
        
        # Create a portfolio
        portfolio = Portfolio(
            id="test_portfolio_1",
            name="Test Portfolio",
            initial_capital=Money(Decimal("100000.00")),
            cash=Money(Decimal("100000.00"))
        )
        
        print(f"✓ Created portfolio: {portfolio.name}")
        print(f"✓ Portfolio ID: {portfolio.id}")
        print(f"✓ Initial capital: ${portfolio.initial_capital.amount}")
        print(f"✓ Current cash: ${portfolio.cash.amount}")
        print(f"✓ Total value: ${portfolio.total_value.amount}")
        print(f"✓ Total return: {portfolio.total_return}%")
        
        print("✓ Portfolio entity working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing portfolio entity: {e}")
        return False


def main():
    """Run all basic component tests."""
    print("=" * 60)
    print("ALGUA BASIC COMPONENT TESTS")
    print("=" * 60)
    
    tests = [
        test_domain_entities,
        test_interfaces,
        test_configuration,
        test_dependency_injection,
        test_logging,
        test_strategy_entity,
        test_portfolio_entity
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"TESTS COMPLETED: {passed}/{total} PASSED")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All basic components are working correctly!")
        print("\nNext steps:")
        print("1. Install external dependencies (pandas, yfinance, vectorbt, alpaca-trade-api)")
        print("2. Create .env file with API keys")
        print("3. Test with real market data")
        print("4. Implement trading strategies")
        print("5. Set up live trading")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the errors above.")
    
    print("\nTo run with external dependencies:")
    print("pip install pandas numpy yfinance vectorbt alpaca-trade-api")
    print("python test_components.py")


if __name__ == "__main__":
    main()