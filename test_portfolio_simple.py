#!/usr/bin/env python3
"""
Simple test for portfolio tracking system.
"""

import sys
import os
from datetime import datetime
from decimal import Decimal

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from domain.value_objects import Symbol, Money, Price, Quantity
from portfolio_tracking import (
    PortfolioConfiguration, PortfolioManager, 
    get_position_manager, get_pnl_calculator
)


def test_basic_functionality():
    """Test basic portfolio functionality."""
    print("Testing basic portfolio functionality...")
    
    try:
        # Test portfolio configuration
        config = PortfolioConfiguration(
            name="Test Portfolio",
            initial_capital=Money(Decimal('100000')),
            currency="USD"
        )
        print("✓ Portfolio configuration created")
        
        # Test portfolio manager creation
        portfolio_manager = PortfolioManager(config)
        print("✓ Portfolio manager created")
        
        # Test position manager
        position_manager = get_position_manager()
        print("✓ Position manager accessed")
        
        # Test P&L calculator
        pnl_calculator = get_pnl_calculator()
        print("✓ P&L calculator accessed")
        
        # Test portfolio summary
        summary = portfolio_manager.get_portfolio_summary()
        print(f"✓ Portfolio summary: {len(summary)} fields")
        
        # Test cash operations
        portfolio_manager.add_cash(Money(Decimal('5000')), "Test deposit")
        assert portfolio_manager.portfolio.cash_balance.amount == Decimal('105000')
        print("✓ Cash deposit works")
        
        # Test market price updates
        prices = {Symbol("AAPL"): Price(Decimal('150.00'))}
        portfolio_manager.update_market_prices(prices)
        print("✓ Market price updates work")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)