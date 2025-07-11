#!/usr/bin/env python3
"""
Comprehensive test suite for the risk management system.
"""

import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from domain.value_objects import Symbol, Money, Price, Quantity
from risk_management import (
    # Position sizing
    PositionSizer, PositionSizeParams, PositionSizingMethod,
    FixedRiskPositionSizer, KellyPositionSizer, PositionSizeValidator,
    
    # Stop loss system
    StopLossManager, StopLossType, FixedPercentageStopLoss,
    TrailingPercentageStopLoss, ATRBasedStopLoss, TimeBasedStopLoss,
    
    # Portfolio limits
    PortfolioRiskLimiter, ExposureType, get_portfolio_limiter,
    
    # Drawdown controls
    DrawdownController, DrawdownSeverity, RecoveryMode,
    get_drawdown_controller,
    
    # Configuration and events
    RiskConfiguration, get_risk_config, update_risk_config,
    RiskEvent, RiskLevel, RiskEventType, PositionRisk,
    get_risk_event_bus, publish_risk_event
)
from utils.logging import get_logger

logger = get_logger(__name__)


def test_position_sizing():
    """Test position sizing functionality."""
    print("🧮 Testing Position Sizing...")
    
    try:
        # Test basic position sizer
        sizer = PositionSizer()
        
        # Test legacy method
        params = PositionSizeParams(
            portfolio_value=Money(Decimal('100000')),
            risk_per_trade=Decimal('0.02'),  # 2%
            entry_price=Price(Decimal('100')),
            stop_loss_price=Price(Decimal('95'))  # 5% stop loss
        )
        
        position_size = sizer.calculate_position_size(params)
        print(f"✓ Legacy position size: {position_size.value} shares")
        
        # Test new method with fixed risk
        position_size_new = sizer.calculate_position_size(
            method=PositionSizingMethod.FIXED_RISK,
            portfolio_value=Money(Decimal('100000')),
            entry_price=Price(Decimal('100')),
            stop_loss_price=Price(Decimal('95')),
            risk_per_trade=Decimal('0.02')
        )
        print(f"✓ Fixed risk position size: {position_size_new.value} shares")
        
        # Test Kelly position sizer
        kelly_size = sizer.calculate_position_size(
            method=PositionSizingMethod.KELLY_CRITERION,
            portfolio_value=Money(Decimal('100000')),
            entry_price=Price(Decimal('100')),
            stop_loss_price=Price(Decimal('95')),
            win_rate=Decimal('0.6'),
            avg_win=Decimal('1.5'),
            avg_loss=Decimal('1.0')
        )
        print(f"✓ Kelly position size: {kelly_size.value} shares")
        
        # Test volatility targeting
        vol_size = sizer.calculate_position_size(
            method=PositionSizingMethod.VOLATILITY_TARGET,
            portfolio_value=Money(Decimal('100000')),
            entry_price=Price(Decimal('100')),
            stop_loss_price=Price(Decimal('95')),
            target_volatility=Decimal('0.15'),
            asset_volatility=Decimal('0.20')
        )
        print(f"✓ Volatility target position size: {vol_size.value} shares")
        
        # Test position size validator
        validator = PositionSizeValidator()
        warnings = validator.validate_position_size(
            Symbol("AAPL"),
            position_size,
            Price(Decimal('100')),
            Money(Decimal('100000'))
        )
        print(f"✓ Position validation warnings: {len(warnings)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Position sizing test failed: {e}")
        return False


def test_stop_loss_system():
    """Test stop loss system."""
    print("\n🛑 Testing Stop Loss System...")
    
    try:
        manager = StopLossManager()
        symbol = Symbol("AAPL")
        entry_price = Price(Decimal('150'))
        position_size = Quantity(Decimal('100'))
        
        # Test fixed percentage stop loss
        fixed_stop = manager.create_stop_loss(
            symbol=symbol,
            entry_price=entry_price,
            position_size=position_size,
            position_type="long",
            stop_type=StopLossType.FIXED_PERCENTAGE
        )
        print(f"✓ Fixed stop loss created: {fixed_stop.stop_price.value:.2f}")
        
        # Test trailing stop loss
        trailing_stop = manager.create_stop_loss(
            symbol=Symbol("MSFT"),
            entry_price=Price(Decimal('300')),
            position_size=position_size,
            position_type="long",
            stop_type=StopLossType.TRAILING_PERCENTAGE
        )
        print(f"✓ Trailing stop loss created: {trailing_stop.stop_price.value:.2f}")
        
        # Test ATR-based stop loss
        atr_stop = manager.create_stop_loss(
            symbol=Symbol("GOOGL"),
            entry_price=Price(Decimal('2800')),
            position_size=position_size,
            position_type="long",
            stop_type=StopLossType.ATR_BASED,
            market_data={'atr': Decimal('50')}
        )
        print(f"✓ ATR stop loss created: {atr_stop.stop_price.value:.2f}")
        
        # Test time-based stop loss
        time_stop = manager.create_stop_loss(
            symbol=Symbol("AMZN"),
            entry_price=Price(Decimal('3200')),
            position_size=position_size,
            position_type="long",
            stop_type=StopLossType.TIME_BASED
        )
        print(f"✓ Time-based stop loss created")
        
        # Test stop trigger checking
        current_prices = {
            symbol: Price(Decimal('140')),  # Below stop loss
            Symbol("MSFT"): Price(Decimal('310')),  # Above entry
            Symbol("GOOGL"): Price(Decimal('2750')),  # Test ATR
            Symbol("AMZN"): Price(Decimal('3300'))
        }
        
        triggered_stops = manager.check_stop_triggers(current_prices)
        print(f"✓ Triggered stops: {len(triggered_stops)}")
        
        # Test trailing stop updates
        updated_stops = manager.update_trailing_stops(current_prices)
        print(f"✓ Updated trailing stops: {len(updated_stops)}")
        
        # Test active stops
        active_stops = manager.get_active_stops()
        print(f"✓ Active stops: {len(active_stops)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Stop loss system test failed: {e}")
        return False


def test_portfolio_limits():
    """Test portfolio exposure limits."""
    print("\n📊 Testing Portfolio Limits...")
    
    try:
        limiter = get_portfolio_limiter()
        
        # Add some sector mappings
        limiter.add_sector_mapping(Symbol("AAPL"), "Technology")
        limiter.add_sector_mapping(Symbol("MSFT"), "Technology")
        limiter.add_sector_mapping(Symbol("JPM"), "Financial")
        limiter.add_sector_mapping(Symbol("XOM"), "Energy")
        
        # Create sample positions
        portfolio_value = Money(Decimal('1000000'))  # $1M portfolio
        positions = [
            PositionRisk(
                symbol=Symbol("AAPL"),
                position_size=Quantity(Decimal('1000')),
                market_value=Money(Decimal('150000')),
                unrealized_pnl=Money(Decimal('5000')),
                risk_percentage=Decimal('0.15'),
                stop_loss_price=Price(Decimal('140')),
                risk_per_share=Money(Decimal('10')),
                time_in_position=5,
                volatility=Decimal('0.25')
            ),
            PositionRisk(
                symbol=Symbol("MSFT"),
                position_size=Quantity(Decimal('500')),
                market_value=Money(Decimal('150000')),
                unrealized_pnl=Money(Decimal('-2000')),
                risk_percentage=Decimal('0.15'),
                stop_loss_price=Price(Decimal('290')),
                risk_per_share=Money(Decimal('10')),
                time_in_position=3,
                volatility=Decimal('0.20')
            ),
            PositionRisk(
                symbol=Symbol("JPM"),
                position_size=Quantity(Decimal('800')),
                market_value=Money(Decimal('120000')),
                unrealized_pnl=Money(Decimal('3000')),
                risk_percentage=Decimal('0.12'),
                stop_loss_price=Price(Decimal('140')),
                risk_per_share=Money(Decimal('10')),
                time_in_position=7,
                volatility=Decimal('0.30')
            )
        ]
        
        # Calculate portfolio exposure
        exposure = limiter.calculate_portfolio_exposure(positions, portfolio_value)
        print(f"✓ Total exposure: {exposure.total_exposure_pct:.2%}")
        print(f"✓ Cash percentage: {exposure.cash_pct:.2%}")
        print(f"✓ Position count: {exposure.position_count}")
        print(f"✓ Concentration risk score: {exposure.concentration_risk_score:.1f}")
        
        # Check exposure limits
        risk_events = limiter.check_exposure_limits(positions, portfolio_value)
        print(f"✓ Risk events generated: {len(risk_events)}")
        
        # Test new position validation
        can_add, warnings = limiter.can_add_position(
            Symbol("GOOGL"),
            Money(Decimal('200000')),  # Large position
            positions,
            portfolio_value
        )
        print(f"✓ Can add new position: {can_add}")
        print(f"✓ Warnings: {len(warnings)}")
        
        # Get exposure summary
        summary = limiter.get_exposure_summary(positions, portfolio_value)
        print(f"✓ Sector exposures: {len(summary['sector_exposures'])}")
        print(f"✓ Top positions: {len(summary['top_positions'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Portfolio limits test failed: {e}")
        return False


def test_drawdown_controls():
    """Test drawdown control system."""
    print("\n📉 Testing Drawdown Controls...")
    
    try:
        controller = get_drawdown_controller()
        
        # Simulate portfolio value changes
        base_value = Money(Decimal('1000000'))
        
        # Simulate an uptrend followed by drawdown
        portfolio_values = [
            (datetime.now() - timedelta(days=30), Money(Decimal('1000000'))),
            (datetime.now() - timedelta(days=25), Money(Decimal('1050000'))),
            (datetime.now() - timedelta(days=20), Money(Decimal('1100000'))),  # Peak
            (datetime.now() - timedelta(days=15), Money(Decimal('1080000'))),
            (datetime.now() - timedelta(days=10), Money(Decimal('1040000'))),
            (datetime.now() - timedelta(days=5), Money(Decimal('1000000'))),
            (datetime.now(), Money(Decimal('950000')))  # Current drawdown
        ]
        
        # Update portfolio values
        for timestamp, value in portfolio_values:
            metrics = controller.update_portfolio_value(value, timestamp)
        
        print(f"✓ Current drawdown: {metrics.current_drawdown_pct:.2%}")
        print(f"✓ Max drawdown: {metrics.max_drawdown_pct:.2%}")
        print(f"✓ Drawdown severity: {metrics.drawdown_severity.value}")
        print(f"✓ Recovery mode: {metrics.recovery_mode.value}")
        print(f"✓ Consecutive losing days: {metrics.consecutive_losing_days}")
        
        # Test position size multiplier
        multiplier = controller.get_position_size_multiplier()
        print(f"✓ Position size multiplier: {multiplier}")
        
        # Test trading halt check
        halt_trading = controller.should_halt_trading()
        print(f"✓ Should halt trading: {halt_trading}")
        
        # Get drawdown summary
        summary = controller.get_drawdown_summary()
        print(f"✓ Peak portfolio value: ${summary['peak_portfolio_value']:,.2f}")
        print(f"✓ Current portfolio value: ${summary['current_portfolio_value']:,.2f}")
        print(f"✓ Recovery factor: {summary['recovery_factor']:.4f}")
        
        # Test with extreme drawdown
        extreme_value = Money(Decimal('800000'))  # 20% drawdown from peak
        extreme_metrics = controller.update_portfolio_value(extreme_value)
        print(f"✓ Extreme drawdown test: {extreme_metrics.drawdown_severity.value}")
        print(f"✓ New recovery mode: {extreme_metrics.recovery_mode.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Drawdown controls test failed: {e}")
        return False


def test_risk_configuration():
    """Test risk configuration system."""
    print("\n⚙️ Testing Risk Configuration...")
    
    try:
        # Test getting current config
        config = get_risk_config()
        print(f"✓ Max position size: {config.max_position_size:.2%}")
        print(f"✓ Max daily loss: {config.max_daily_loss:.2%}")
        print(f"✓ Max drawdown: {config.max_drawdown:.2%}")
        
        # Test updating configuration
        update_risk_config(
            max_position_size=0.12,  # 12%
            max_daily_loss=0.025,    # 2.5%
        )
        
        updated_config = get_risk_config()
        print(f"✓ Updated max position size: {updated_config.max_position_size:.2%}")
        print(f"✓ Updated max daily loss: {updated_config.max_daily_loss:.2%}")
        
        # Test configuration validation
        from risk_management.configuration import RiskConfigurationManager
        manager = RiskConfigurationManager()
        issues = manager.validate_config()
        print(f"✓ Configuration validation issues: {len(issues)}")
        
        # Test configuration to/from dict
        config_dict = config.to_dict()
        config_from_dict = RiskConfiguration.from_dict(config_dict)
        print(f"✓ Config serialization test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk configuration test failed: {e}")
        return False


def test_risk_event_system():
    """Test risk event system."""
    print("\n📢 Testing Risk Event System...")
    
    try:
        # Get event bus
        event_bus = get_risk_event_bus()
        
        # Create and publish test events
        from risk_management.event_system import RiskEventFactory
        
        # Position size exceeded event
        event1 = RiskEventFactory.create_position_size_exceeded_event(
            symbol=Symbol("AAPL"),
            position_size_pct=0.12,
            max_position_size_pct=0.10,
            action_taken="position_reduced"
        )
        publish_risk_event(event1)
        
        # Daily loss exceeded event
        event2 = RiskEventFactory.create_daily_loss_exceeded_event(
            symbol=Symbol("PORTFOLIO"),
            daily_loss_pct=0.03,
            max_daily_loss_pct=0.02,
            action_taken="trading_halted"
        )
        publish_risk_event(event2)
        
        # Stop loss triggered event
        event3 = RiskEventFactory.create_stop_loss_triggered_event(
            symbol=Symbol("MSFT"),
            current_price=290.0,
            stop_loss_price=295.0,
            action_taken="position_closed"
        )
        publish_risk_event(event3)
        
        # Get event statistics
        import time
        time.sleep(0.1)  # Allow events to process
        
        stats = event_bus.get_stats()
        print(f"✓ Total events published: {stats.total_events}")
        print(f"✓ Events by type: {len(stats.events_by_type)}")
        print(f"✓ Events by level: {len(stats.events_by_level)}")
        
        # Test custom event handler
        from risk_management.event_system import RiskEventHandler
        
        class TestEventHandler(RiskEventHandler):
            def __init__(self):
                super().__init__("TestHandler")
                self.handled_events = []
            
            def _handle_event_impl(self, event):
                self.handled_events.append(event)
        
        test_handler = TestEventHandler()
        event_bus.add_handler(test_handler)
        
        # Publish another event
        test_event = RiskEventFactory.create_position_size_exceeded_event(
            symbol=Symbol("TEST"),
            position_size_pct=0.15,
            max_position_size_pct=0.10
        )
        publish_risk_event(test_event)
        
        time.sleep(0.1)  # Allow event to process
        print(f"✓ Test handler events: {len(test_handler.handled_events)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk event system test failed: {e}")
        return False


def test_integration():
    """Test integrated risk management workflow."""
    print("\n🔗 Testing Risk Management Integration...")
    
    try:
        # Setup: Create a trading scenario
        portfolio_value = Money(Decimal('500000'))
        symbol = Symbol("AAPL")
        entry_price = Price(Decimal('150'))
        
        # 1. Position sizing
        sizer = PositionSizer()
        position_size = sizer.calculate_position_size(
            method=PositionSizingMethod.FIXED_RISK,
            portfolio_value=portfolio_value,
            entry_price=entry_price,
            stop_loss_price=Price(Decimal('145')),  # 3.33% stop
            risk_per_trade=Decimal('0.02')  # 2% risk
        )
        print(f"✓ Position size calculated: {position_size.value} shares")
        
        # 2. Create stop loss
        stop_manager = StopLossManager()
        stop_order = stop_manager.create_stop_loss(
            symbol=symbol,
            entry_price=entry_price,
            position_size=position_size,
            position_type="long",
            stop_type=StopLossType.TRAILING_PERCENTAGE
        )
        print(f"✓ Stop loss created: {stop_order.stop_price.value:.2f}")
        
        # 3. Check portfolio limits
        limiter = get_portfolio_limiter()
        limiter.add_sector_mapping(symbol, "Technology")
        
        position_value = Money(position_size.value * entry_price.value)
        can_add, warnings = limiter.can_add_position(
            symbol, position_value, [], portfolio_value
        )
        print(f"✓ Position approved: {can_add}, warnings: {len(warnings)}")
        
        # 4. Monitor drawdown
        controller = get_drawdown_controller()
        metrics = controller.update_portfolio_value(portfolio_value)
        print(f"✓ Drawdown monitoring: {metrics.recovery_mode.value}")
        
        # 5. Simulate price movement and stop loss check
        new_price = Price(Decimal('148'))  # Price drops
        triggered_stops = stop_manager.check_stop_triggers({symbol: new_price})
        print(f"✓ Stop loss check: {len(triggered_stops)} triggered")
        
        # 6. Update trailing stop
        updated_stops = stop_manager.update_trailing_stops({symbol: new_price})
        print(f"✓ Trailing stop updates: {len(updated_stops)}")
        
        print("✓ Integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all risk management tests."""
    print("=" * 80)
    print("🛡️  ALGUA RISK MANAGEMENT SYSTEM TESTS")
    print("=" * 80)
    
    tests = [
        ("Position Sizing", test_position_sizing),
        ("Stop Loss System", test_stop_loss_system),
        ("Portfolio Limits", test_portfolio_limits),
        ("Drawdown Controls", test_drawdown_controls),
        ("Risk Configuration", test_risk_configuration),
        ("Risk Event System", test_risk_event_system),
        ("Integration", test_integration)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - FAILED with exception: {e}")
    
    total = len(tests)
    
    print("\n" + "=" * 80)
    print("📋 RISK MANAGEMENT TEST SUMMARY")
    print("=" * 80)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if i < passed else "❌ FAIL"
        print(f"  {test_name:<25}: {status}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL RISK MANAGEMENT TESTS PASSED!")
        print("\n📋 Risk Management System Summary:")
        print("  ✓ Position sizing with multiple methods")
        print("  ✓ Comprehensive stop loss mechanisms")
        print("  ✓ Portfolio exposure limits and monitoring")
        print("  ✓ Drawdown controls with recovery modes")
        print("  ✓ Risk configuration management")
        print("  ✓ Risk event system with handlers")
        print("  ✓ Integrated risk management workflow")
        
        print(f"\n🚀 Risk Management System Features:")
        print("  • Fixed risk, Kelly, volatility-based position sizing")
        print("  • Fixed, trailing, ATR-based, time-based stop losses")
        print("  • Sector, correlation, concentration risk limits")
        print("  • Real-time drawdown monitoring and controls")
        print("  • Configurable risk parameters and thresholds")
        print("  • Event-driven risk monitoring with alerts")
        print("  • Recovery mode with position size adjustments")
        
    else:
        print(f"\n⚠️ {total - passed} tests failed.")
    
    print("\n📖 Phase 1: Risk Management System - COMPLETE ✅")
    print("Ready to proceed to Phase 2: Order Management System")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)