#!/usr/bin/env python3
"""
Demonstration of the enhanced logging system capabilities.
"""

import time
import uuid
from decimal import Decimal
from datetime import datetime

from utils.logging import (
    LogConfig, setup_logging, get_enhanced_logger, get_log_manager,
    LogCategory, performance_logging, set_correlation_id
)
from error_handling import (
    get_error_manager, get_error_logging_integration, get_audit_logger,
    ErrorSeverity, ErrorCategory, ErrorType, TradingError
)


def main():
    """Demonstrate enhanced logging capabilities."""
    print("=== Enhanced Logging System Demo ===\n")
    
    # 1. Basic logging setup
    print("1. Setting up enhanced logging...")
    config = LogConfig(
        level="DEBUG",
        format_type="json",
        enable_performance_logging=True,
        enable_audit_logging=True,
        enable_trading_logging=True
    )
    setup_logging(config)
    print("✓ Enhanced logging configured\n")
    
    # 2. Basic enhanced logging
    print("2. Testing basic enhanced logging...")
    logger = get_enhanced_logger(__name__, LogCategory.SYSTEM)
    
    # Set correlation ID for request tracking
    correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    
    logger.info("System startup initiated", system_version="1.0.0", environment="demo")
    logger.debug("Debug information", module="logging_demo", function="main")
    logger.warning("This is a warning message", component="demo")
    print("✓ Basic logging completed\n")
    
    # 3. Trading-specific logging
    print("3. Testing trading-specific logging...")
    trading_logger = get_enhanced_logger("trading", LogCategory.TRADING)
    
    # Log trade execution
    trading_logger.log_trade_execution(
        action="BUY",
        symbol="AAPL",
        quantity=100,
        price=Decimal("150.25"),
        order_id="ORD123",
        strategy_id="RSI_MEAN_REVERSION"
    )
    
    # Log order updates
    trading_logger.log_order_update(
        order_id="ORD123",
        status="FILLED",
        symbol="AAPL",
        fill_price=150.30,
        fill_quantity=100
    )
    
    # Log portfolio update
    trading_logger.log_portfolio_update(
        total_value=Decimal("50000.00"),
        daily_pnl=Decimal("250.50"),
        positions_count=5
    )
    
    # Log strategy signal
    trading_logger.log_strategy_signal(
        strategy_name="RSI_MEAN_REVERSION",
        signal="BUY",
        symbol="AAPL",
        confidence=0.85,
        rsi_value=25.3,
        signal_strength="STRONG"
    )
    
    print("✓ Trading logging completed\n")
    
    # 4. Performance logging with decorator
    print("4. Testing performance logging...")
    
    @performance_logging(include_args=True)
    def cpu_intensive_task(iterations: int = 1000000):
        """Simulate CPU-intensive task."""
        total = 0
        for i in range(iterations):
            total += i ** 0.5
        return total
    
    @performance_logging()
    def io_simulation():
        """Simulate I/O operation."""
        time.sleep(0.1)  # Simulate I/O delay
        return "I/O completed"
    
    # Execute performance-monitored functions
    result1 = cpu_intensive_task(500000)
    result2 = io_simulation()
    
    print("✓ Performance logging completed\n")
    
    # 5. Error logging integration
    print("5. Testing error logging integration...")
    error_manager = get_error_manager()
    error_integration = get_error_logging_integration()
    
    try:
        # Simulate a trading error
        raise TradingError(
            "Insufficient funds for order",
            error_type=ErrorType.INSUFFICIENT_FUNDS,
            severity=ErrorSeverity.HIGH,
            metadata={
                "order_id": "ORD124",
                "symbol": "TSLA",
                "requested_quantity": 50,
                "available_funds": 5000.0,
                "required_funds": 7500.0
            }
        )
    except TradingError as e:
        # Capture and log the error
        error_context = error_manager.capture_error(
            exception=e,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TRADING,
            user_id="user123",
            session_id=correlation_id,
            strategy_id="MOMENTUM_STRATEGY",
            order_id="ORD124",
            symbol="TSLA"
        )
        
        # Log through integration layer
        error_integration.log_error_context(error_context)
    
    print("✓ Error logging integration completed\n")
    
    # 6. Audit logging
    print("6. Testing audit logging...")
    audit_logger = get_audit_logger()
    
    # Log user actions
    audit_logger.log_user_action(
        user_id="user123",
        action="LOGIN",
        resource="trading_platform",
        success=True,
        details={"ip_address": "192.168.1.100", "user_agent": "TradingApp/1.0"}
    )
    
    # Log trading decision
    audit_logger.log_trading_decision(
        strategy_id="RSI_MEAN_REVERSION",
        symbol="AAPL",
        decision="BUY",
        reasoning="RSI below 30, indicating oversold condition",
        confidence=0.85,
        metadata={"rsi_value": 25.3, "price": 150.25}
    )
    
    # Log risk threshold change
    audit_logger.log_risk_threshold_change(
        user_id="admin",
        threshold_type="max_position_size",
        old_value=10000.0,
        new_value=15000.0,
        reason="Increased risk appetite due to market conditions"
    )
    
    # Log configuration change
    audit_logger.log_system_configuration_change(
        user_id="admin",
        config_key="trading.max_orders_per_minute",
        old_value=10,
        new_value=15,
        reason="Performance optimization"
    )
    
    print("✓ Audit logging completed\n")
    
    # 7. Risk event logging
    print("7. Testing risk event logging...")
    risk_logger = get_enhanced_logger("risk_manager", LogCategory.RISK)
    
    risk_logger.log_risk_event(
        event_type="POSITION_LIMIT_BREACH",
        severity="high",
        description="Position size exceeded maximum allowed limit",
        affected_positions=["AAPL", "TSLA"],
        current_exposure=25000.0,
        limit=20000.0,
        breach_amount=5000.0
    )
    
    print("✓ Risk event logging completed\n")
    
    # 8. Log management operations
    print("8. Testing log management...")
    log_manager = get_log_manager()
    
    # Get logging statistics
    stats = log_manager.get_log_statistics()
    print(f"Log statistics: {stats}")
    
    # Analyze log patterns (placeholder)
    patterns = log_manager.analyze_log_patterns(hours=1)
    print(f"Log patterns: {patterns}")
    
    print("✓ Log management completed\n")
    
    # 9. Custom structured logging
    print("9. Testing custom structured logging...")
    custom_logger = get_enhanced_logger("custom", LogCategory.SYSTEM)
    
    # Log with custom fields
    custom_logger.info(
        "Custom event occurred",
        category=LogCategory.SYSTEM,
        event_type="CUSTOM_EVENT",
        severity="INFO",
        custom_field1="value1",
        custom_field2=42,
        custom_field3={"nested": "data"}
    )
    
    # Log with performance metrics
    custom_logger.performance(
        "Custom performance measurement",
        performance_metrics={
            "execution_time_ms": 150.5,
            "memory_usage_mb": 45.2,
            "cpu_usage_percent": 25.8,
            "custom_metric": "value"
        }
    )
    
    print("✓ Custom structured logging completed\n")
    
    print("=== Demo Completed Successfully ===")
    print("\nCheck the logs/ directory for generated log files:")
    print("- algua.log (main application log)")
    print("- trading.log (trading-specific events)")
    print("- performance.log (performance metrics)")
    print("- audit.log (audit trail)")
    print("- errors.log (error events)")


if __name__ == "__main__":
    main()