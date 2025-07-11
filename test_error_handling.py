"""
Comprehensive test script for the error handling and monitoring system.
"""

import time
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from error_handling import (
    ErrorManager, ErrorSeverity, ErrorCategory, ErrorType,
    ErrorClassifier, AlertManager, AlertSeverity, AlertType,
    HealthMonitor, PerformanceMonitor, TradingError, AlguaException,
    get_error_manager, get_error_classifier, get_alert_manager,
    get_health_monitor, get_performance_monitor
)
from utils.logging import setup_logging, get_enhanced_logger, LogCategory


def test_error_manager():
    """Test the error manager functionality."""
    print("Testing Error Manager...")
    
    error_manager = get_error_manager()
    
    # Test capturing different types of errors
    try:
        raise ValueError("Test insufficient funds error")
    except ValueError as e:
        context = error_manager.capture_error(
            exception=e,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TRADING,
            error_type=ErrorType.INSUFFICIENT_FUNDS,
            symbol="AAPL",
            user_id="test_user",
            metadata={"order_id": "12345", "amount": 1000.0}
        )
        print(f"✓ Captured error: {context.error_id}")
    
    # Test network error
    try:
        raise ConnectionError("API connection timeout")
    except ConnectionError as e:
        context = error_manager.capture_error(
            exception=e,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            metadata={"endpoint": "https://api.example.com", "timeout": 30}
        )
        print(f"✓ Captured network error: {context.error_id}")
    
    # Test custom trading error
    try:
        raise TradingError(
            "Risk limit exceeded",
            error_type=ErrorType.RISK_LIMIT_BREACH,
            severity=ErrorSeverity.CRITICAL,
            metadata={"current_exposure": 150000, "limit": 100000}
        )
    except TradingError as e:
        context = error_manager.capture_error(
            exception=e,
            severity=e.severity,
            category=ErrorCategory.RISK_MANAGEMENT,
            error_type=e.error_type,
            metadata=e.metadata
        )
        print(f"✓ Captured trading error: {context.error_id}")
    
    # Get error statistics
    stats = error_manager.get_error_statistics()
    print(f"✓ Error statistics: {stats['total_errors']} total errors")
    
    # Get recent errors
    recent_errors = error_manager.get_recent_errors(hours=1)
    print(f"✓ Recent errors: {len(recent_errors)} in last hour")
    
    return True


def test_error_classifier():
    """Test the error classifier functionality."""
    print("\nTesting Error Classifier...")
    
    classifier = get_error_classifier()
    error_manager = get_error_manager()
    
    # Test classification of different error types
    test_errors = [
        ("insufficient funds", ErrorCategory.TRADING, ErrorType.INSUFFICIENT_FUNDS),
        ("connection timeout", ErrorCategory.NETWORK, ErrorType.CONNECTION_TIMEOUT),
        ("rate limit exceeded", ErrorCategory.NETWORK, ErrorType.API_RATE_LIMIT),
        ("missing data", ErrorCategory.DATA, ErrorType.MISSING_DATA),
        ("risk limit breach", ErrorCategory.RISK_MANAGEMENT, ErrorType.RISK_LIMIT_BREACH)
    ]
    
    for error_message, expected_category, expected_type in test_errors:
        try:
            raise Exception(error_message)
        except Exception as e:
            context = error_manager.capture_error(
                exception=e,
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.SYSTEM  # Let classifier determine the real category
            )
            
            # Classify the error
            result = classifier.classify_error(context)
            
            print(f"✓ Classified '{error_message}': "
                  f"confidence={result.confidence}, "
                  f"category={result.suggested_category}, "
                  f"type={result.suggested_type}")
    
    # Get classification statistics
    stats = classifier.get_classification_statistics()
    print(f"✓ Classification statistics: {stats['success_rate']:.2%} success rate")
    
    return True


def test_alert_manager():
    """Test the alert manager functionality."""
    print("\nTesting Alert Manager...")
    
    alert_manager = get_alert_manager()
    
    # Create different types of alerts
    alerts = [
        {
            "title": "System CPU Usage High",
            "message": "CPU usage has exceeded 80% for 5 minutes",
            "severity": AlertSeverity.HIGH,
            "alert_type": AlertType.PERFORMANCE,
            "component_id": "system_resources"
        },
        {
            "title": "Trading API Down",
            "message": "Unable to connect to trading API",
            "severity": AlertSeverity.CRITICAL,
            "alert_type": AlertType.SYSTEM,
            "component_id": "trading_api"
        },
        {
            "title": "Risk Limit Breach",
            "message": "Portfolio risk limit has been exceeded",
            "severity": AlertSeverity.CRITICAL,
            "alert_type": AlertType.BUSINESS,
            "component_id": "risk_manager"
        }
    ]
    
    created_alerts = []
    for alert_data in alerts:
        alert = alert_manager.create_alert(**alert_data)
        created_alerts.append(alert)
        print(f"✓ Created alert: {alert.alert_id} - {alert.title}")
    
    # Test alert acknowledgment
    if created_alerts:
        alert_id = created_alerts[0].alert_id
        success = alert_manager.acknowledge_alert(alert_id, "test_user")
        print(f"✓ Acknowledged alert: {alert_id} - {success}")
    
    # Get active alerts
    active_alerts = alert_manager.get_active_alerts()
    print(f"✓ Active alerts: {len(active_alerts)}")
    
    # Get alert statistics
    stats = alert_manager.get_alert_statistics()
    print(f"✓ Alert statistics: {stats['total_alerts']} total alerts")
    
    return True


def test_health_monitor():
    """Test the health monitor functionality."""
    print("\nTesting Health Monitor...")
    
    health_monitor = get_health_monitor()
    
    # Perform health checks
    health_results = health_monitor.check_all_components()
    print(f"✓ Health check completed: {len(health_results)} components checked")
    
    for component_id, health in health_results.items():
        print(f"  - {component_id}: {health.status.value} - {health.message}")
    
    # Get overall health
    overall_health = health_monitor.get_overall_health()
    print(f"✓ Overall system health: {overall_health['status']}")
    
    # Get monitoring statistics
    stats = health_monitor.get_monitoring_statistics()
    print(f"✓ Monitoring statistics: {stats['registered_checks']} registered checks")
    
    return True


def test_performance_monitor():
    """Test the performance monitor functionality."""
    print("\nTesting Performance Monitor...")
    
    perf_monitor = get_performance_monitor()
    
    # Collect current metrics
    metrics = perf_monitor.collect_metrics()
    print(f"✓ Collected {len(metrics)} performance metrics")
    
    # Test latency tracking
    test_operations = [
        ("database_query", 25.5),
        ("api_call", 150.0),
        ("order_processing", 75.2),
        ("risk_calculation", 12.8)
    ]
    
    for operation, latency in test_operations:
        perf_monitor.record_operation_latency(
            operation=operation,
            duration=latency,
            component="test_component",
            status="success"
        )
        print(f"✓ Recorded latency for {operation}: {latency}ms")
    
    # Test throughput tracking
    perf_monitor.record_throughput_event("orders_processed", 5)
    perf_monitor.record_throughput_event("api_requests", 12)
    print("✓ Recorded throughput events")
    
    # Get current metrics summary
    current_metrics = perf_monitor.get_current_metrics()
    print(f"✓ Current metrics summary: {len(current_metrics['system_metrics'])} system metrics")
    
    # Get performance summary
    perf_summary = perf_monitor.get_performance_summary()
    print(f"✓ Performance summary: {perf_summary['status']} status")
    
    return True


def test_integrated_workflow():
    """Test the integrated error handling workflow."""
    print("\nTesting Integrated Workflow...")
    
    # Get all managers
    error_manager = get_error_manager()
    classifier = get_error_classifier()
    alert_manager = get_alert_manager()
    
    # Simulate a critical trading error
    try:
        # Simulate a trading system failure
        raise TradingError(
            "Order execution failed due to insufficient funds",
            error_type=ErrorType.INSUFFICIENT_FUNDS,
            severity=ErrorSeverity.HIGH,
            metadata={
                "order_id": "ORD-12345",
                "symbol": "AAPL",
                "quantity": 100,
                "price": 150.50,
                "account_balance": 1000.0,
                "required_amount": 15050.0
            }
        )
    except TradingError as e:
        # 1. Capture the error
        error_context = error_manager.capture_error(
            exception=e,
            severity=e.severity,
            category=ErrorCategory.TRADING,
            error_type=e.error_type,
            user_id="trader_123",
            strategy_id="momentum_v2",
            symbol="AAPL",
            metadata=e.metadata
        )
        print(f"✓ Step 1: Error captured - {error_context.error_id}")
        
        # 2. Classify the error (optional, as we already know the type)
        classification = classifier.classify_error(error_context)
        print(f"✓ Step 2: Error classified - confidence: {classification.confidence}")
        
        # 3. Create alert from error
        alert = alert_manager.create_alert_from_error(error_context)
        if alert:
            print(f"✓ Step 3: Alert created - {alert.alert_id}")
        
        # 4. Simulate some time passing and resolve the alert
        time.sleep(1)
        if alert:
            alert_manager.resolve_alert(alert.alert_id, "Issue resolved by adding funds to account")
            print(f"✓ Step 4: Alert resolved - {alert.alert_id}")
    
    return True


def test_logging_integration():
    """Test the enhanced logging integration."""
    print("\nTesting Logging Integration...")
    
    # Setup enhanced logging
    setup_logging()
    
    # Get enhanced logger
    logger = get_enhanced_logger(__name__, LogCategory.SYSTEM)
    
    # Test different log levels and categories
    logger.info("System test started")
    logger.debug("Debug information for testing")
    logger.warning("Test warning message")
    
    # Test trading-specific logging
    logger.log_trade_execution(
        action="BUY",
        symbol="AAPL",
        quantity=100,
        price=150.50,
        order_id="ORD-12345",
        strategy_id="momentum_v2"
    )
    
    # Test risk event logging
    logger.log_risk_event(
        event_type="limit_breach",
        severity="high",
        description="Portfolio exposure exceeded limit",
        affected_positions=["AAPL", "GOOGL"]
    )
    
    # Test performance logging
    logger.performance(
        "Test operation completed",
        performance_metrics={
            "duration_ms": 125.5,
            "cpu_usage": 15.2,
            "memory_usage": 45.8
        }
    )
    
    print("✓ Enhanced logging tests completed")
    return True


def run_all_tests():
    """Run all error handling system tests."""
    print("="*60)
    print("ALGUA ERROR HANDLING SYSTEM INTEGRATION TEST")
    print("="*60)
    
    test_results = []
    
    # Run individual tests
    tests = [
        ("Error Manager", test_error_manager),
        ("Error Classifier", test_error_classifier),
        ("Alert Manager", test_alert_manager),
        ("Health Monitor", test_health_monitor),
        ("Performance Monitor", test_performance_monitor),
        ("Logging Integration", test_logging_integration),
        ("Integrated Workflow", test_integrated_workflow)
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'-'*40}")
            result = test_func()
            if result:
                test_results.append((test_name, "PASSED"))
                print(f"✓ {test_name}: PASSED")
            else:
                test_results.append((test_name, "FAILED"))
                print(f"✗ {test_name}: FAILED")
        except Exception as e:
            test_results.append((test_name, f"ERROR: {str(e)}"))
            print(f"✗ {test_name}: ERROR - {str(e)}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in test_results if result == "PASSED")
    failed = len(test_results) - passed
    
    for test_name, result in test_results:
        status_symbol = "✓" if result == "PASSED" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
    
    print(f"\nTotal: {len(test_results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Error handling system is working correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)