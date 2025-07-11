"""
Simple test script for the error handling system.
"""

import sys
import os
from datetime import datetime

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic error handling functionality."""
    print("Testing basic error handling functionality...")
    
    try:
        # Test imports
        from error_handling import (
            ErrorManager, ErrorSeverity, ErrorCategory, ErrorType,
            TradingError, get_error_manager
        )
        print("✓ Successfully imported error handling components")
        
        # Test error manager
        error_manager = get_error_manager()
        print("✓ Error manager initialized")
        
        # Test error capture
        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = error_manager.capture_error(
                exception=e,
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.TRADING,
                error_type=ErrorType.INSUFFICIENT_FUNDS
            )
            print(f"✓ Error captured: {context.error_id}")
        
        # Test error statistics
        stats = error_manager.get_error_statistics()
        print(f"✓ Error statistics: {stats['total_errors']} total errors")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in basic functionality test: {e}")
        return False

def test_alert_system():
    """Test alert system functionality."""
    print("\nTesting alert system...")
    
    try:
        from error_handling import (
            AlertManager, AlertSeverity, AlertType, get_alert_manager
        )
        print("✓ Successfully imported alert components")
        
        # Test alert manager
        alert_manager = get_alert_manager()
        print("✓ Alert manager initialized")
        
        # Test alert creation
        alert = alert_manager.create_alert(
            title="Test Alert",
            message="This is a test alert",
            severity=AlertSeverity.HIGH,
            alert_type=AlertType.SYSTEM,
            auto_notify=False  # Disable notifications for testing
        )
        print(f"✓ Alert created: {alert.alert_id}")
        
        # Test alert acknowledgment
        success = alert_manager.acknowledge_alert(alert.alert_id)
        print(f"✓ Alert acknowledged: {success}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in alert system test: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring functionality."""
    print("\nTesting performance monitoring...")
    
    try:
        from error_handling import (
            PerformanceMonitor, get_performance_monitor
        )
        print("✓ Successfully imported performance monitoring components")
        
        # Test performance monitor
        perf_monitor = get_performance_monitor()
        print("✓ Performance monitor initialized")
        
        # Test latency recording
        perf_monitor.record_operation_latency(
            operation="test_operation",
            duration=25.5,
            component="test_component"
        )
        print("✓ Latency recorded")
        
        # Test throughput recording
        perf_monitor.record_throughput_event("test_events", 5)
        print("✓ Throughput recorded")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in performance monitoring test: {e}")
        return False

def test_logging_system():
    """Test logging system functionality."""
    print("\nTesting logging system...")
    
    try:
        from utils.logging import setup_logging, get_enhanced_logger, LogCategory
        print("✓ Successfully imported logging components")
        
        # Setup logging
        setup_logging()
        print("✓ Logging system initialized")
        
        # Test enhanced logger
        logger = get_enhanced_logger(__name__, LogCategory.SYSTEM)
        logger.info("Test log message")
        print("✓ Enhanced logger working")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in logging system test: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("ALGUA ERROR HANDLING SYSTEM - BASIC INTEGRATION TEST")
    print("="*60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Alert System", test_alert_system),
        ("Performance Monitoring", test_performance_monitoring),
        ("Logging System", test_logging_system)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            results.append((test_name, f"ERROR: {str(e)}"))
            print(f"✗ {test_name}: ERROR - {str(e)}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result == "PASSED")
    total = len(results)
    
    for test_name, result in results:
        status_symbol = "✓" if result == "PASSED" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All basic tests passed! Error handling system is functional.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)