"""
Error handling and monitoring system for Algua trading platform.

This module provides comprehensive error handling, monitoring, and recovery
functionality including:
- Error classification and context capture
- Error propagation and recovery strategies  
- System health monitoring and alerting
- Performance monitoring and diagnostics
- Graceful degradation and failover mechanisms
"""

# Core error handling
from .error_manager import (
    ErrorManager, ErrorSeverity, ErrorCategory, ErrorContext,
    TradingError, AlguaException, get_error_manager
)

# Error classification and handling
from .error_classifier import (
    ErrorClassifier, ErrorType, ErrorPattern,
    ClassificationResult, get_error_classifier
)

# Recovery strategies
from .recovery_manager import (
    RecoveryManager, RecoveryStrategy, RecoveryAction,
    RecoveryResult, get_recovery_manager
)

# Logging integration
from .logging_integration import (
    ErrorLoggingIntegration, AuditLogger,
    get_error_logging_integration, get_audit_logger
)

# System health monitoring
from .health_monitor import (
    HealthMonitor, HealthStatus, ComponentHealth,
    HealthCheck, get_health_monitor
)

# Performance monitoring
from .performance_monitor import (
    PerformanceMonitor, PerformanceMetrics, LatencyTracker,
    ThroughputMonitor, get_performance_monitor
)

# Alerting system
from .alert_manager import (
    AlertManager, Alert, AlertSeverity, AlertType, AlertChannel,
    AlertRule, get_alert_manager
)

__all__ = [
    # Core error handling
    "ErrorManager", "ErrorSeverity", "ErrorCategory", "ErrorContext",
    "TradingError", "AlguaException", "get_error_manager",
    
    # Error classification
    "ErrorClassifier", "ErrorType", "ErrorPattern",
    "ClassificationResult", "get_error_classifier",
    
    # Recovery management
    "RecoveryManager", "RecoveryStrategy", "RecoveryAction",
    "RecoveryResult", "get_recovery_manager",
    
    # Logging integration
    "ErrorLoggingIntegration", "AuditLogger",
    "get_error_logging_integration", "get_audit_logger",
    
    # Health monitoring
    "HealthMonitor", "HealthStatus", "ComponentHealth",
    "HealthCheck", "get_health_monitor",
    
    # Performance monitoring
    "PerformanceMonitor", "PerformanceMetrics", "LatencyTracker",
    "ThroughputMonitor", "get_performance_monitor",
    
    # Alerting
    "AlertManager", "Alert", "AlertSeverity", "AlertType", "AlertChannel",
    "AlertRule", "get_alert_manager"
]