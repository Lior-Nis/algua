"""
Integration between error handling and enhanced logging systems.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import threading

from .error_manager import ErrorContext, ErrorSeverity, ErrorCategory, ErrorType
from .health_monitor import ComponentHealth, HealthStatus
from .recovery_manager import RecoveryResult, RecoveryStatus
from utils.logging import get_enhanced_logger, LogCategory, LogLevel, set_correlation_id


class ErrorLoggingIntegration:
    """Integration layer between error handling and logging."""
    
    def __init__(self):
        self.logger = get_enhanced_logger(__name__, LogCategory.ERROR)
        self._lock = threading.Lock()
    
    def log_error_context(self, error_context: ErrorContext) -> None:
        """Log error with full context using enhanced logging."""
        with self._lock:
            # Set correlation ID if available
            if error_context.session_id:
                set_correlation_id(error_context.session_id)
            
            # Prepare error context for logging
            error_log_context = {
                'error_id': error_context.error_id,
                'severity': error_context.severity.value,
                'category': error_context.category.value,
                'error_type': error_context.error_type.value,
                'recoverable': error_context.recoverable,
                'recovery_attempted': error_context.recovery_attempted,
                'recovery_successful': error_context.recovery_successful,
                'module': error_context.module,
                'function': error_context.function,
                'line_number': error_context.line_number,
                'hostname': error_context.hostname,
                'process_id': error_context.process_id,
                'thread_id': error_context.thread_id,
                'memory_usage': error_context.memory_usage,
                'cpu_usage': error_context.cpu_usage
            }
            
            # Add business context if available
            business_context = {}
            if error_context.user_id:
                business_context['user_id'] = error_context.user_id
            if error_context.strategy_id:
                business_context['strategy_id'] = error_context.strategy_id
            if error_context.order_id:
                business_context['order_id'] = error_context.order_id
            if error_context.symbol:
                business_context['symbol'] = error_context.symbol
            
            if business_context:
                error_log_context['business_context'] = business_context
            
            # Map severity to log level
            level_map = {
                ErrorSeverity.INFO: LogLevel.INFO.value,
                ErrorSeverity.LOW: LogLevel.WARNING.value,
                ErrorSeverity.MEDIUM: LogLevel.ERROR.value,
                ErrorSeverity.HIGH: LogLevel.ERROR.value,
                ErrorSeverity.CRITICAL: LogLevel.CRITICAL.value
            }
            
            log_level = level_map.get(error_context.severity, LogLevel.ERROR.value)
            
            # Log based on category
            if error_context.category == ErrorCategory.TRADING:
                self.logger._log(
                    log_level,
                    f"Trading error: {error_context.message}",
                    category=LogCategory.TRADING,
                    error_context=error_log_context,
                    trading_context=business_context
                )
            elif error_context.category == ErrorCategory.RISK_MANAGEMENT:
                self.logger._log(
                    log_level,
                    f"Risk management error: {error_context.message}",
                    category=LogCategory.RISK,
                    error_context=error_log_context
                )
            elif error_context.category == ErrorCategory.PERFORMANCE:
                self.logger._log(
                    log_level,
                    f"Performance error: {error_context.message}",
                    category=LogCategory.PERFORMANCE,
                    error_context=error_log_context
                )
            else:
                self.logger._log(
                    log_level,
                    f"System error: {error_context.message}",
                    category=LogCategory.SYSTEM,
                    error_context=error_log_context
                )
            
            # Log stack trace if available
            if error_context.stack_trace and error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                self.logger.debug(
                    f"Stack trace for error {error_context.error_id}",
                    error_context={'stack_trace': error_context.stack_trace}
                )
    
    def log_recovery_attempt(self, recovery_result: RecoveryResult) -> None:
        """Log recovery attempt results."""
        with self._lock:
            recovery_context = {
                'recovery_id': recovery_result.recovery_id,
                'error_id': recovery_result.error_id,
                'strategy': recovery_result.action.strategy.value,
                'status': recovery_result.status.value,
                'attempts_made': recovery_result.attempts_made,
                'success': recovery_result.success,
                'error_resolved': recovery_result.error_resolved,
                'partial_recovery': recovery_result.partial_recovery,
                'total_duration': str(recovery_result.total_duration) if recovery_result.total_duration else None,
                'recovery_cost': float(recovery_result.recovery_cost)
            }
            
            if recovery_result.status == RecoveryStatus.SUCCESS:
                self.logger.info(
                    f"Recovery successful: {recovery_result.recovery_message}",
                    category=LogCategory.SYSTEM,
                    error_context=recovery_context
                )
            elif recovery_result.status == RecoveryStatus.FAILED:
                self.logger.error(
                    f"Recovery failed: {recovery_result.recovery_message}",
                    category=LogCategory.ERROR,
                    error_context=recovery_context
                )
            else:
                self.logger.info(
                    f"Recovery status: {recovery_result.status.value} - {recovery_result.recovery_message}",
                    category=LogCategory.SYSTEM,
                    error_context=recovery_context
                )
    
    def log_health_status_change(
        self,
        component_id: str,
        old_status: HealthStatus,
        new_status: HealthStatus,
        health: ComponentHealth
    ) -> None:
        """Log health status changes."""
        with self._lock:
            health_context = {
                'component_id': component_id,
                'component_type': health.component_type.value,
                'old_status': old_status.value,
                'new_status': new_status.value,
                'last_check': health.last_check.isoformat(),
                'message': health.message,
                'error_count': health.error_count,
                'metrics': {
                    name: {
                        'value': metric.value,
                        'status': metric.get_status().value
                    }
                    for name, metric in health.metrics.items()
                }
            }
            
            # Log level based on new status
            if new_status == HealthStatus.CRITICAL:
                level = LogLevel.CRITICAL.value
            elif new_status == HealthStatus.WARNING:
                level = LogLevel.WARNING.value
            elif new_status == HealthStatus.DEGRADED:
                level = LogLevel.WARNING.value
            else:
                level = LogLevel.INFO.value
            
            self.logger._log(
                level,
                f"Health status change for {component_id}: {old_status.value} -> {new_status.value}",
                category=LogCategory.SYSTEM,
                error_context=health_context
            )
    
    def log_system_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log system-wide metrics."""
        with self._lock:
            self.logger.info(
                "System metrics update",
                category=LogCategory.PERFORMANCE,
                performance_metrics=metrics
            )


class AuditLogger:
    """Specialized logger for audit trail and compliance."""
    
    def __init__(self):
        self.logger = get_enhanced_logger(__name__, LogCategory.AUDIT)
        self._lock = threading.Lock()
    
    def log_user_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log user actions for audit trail."""
        with self._lock:
            audit_context = {
                'user_id': user_id,
                'action': action,
                'resource': resource,
                'success': success,
                'timestamp': datetime.utcnow().isoformat(),
                'details': details or {}
            }
            
            level = LogLevel.AUDIT.value
            message = f"User {user_id} {action} on {resource}: {'SUCCESS' if success else 'FAILED'}"
            
            self.logger._log(
                level,
                message,
                category=LogCategory.AUDIT,
                error_context=audit_context
            )
    
    def log_trading_decision(
        self,
        strategy_id: str,
        symbol: str,
        decision: str,
        reasoning: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log trading decisions for compliance."""
        with self._lock:
            audit_context = {
                'strategy_id': strategy_id,
                'symbol': symbol,
                'decision': decision,
                'reasoning': reasoning,
                'confidence': confidence,
                'timestamp': datetime.utcnow().isoformat(),
                'metadata': metadata or {}
            }
            
            self.logger.audit(
                f"Trading decision: {decision} for {symbol} by {strategy_id}",
                trading_context=audit_context
            )
    
    def log_risk_threshold_change(
        self,
        user_id: str,
        threshold_type: str,
        old_value: float,
        new_value: float,
        reason: str
    ) -> None:
        """Log risk threshold changes."""
        with self._lock:
            audit_context = {
                'user_id': user_id,
                'threshold_type': threshold_type,
                'old_value': old_value,
                'new_value': new_value,
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.audit(
                f"Risk threshold changed: {threshold_type} from {old_value} to {new_value}",
                error_context=audit_context
            )
    
    def log_system_configuration_change(
        self,
        user_id: str,
        config_key: str,
        old_value: Any,
        new_value: Any,
        reason: Optional[str] = None
    ) -> None:
        """Log system configuration changes."""
        with self._lock:
            audit_context = {
                'user_id': user_id,
                'config_key': config_key,
                'old_value': str(old_value),
                'new_value': str(new_value),
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self.logger.audit(
                f"Configuration changed: {config_key} from {old_value} to {new_value}",
                error_context=audit_context
            )


# Global instances
_error_logging_integration = None
_audit_logger = None


def get_error_logging_integration() -> ErrorLoggingIntegration:
    """Get global error logging integration instance."""
    global _error_logging_integration
    if _error_logging_integration is None:
        _error_logging_integration = ErrorLoggingIntegration()
    return _error_logging_integration


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger