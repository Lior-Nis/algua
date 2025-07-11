"""
Core error management system for Algua trading platform.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import traceback
import sys
import threading
import uuid
from pathlib import Path
import json

from utils.logging import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"      # System-threatening errors requiring immediate attention
    HIGH = "high"             # Errors affecting trading operations
    MEDIUM = "medium"         # Errors affecting non-critical functionality
    LOW = "low"              # Minor errors and warnings
    INFO = "info"            # Informational messages


class ErrorCategory(Enum):
    """Error categories for classification."""
    TRADING = "trading"              # Trading execution errors
    RISK_MANAGEMENT = "risk"         # Risk management violations
    DATA = "data"                   # Data quality and connectivity issues
    SYSTEM = "system"               # System and infrastructure errors
    CONFIGURATION = "configuration" # Configuration and setup errors
    VALIDATION = "validation"       # Input validation errors
    NETWORK = "network"             # Network and connectivity errors
    AUTHENTICATION = "authentication" # Authentication and authorization errors
    PERFORMANCE = "performance"     # Performance and timeout errors
    BUSINESS_LOGIC = "business"     # Business logic errors


class ErrorType(Enum):
    """Specific error types."""
    # Trading errors
    ORDER_REJECTION = "order_rejection"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    MARKET_CLOSED = "market_closed"
    INVALID_SYMBOL = "invalid_symbol"
    
    # Risk management errors
    RISK_LIMIT_BREACH = "risk_limit_breach"
    STOP_LOSS_FAILURE = "stop_loss_failure"
    PORTFOLIO_LIMIT_EXCEEDED = "portfolio_limit_exceeded"
    DRAWDOWN_LIMIT_EXCEEDED = "drawdown_limit_exceeded"
    
    # Data errors
    MISSING_DATA = "missing_data"
    STALE_DATA = "stale_data"
    DATA_CORRUPTION = "data_corruption"
    FEED_DISCONNECTION = "feed_disconnection"
    
    # System errors
    DATABASE_ERROR = "database_error"
    FILE_SYSTEM_ERROR = "file_system_error"
    MEMORY_ERROR = "memory_error"
    CPU_OVERLOAD = "cpu_overload"
    
    # Network errors
    CONNECTION_TIMEOUT = "connection_timeout"
    API_RATE_LIMIT = "api_rate_limit"
    NETWORK_UNAVAILABLE = "network_unavailable"
    
    # Configuration errors
    INVALID_CONFIG = "invalid_config"
    MISSING_CONFIG = "missing_config"
    CONFIG_VALIDATION_ERROR = "config_validation_error"


@dataclass
class ErrorContext:
    """Context information for an error."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    error_type: ErrorType
    message: str
    
    # Technical details
    exception: Optional[Exception] = None
    stack_trace: Optional[str] = None
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    
    # Business context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    strategy_id: Optional[str] = None
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    
    # System context
    hostname: Optional[str] = None
    process_id: Optional[int] = None
    thread_id: Optional[int] = None
    memory_usage: Optional[int] = None
    cpu_usage: Optional[float] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_errors: List[str] = field(default_factory=list)
    
    # Recovery information
    recoverable: bool = True
    recovery_attempted: bool = False
    recovery_successful: Optional[bool] = None
    recovery_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'category': self.category.value,
            'error_type': self.error_type.value,
            'message': self.message,
            'exception': str(self.exception) if self.exception else None,
            'stack_trace': self.stack_trace,
            'module': self.module,
            'function': self.function,
            'line_number': self.line_number,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'strategy_id': self.strategy_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'hostname': self.hostname,
            'process_id': self.process_id,
            'thread_id': self.thread_id,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'metadata': self.metadata,
            'related_errors': self.related_errors,
            'recoverable': self.recoverable,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'recovery_actions': self.recovery_actions
        }


class TradingError(Exception):
    """Base exception for trading-related errors."""
    
    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.ORDER_REJECTION,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        recoverable: bool = True,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.severity = severity
        self.recoverable = recoverable
        self.metadata = metadata or {}
        self.timestamp = datetime.now()


class AlguaException(Exception):
    """Base exception for Algua platform errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recoverable: bool = True,
        metadata: Dict[str, Any] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.recoverable = recoverable
        self.metadata = metadata or {}
        self.timestamp = datetime.now()


class ErrorManager:
    """Central error management and coordination system."""
    
    def __init__(
        self,
        max_error_history: int = 10000,
        error_log_path: Optional[Path] = None
    ):
        self.max_error_history = max_error_history
        self.error_log_path = error_log_path or Path("logs/errors.jsonl")
        
        # Error storage
        self.error_history: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = {}
        self.recent_errors: Dict[str, datetime] = {}
        
        # Error handlers
        self.error_handlers: Dict[ErrorCategory, List[Callable]] = {}
        self.recovery_handlers: Dict[ErrorType, Callable] = {}
        
        # Configuration
        self.deduplication_window = timedelta(minutes=5)
        self.max_recovery_attempts = 3
        self.critical_error_threshold = 10
        
        # Statistics
        self.error_stats = {
            'total_errors': 0,
            'errors_by_severity': {severity.value: 0 for severity in ErrorSeverity},
            'errors_by_category': {category.value: 0 for category in ErrorCategory},
            'recovery_success_rate': 0.0,
            'avg_resolution_time': 0.0
        }
        
        self._lock = threading.Lock()
        
        # Ensure log directory exists
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Error manager initialized")
    
    def capture_error(
        self,
        exception: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        error_type: Optional[ErrorType] = None,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> ErrorContext:
        """
        Capture and process an error with full context.
        
        Args:
            exception: The exception that occurred
            severity: Error severity level
            category: Error category
            error_type: Specific error type
            message: Custom error message
            metadata: Additional metadata
            user_id: User associated with error
            session_id: Session ID
            strategy_id: Strategy ID
            order_id: Order ID
            symbol: Trading symbol
            
        Returns:
            ErrorContext object
        """
        with self._lock:
            # Generate unique error ID
            error_id = str(uuid.uuid4())
            
            # Extract stack trace information
            tb = traceback.extract_tb(exception.__traceback__) if exception.__traceback__ else None
            stack_trace = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
            
            # Get caller information
            frame = sys._getframe(1) if sys._getframe else None
            module_name = frame.f_globals.get('__name__') if frame else None
            function_name = frame.f_code.co_name if frame else None
            line_number = frame.f_lineno if frame else None
            
            # Determine error type if not provided
            if error_type is None:
                error_type = self._infer_error_type(exception, category)
            
            # Create error context
            context = ErrorContext(
                error_id=error_id,
                timestamp=datetime.now(),
                severity=severity,
                category=category,
                error_type=error_type,
                message=message or str(exception),
                exception=exception,
                stack_trace=stack_trace,
                module=module_name,
                function=function_name,
                line_number=line_number,
                user_id=user_id,
                session_id=session_id,
                strategy_id=strategy_id,
                order_id=order_id,
                symbol=symbol,
                metadata=metadata or {},
                recoverable=self._is_recoverable(exception, error_type)
            )
            
            # Add system context
            self._add_system_context(context)
            
            # Check for duplicate errors
            if self._is_duplicate_error(context):
                self._update_duplicate_error(context)
                return context
            
            # Store error
            self._store_error(context)
            
            # Update statistics
            self._update_error_statistics(context)
            
            # Log error
            self._log_error(context)
            
            # Notify handlers
            self._notify_error_handlers(context)
            
            # Attempt recovery if appropriate
            if context.recoverable and severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                self._attempt_recovery(context)
            
            logger.error(
                f"Error captured: {error_id} - {severity.value} - {category.value} - {message or str(exception)}"
            )
            
            return context
    
    def register_error_handler(
        self,
        category: ErrorCategory,
        handler: Callable[[ErrorContext], None]
    ) -> None:
        """Register an error handler for a specific category."""
        if category not in self.error_handlers:
            self.error_handlers[category] = []
        self.error_handlers[category].append(handler)
        logger.info(f"Registered error handler for category: {category.value}")
    
    def register_recovery_handler(
        self,
        error_type: ErrorType,
        handler: Callable[[ErrorContext], bool]
    ) -> None:
        """Register a recovery handler for a specific error type."""
        self.recovery_handlers[error_type] = handler
        logger.info(f"Registered recovery handler for error type: {error_type.value}")
    
    def get_error_by_id(self, error_id: str) -> Optional[ErrorContext]:
        """Get error by ID."""
        with self._lock:
            for error in self.error_history:
                if error.error_id == error_id:
                    return error
            return None
    
    def get_recent_errors(
        self,
        hours: int = 24,
        severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None
    ) -> List[ErrorContext]:
        """Get recent errors within specified time window."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent = [
                error for error in self.error_history
                if error.timestamp >= cutoff_time
            ]
            
            if severity:
                recent = [error for error in recent if error.severity == severity]
            
            if category:
                recent = [error for error in recent if error.category == category]
            
            return sorted(recent, key=lambda x: x.timestamp, reverse=True)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and metrics."""
        with self._lock:
            stats = self.error_stats.copy()
            
            # Add recent error counts
            recent_errors = self.get_recent_errors(hours=24)
            stats['errors_last_24h'] = len(recent_errors)
            
            # Add top error types
            error_type_counts = {}
            for error in recent_errors:
                error_type = error.error_type.value
                error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            
            stats['top_error_types'] = sorted(
                error_type_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Calculate recovery rate
            recoverable_errors = [e for e in recent_errors if e.recoverable]
            if recoverable_errors:
                successful_recoveries = [e for e in recoverable_errors if e.recovery_successful]
                stats['recovery_success_rate'] = len(successful_recoveries) / len(recoverable_errors)
            
            return stats
    
    def clear_error_history(self, older_than_days: int = 30) -> int:
        """Clear old errors from history."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(days=older_than_days)
            
            initial_count = len(self.error_history)
            self.error_history = [
                error for error in self.error_history
                if error.timestamp >= cutoff_time
            ]
            
            cleared_count = initial_count - len(self.error_history)
            logger.info(f"Cleared {cleared_count} old errors from history")
            
            return cleared_count
    
    def export_errors(
        self,
        file_path: Path,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        format: str = 'json'
    ) -> None:
        """Export errors to file."""
        with self._lock:
            errors_to_export = self.error_history
            
            if start_date:
                errors_to_export = [e for e in errors_to_export if e.timestamp >= start_date]
            
            if end_date:
                errors_to_export = [e for e in errors_to_export if e.timestamp <= end_date]
            
            if format.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(
                        [error.to_dict() for error in errors_to_export],
                        f,
                        indent=2,
                        default=str
                    )
            
            logger.info(f"Exported {len(errors_to_export)} errors to {file_path}")
    
    # Private methods
    
    def _infer_error_type(self, exception: Exception, category: ErrorCategory) -> ErrorType:
        """Infer error type from exception and category."""
        # Check if it's a custom exception with error type
        if hasattr(exception, 'error_type'):
            return exception.error_type
        
        # Infer from exception type and message
        exception_name = type(exception).__name__.lower()
        message = str(exception).lower()
        
        # Trading-related inference
        if category == ErrorCategory.TRADING:
            if 'insufficient funds' in message or 'insufficient balance' in message:
                return ErrorType.INSUFFICIENT_FUNDS
            elif 'invalid symbol' in message or 'symbol not found' in message:
                return ErrorType.INVALID_SYMBOL
            elif 'market closed' in message:
                return ErrorType.MARKET_CLOSED
            else:
                return ErrorType.ORDER_REJECTION
        
        # Network-related inference
        elif category == ErrorCategory.NETWORK:
            if 'timeout' in message or 'timed out' in message:
                return ErrorType.CONNECTION_TIMEOUT
            elif 'rate limit' in message or 'too many requests' in message:
                return ErrorType.API_RATE_LIMIT
            else:
                return ErrorType.NETWORK_UNAVAILABLE
        
        # Data-related inference
        elif category == ErrorCategory.DATA:
            if 'missing' in message or 'not found' in message:
                return ErrorType.MISSING_DATA
            elif 'stale' in message or 'outdated' in message:
                return ErrorType.STALE_DATA
            else:
                return ErrorType.DATA_CORRUPTION
        
        # Default based on exception type
        if 'memory' in exception_name:
            return ErrorType.MEMORY_ERROR
        elif 'file' in exception_name or 'io' in exception_name:
            return ErrorType.FILE_SYSTEM_ERROR
        elif 'connection' in exception_name or 'network' in exception_name:
            return ErrorType.CONNECTION_TIMEOUT
        else:
            return ErrorType.DATABASE_ERROR  # Default fallback
    
    def _is_recoverable(self, exception: Exception, error_type: ErrorType) -> bool:
        """Determine if an error is recoverable."""
        # Check if custom exception specifies recoverability
        if hasattr(exception, 'recoverable'):
            return exception.recoverable
        
        # Determine based on error type
        non_recoverable_types = {
            ErrorType.INVALID_CONFIG,
            ErrorType.MISSING_CONFIG,
            ErrorType.INVALID_SYMBOL,
            ErrorType.DATA_CORRUPTION
        }
        
        return error_type not in non_recoverable_types
    
    def _add_system_context(self, context: ErrorContext) -> None:
        """Add system context information to error."""
        import os
        import psutil
        
        try:
            context.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
            context.process_id = os.getpid()
            context.thread_id = threading.get_ident()
            
            # Get memory usage
            process = psutil.Process()
            context.memory_usage = process.memory_info().rss
            context.cpu_usage = process.cpu_percent()
            
        except Exception:
            # If we can't get system info, continue without it
            pass
    
    def _is_duplicate_error(self, context: ErrorContext) -> bool:
        """Check if this is a duplicate error within the deduplication window."""
        error_key = f"{context.category.value}:{context.error_type.value}:{context.message}"
        
        if error_key in self.recent_errors:
            last_occurrence = self.recent_errors[error_key]
            if datetime.now() - last_occurrence < self.deduplication_window:
                return True
        
        return False
    
    def _update_duplicate_error(self, context: ErrorContext) -> None:
        """Update duplicate error information."""
        error_key = f"{context.category.value}:{context.error_type.value}:{context.message}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.recent_errors[error_key] = context.timestamp
    
    def _store_error(self, context: ErrorContext) -> None:
        """Store error in history."""
        self.error_history.append(context)
        
        # Maintain max history size
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
        
        # Update recent errors tracking
        error_key = f"{context.category.value}:{context.error_type.value}:{context.message}"
        self.recent_errors[error_key] = context.timestamp
    
    def _update_error_statistics(self, context: ErrorContext) -> None:
        """Update error statistics."""
        self.error_stats['total_errors'] += 1
        self.error_stats['errors_by_severity'][context.severity.value] += 1
        self.error_stats['errors_by_category'][context.category.value] += 1
    
    def _log_error(self, context: ErrorContext) -> None:
        """Log error to file."""
        try:
            with open(self.error_log_path, 'a') as f:
                json.dump(context.to_dict(), f, default=str)
                f.write('\n')
        except Exception as e:
            logger.warning(f"Failed to write error to log file: {e}")
    
    def _notify_error_handlers(self, context: ErrorContext) -> None:
        """Notify registered error handlers."""
        handlers = self.error_handlers.get(context.category, [])
        
        for handler in handlers:
            try:
                handler(context)
            except Exception as e:
                logger.error(f"Error handler failed: {e}")
    
    def _attempt_recovery(self, context: ErrorContext) -> None:
        """Attempt to recover from error."""
        if context.error_type in self.recovery_handlers:
            try:
                context.recovery_attempted = True
                recovery_handler = self.recovery_handlers[context.error_type]
                success = recovery_handler(context)
                context.recovery_successful = success
                
                if success:
                    logger.info(f"Successfully recovered from error: {context.error_id}")
                else:
                    logger.warning(f"Recovery failed for error: {context.error_id}")
                    
            except Exception as e:
                context.recovery_successful = False
                logger.error(f"Recovery handler failed for error {context.error_id}: {e}")


# Global error manager instance
_error_manager = None


def get_error_manager() -> ErrorManager:
    """Get global error manager instance."""
    global _error_manager
    if _error_manager is None:
        _error_manager = ErrorManager()
    return _error_manager