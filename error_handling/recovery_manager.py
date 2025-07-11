"""
Automatic error recovery and resolution system.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import asyncio
from abc import ABC, abstractmethod

from .error_manager import ErrorContext, ErrorSeverity, ErrorCategory, ErrorType
from utils.logging import get_logger

logger = get_logger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    RESTART_COMPONENT = "restart_component"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    MANUAL_INTERVENTION = "manual_intervention"
    NO_RECOVERY = "no_recovery"


class RecoveryStatus(Enum):
    """Recovery attempt status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ABORTED = "aborted"


@dataclass
class RecoveryAction:
    """Definition of a recovery action."""
    action_id: str
    name: str
    strategy: RecoveryStrategy
    
    # Execution parameters
    max_attempts: int = 3
    initial_delay: timedelta = field(default_factory=lambda: timedelta(seconds=1))
    max_delay: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    backoff_multiplier: Decimal = Decimal('2.0')
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=10))
    
    # Conditions
    applicable_error_types: List[ErrorType] = field(default_factory=list)
    applicable_categories: List[ErrorCategory] = field(default_factory=list)
    min_severity: ErrorSeverity = ErrorSeverity.LOW
    max_severity: ErrorSeverity = ErrorSeverity.CRITICAL
    
    # Recovery function
    recovery_function: Optional[Callable] = None
    fallback_function: Optional[Callable] = None
    
    # Metadata
    description: Optional[str] = None
    prerequisites: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    
    def is_applicable(self, error_context: ErrorContext) -> bool:
        """Check if this recovery action is applicable to the error."""
        # Check severity range
        severity_values = {
            ErrorSeverity.INFO: 0,
            ErrorSeverity.LOW: 1,
            ErrorSeverity.MEDIUM: 2,
            ErrorSeverity.HIGH: 3,
            ErrorSeverity.CRITICAL: 4
        }
        
        min_val = severity_values[self.min_severity]
        max_val = severity_values[self.max_severity]
        error_val = severity_values[error_context.severity]
        
        if not (min_val <= error_val <= max_val):
            return False
        
        # Check error type
        if self.applicable_error_types and error_context.error_type not in self.applicable_error_types:
            return False
        
        # Check category
        if self.applicable_categories and error_context.category not in self.applicable_categories:
            return False
        
        return True


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    recovery_id: str
    error_id: str
    action: RecoveryAction
    
    # Execution details
    status: RecoveryStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    attempts_made: int = 0
    
    # Results
    success: bool = False
    error_resolved: bool = False
    partial_recovery: bool = False
    
    # Details
    recovery_message: Optional[str] = None
    recovery_data: Dict[str, Any] = field(default_factory=dict)
    side_effects: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_duration: Optional[timedelta] = None
    recovery_cost: Decimal = Decimal('0.0')  # Time, resources, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'recovery_id': self.recovery_id,
            'error_id': self.error_id,
            'action_id': self.action.action_id,
            'action_name': self.action.name,
            'strategy': self.action.strategy.value,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'attempts_made': self.attempts_made,
            'success': self.success,
            'error_resolved': self.error_resolved,
            'partial_recovery': self.partial_recovery,
            'recovery_message': self.recovery_message,
            'recovery_data': self.recovery_data,
            'side_effects': self.side_effects,
            'total_duration': str(self.total_duration) if self.total_duration else None,
            'recovery_cost': float(self.recovery_cost)
        }


class BaseRecoveryHandler(ABC):
    """Base class for recovery handlers."""
    
    @abstractmethod
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this handler can handle the given error."""
        pass
    
    @abstractmethod
    def recover(self, error_context: ErrorContext) -> RecoveryResult:
        """Attempt to recover from the error."""
        pass


class RetryHandler(BaseRecoveryHandler):
    """Simple retry recovery handler."""
    
    def __init__(self, max_attempts: int = 3, delay: timedelta = timedelta(seconds=1)):
        self.max_attempts = max_attempts
        self.delay = delay
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if error is retryable."""
        retryable_types = {
            ErrorType.CONNECTION_TIMEOUT,
            ErrorType.NETWORK_UNAVAILABLE,
            ErrorType.API_RATE_LIMIT,
            ErrorType.DATABASE_ERROR
        }
        return error_context.error_type in retryable_types
    
    def recover(self, error_context: ErrorContext) -> RecoveryResult:
        """Attempt recovery with simple retry."""
        import uuid
        
        result = RecoveryResult(
            recovery_id=str(uuid.uuid4()),
            error_id=error_context.error_id,
            action=RecoveryAction(
                action_id="simple_retry",
                name="Simple Retry",
                strategy=RecoveryStrategy.RETRY,
                max_attempts=self.max_attempts
            ),
            status=RecoveryStatus.IN_PROGRESS,
            start_time=datetime.now()
        )
        
        for attempt in range(1, self.max_attempts + 1):
            result.attempts_made = attempt
            
            try:
                # Simulate retry logic - in real implementation, this would
                # call the original function that failed
                time.sleep(self.delay.total_seconds())
                
                # For demo purposes, assume success after 2 attempts
                if attempt >= 2:
                    result.success = True
                    result.error_resolved = True
                    result.status = RecoveryStatus.SUCCESS
                    result.recovery_message = f"Successfully recovered after {attempt} attempts"
                    break
                    
            except Exception as e:
                result.recovery_message = f"Retry attempt {attempt} failed: {str(e)}"
                logger.warning(f"Recovery attempt {attempt} failed: {e}")
        
        result.end_time = datetime.now()
        result.total_duration = result.end_time - result.start_time
        
        if not result.success:
            result.status = RecoveryStatus.FAILED
            result.recovery_message = f"Recovery failed after {self.max_attempts} attempts"
        
        return result


class ExponentialBackoffHandler(BaseRecoveryHandler):
    """Exponential backoff recovery handler."""
    
    def __init__(
        self,
        max_attempts: int = 5,
        initial_delay: timedelta = timedelta(seconds=1),
        max_delay: timedelta = timedelta(minutes=5),
        backoff_multiplier: Decimal = Decimal('2.0')
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if error requires exponential backoff."""
        backoff_types = {
            ErrorType.API_RATE_LIMIT,
            ErrorType.CONNECTION_TIMEOUT,
            ErrorType.CPU_OVERLOAD,
            ErrorType.MEMORY_ERROR
        }
        return error_context.error_type in backoff_types
    
    def recover(self, error_context: ErrorContext) -> RecoveryResult:
        """Attempt recovery with exponential backoff."""
        import uuid
        
        result = RecoveryResult(
            recovery_id=str(uuid.uuid4()),
            error_id=error_context.error_id,
            action=RecoveryAction(
                action_id="exponential_backoff",
                name="Exponential Backoff Retry",
                strategy=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=self.max_attempts
            ),
            status=RecoveryStatus.IN_PROGRESS,
            start_time=datetime.now()
        )
        
        current_delay = self.initial_delay
        
        for attempt in range(1, self.max_attempts + 1):
            result.attempts_made = attempt
            
            try:
                # Wait with current delay
                time.sleep(current_delay.total_seconds())
                
                # Simulate recovery attempt
                # For demo, assume success after 3 attempts for rate limit errors
                if (error_context.error_type == ErrorType.API_RATE_LIMIT and attempt >= 3):
                    result.success = True
                    result.error_resolved = True
                    result.status = RecoveryStatus.SUCCESS
                    result.recovery_message = f"Rate limit resolved after {attempt} attempts with backoff"
                    break
                
                # Calculate next delay
                next_delay_seconds = current_delay.total_seconds() * float(self.backoff_multiplier)
                current_delay = timedelta(seconds=min(next_delay_seconds, self.max_delay.total_seconds()))
                
            except Exception as e:
                result.recovery_message = f"Backoff attempt {attempt} failed: {str(e)}"
                logger.warning(f"Recovery attempt {attempt} failed: {e}")
        
        result.end_time = datetime.now()
        result.total_duration = result.end_time - result.start_time
        
        if not result.success:
            result.status = RecoveryStatus.FAILED
            result.recovery_message = f"Recovery failed after {self.max_attempts} attempts with exponential backoff"
        
        return result


class CircuitBreakerHandler(BaseRecoveryHandler):
    """Circuit breaker pattern recovery handler."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: timedelta = timedelta(minutes=1),
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        # Circuit breaker state
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if circuit breaker should handle this error."""
        circuit_breaker_types = {
            ErrorType.CONNECTION_TIMEOUT,
            ErrorType.DATABASE_ERROR,
            ErrorType.NETWORK_UNAVAILABLE,
            ErrorType.API_RATE_LIMIT
        }
        return error_context.error_type in circuit_breaker_types
    
    def recover(self, error_context: ErrorContext) -> RecoveryResult:
        """Implement circuit breaker pattern."""
        import uuid
        
        result = RecoveryResult(
            recovery_id=str(uuid.uuid4()),
            error_id=error_context.error_id,
            action=RecoveryAction(
                action_id="circuit_breaker",
                name="Circuit Breaker",
                strategy=RecoveryStrategy.CIRCUIT_BREAKER
            ),
            status=RecoveryStatus.IN_PROGRESS,
            start_time=datetime.now()
        )
        
        current_time = datetime.now()
        
        # Check circuit breaker state
        if self.state == "open":
            # Check if we should transition to half-open
            if (self.last_failure_time and 
                current_time - self.last_failure_time >= self.recovery_timeout):
                self.state = "half-open"
                self.failure_count = 0
                result.recovery_message = "Circuit breaker transitioning to half-open state"
            else:
                # Circuit is open, fail fast
                result.status = RecoveryStatus.FAILED
                result.recovery_message = "Circuit breaker is open, failing fast"
                result.end_time = current_time
                result.total_duration = result.end_time - result.start_time
                return result
        
        # Attempt recovery
        try:
            # Simulate recovery attempt
            result.attempts_made = 1
            
            # For demo, assume recovery succeeds in half-open state
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                result.success = True
                result.error_resolved = True
                result.status = RecoveryStatus.SUCCESS
                result.recovery_message = "Circuit breaker closed, service recovered"
            else:
                # Simulate failure to test circuit breaker logic
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    self.last_failure_time = current_time
                    result.recovery_message = f"Circuit breaker opened after {self.failure_count} failures"
                else:
                    result.recovery_message = f"Service still failing, failure count: {self.failure_count}"
                
                result.status = RecoveryStatus.FAILED
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            result.status = RecoveryStatus.FAILED
            result.recovery_message = f"Circuit breaker recovery failed: {str(e)}"
        
        result.end_time = datetime.now()
        result.total_duration = result.end_time - result.start_time
        
        return result


class RecoveryManager:
    """Central recovery management system."""
    
    def __init__(self):
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.recovery_handlers: Dict[str, BaseRecoveryHandler] = {}
        self.recovery_history: List[RecoveryResult] = []
        
        # Configuration
        self.max_concurrent_recoveries = 10
        self.recovery_timeout = timedelta(minutes=15)
        self.max_recovery_history = 1000
        
        # State tracking
        self.active_recoveries: Dict[str, RecoveryResult] = {}
        self.recovery_queue = []
        
        # Statistics
        self.recovery_stats = {
            'total_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'timeouts': 0,
            'avg_recovery_time': 0.0,
            'recovery_rate': 0.0
        }
        
        self._lock = threading.Lock()
        
        # Initialize default handlers
        self._initialize_default_handlers()
        
        logger.info("Recovery manager initialized")
    
    def attempt_recovery(self, error_context: ErrorContext) -> Optional[RecoveryResult]:
        """
        Attempt to recover from an error.
        
        Args:
            error_context: The error to recover from
            
        Returns:
            RecoveryResult if recovery was attempted, None if no recovery available
        """
        with self._lock:
            # Check if recovery is already in progress for this error
            if error_context.error_id in self.active_recoveries:
                logger.info(f"Recovery already in progress for error: {error_context.error_id}")
                return self.active_recoveries[error_context.error_id]
            
            # Find applicable recovery handler
            handler = self._find_recovery_handler(error_context)
            if not handler:
                logger.info(f"No recovery handler available for error: {error_context.error_id}")
                return None
            
            # Check recovery limits
            if len(self.active_recoveries) >= self.max_concurrent_recoveries:
                logger.warning("Maximum concurrent recoveries reached, queuing recovery")
                self.recovery_queue.append(error_context)
                return None
            
            # Attempt recovery
            logger.info(f"Attempting recovery for error: {error_context.error_id}")
            
            try:
                result = handler.recover(error_context)
                
                # Track active recovery
                self.active_recoveries[error_context.error_id] = result
                
                # Update statistics
                self.recovery_stats['total_attempts'] += 1
                
                if result.success:
                    self.recovery_stats['successful_recoveries'] += 1
                    logger.info(f"Recovery successful for error: {error_context.error_id}")
                else:
                    self.recovery_stats['failed_recoveries'] += 1
                    logger.warning(f"Recovery failed for error: {error_context.error_id}")
                
                # Store in history
                self.recovery_history.append(result)
                self._cleanup_history()
                
                # Remove from active recoveries
                if error_context.error_id in self.active_recoveries:
                    del self.active_recoveries[error_context.error_id]
                
                # Process queued recoveries
                self._process_recovery_queue()
                
                return result
                
            except Exception as e:
                logger.error(f"Recovery handler failed for error {error_context.error_id}: {e}")
                self.recovery_stats['failed_recoveries'] += 1
                return None
    
    def register_recovery_handler(self, name: str, handler: BaseRecoveryHandler) -> None:
        """Register a recovery handler."""
        with self._lock:
            self.recovery_handlers[name] = handler
            logger.info(f"Registered recovery handler: {name}")
    
    def register_recovery_action(self, action: RecoveryAction) -> None:
        """Register a recovery action."""
        with self._lock:
            self.recovery_actions[action.action_id] = action
            logger.info(f"Registered recovery action: {action.action_id}")
    
    def get_recovery_status(self, recovery_id: str) -> Optional[RecoveryResult]:
        """Get status of a recovery attempt."""
        with self._lock:
            # Check active recoveries
            for result in self.active_recoveries.values():
                if result.recovery_id == recovery_id:
                    return result
            
            # Check history
            for result in self.recovery_history:
                if result.recovery_id == recovery_id:
                    return result
            
            return None
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        with self._lock:
            stats = self.recovery_stats.copy()
            
            # Calculate derived metrics
            total = stats['total_attempts']
            if total > 0:
                stats['recovery_rate'] = stats['successful_recoveries'] / total
                
                # Calculate average recovery time
                completed_recoveries = [
                    r for r in self.recovery_history
                    if r.total_duration is not None
                ]
                
                if completed_recoveries:
                    avg_duration = sum(
                        r.total_duration.total_seconds() for r in completed_recoveries
                    ) / len(completed_recoveries)
                    stats['avg_recovery_time'] = avg_duration
            
            # Add current state
            stats['active_recoveries'] = len(self.active_recoveries)
            stats['queued_recoveries'] = len(self.recovery_queue)
            stats['registered_handlers'] = len(self.recovery_handlers)
            stats['registered_actions'] = len(self.recovery_actions)
            
            return stats
    
    def get_recovery_history(
        self,
        hours: int = 24,
        status: Optional[RecoveryStatus] = None
    ) -> List[RecoveryResult]:
        """Get recovery history."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            history = [
                result for result in self.recovery_history
                if result.start_time >= cutoff_time
            ]
            
            if status:
                history = [result for result in history if result.status == status]
            
            return sorted(history, key=lambda x: x.start_time, reverse=True)
    
    def cancel_recovery(self, recovery_id: str) -> bool:
        """Cancel an active recovery."""
        with self._lock:
            for error_id, result in self.active_recoveries.items():
                if result.recovery_id == recovery_id:
                    result.status = RecoveryStatus.ABORTED
                    result.end_time = datetime.now()
                    result.total_duration = result.end_time - result.start_time
                    
                    # Move to history
                    self.recovery_history.append(result)
                    del self.active_recoveries[error_id]
                    
                    logger.info(f"Cancelled recovery: {recovery_id}")
                    return True
            
            return False
    
    # Private methods
    
    def _initialize_default_handlers(self) -> None:
        """Initialize default recovery handlers."""
        handlers = {
            'retry': RetryHandler(),
            'exponential_backoff': ExponentialBackoffHandler(),
            'circuit_breaker': CircuitBreakerHandler()
        }
        
        for name, handler in handlers.items():
            self.recovery_handlers[name] = handler
        
        logger.info(f"Initialized {len(handlers)} default recovery handlers")
    
    def _find_recovery_handler(self, error_context: ErrorContext) -> Optional[BaseRecoveryHandler]:
        """Find the best recovery handler for an error."""
        # Priority order for handlers
        handler_priority = ['circuit_breaker', 'exponential_backoff', 'retry']
        
        for handler_name in handler_priority:
            if handler_name in self.recovery_handlers:
                handler = self.recovery_handlers[handler_name]
                if handler.can_handle(error_context):
                    return handler
        
        return None
    
    def _process_recovery_queue(self) -> None:
        """Process queued recovery attempts."""
        while (self.recovery_queue and 
               len(self.active_recoveries) < self.max_concurrent_recoveries):
            
            error_context = self.recovery_queue.pop(0)
            
            # Attempt recovery for queued error
            result = self.attempt_recovery(error_context)
            if result:
                logger.info(f"Processed queued recovery for error: {error_context.error_id}")
    
    def _cleanup_history(self) -> None:
        """Clean up old recovery history."""
        if len(self.recovery_history) > self.max_recovery_history:
            # Keep only the most recent entries
            self.recovery_history = self.recovery_history[-self.max_recovery_history:]


# Global recovery manager instance
_recovery_manager = None


def get_recovery_manager() -> RecoveryManager:
    """Get global recovery manager instance."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = RecoveryManager()
    return _recovery_manager