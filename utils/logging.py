"""
Enhanced logging utilities for the Algua trading platform.
Provides comprehensive structured logging with performance monitoring,
trading-specific log levels, and advanced formatting capabilities.
"""

import logging
import logging.config
import logging.handlers
import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import functools
import psutil


class LogLevel(Enum):
    """Enhanced log levels for trading applications."""
    TRACE = 5        # Detailed execution traces
    DEBUG = 10       # Debug information
    INFO = 20        # General information
    AUDIT = 25       # Audit trail for compliance
    WARNING = 30     # Warning messages
    ERROR = 40       # Error conditions
    CRITICAL = 50    # Critical failures
    TRADE = 35       # Trading-specific events
    PERFORMANCE = 15 # Performance metrics


class LogCategory(Enum):
    """Log categories for classification."""
    TRADING = "trading"
    RISK = "risk"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    AUDIT = "audit"
    ERROR = "error"
    NETWORK = "network"
    DATA = "data"
    STRATEGY = "strategy"
    PORTFOLIO = "portfolio"


@dataclass
class LogConfig:
    """Configuration for enhanced logging system."""
    level: str = "INFO"
    format_type: str = "json"
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    max_file_size: int = 100_000_000  # 100MB
    backup_count: int = 10
    enable_performance_logging: bool = True
    enable_audit_logging: bool = True
    enable_trading_logging: bool = True
    console_output: bool = True
    structured_metadata: bool = True
    correlation_id_enabled: bool = True
    

@dataclass
class PerformanceMetrics:
    """Performance metrics for logging."""
    start_time: float
    end_time: float
    duration: float
    cpu_usage_start: float
    cpu_usage_end: float
    memory_usage_start: int
    memory_usage_end: int
    function_name: str
    args_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'duration_ms': round(self.duration * 1000, 3),
            'cpu_usage_delta': round(self.cpu_usage_end - self.cpu_usage_start, 2),
            'memory_delta_mb': round((self.memory_usage_end - self.memory_usage_start) / 1024 / 1024, 2),
            'function': self.function_name,
            'args_hash': self.args_hash
        }


def setup_logging(
    config: Optional[LogConfig] = None,
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None
) -> None:
    """
    Setup enhanced logging for the application.
    
    Args:
        config: LogConfig object for advanced configuration
        level: Logging level (for backward compatibility)
        format_type: Format type ('json' or 'text')
        log_file: Optional log file path (for backward compatibility)
    """
    if config is None:
        config = LogConfig(level=level, format_type=format_type)
        if log_file:
            config.log_dir = Path(log_file).parent
    
    # Add custom log levels
    for log_level in LogLevel:
        logging.addLevelName(log_level.value, log_level.name)
    
    # Create log directory
    config.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure formatters
    if config.format_type == "json":
        formatter = EnhancedJsonFormatter(config)
    else:
        formatter = EnhancedTextFormatter()
    
    handlers = []
    
    # Console handler
    if config.console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.level.upper()))
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # Main application log with rotation
    main_log_file = config.log_dir / "algua.log"
    main_handler = logging.handlers.RotatingFileHandler(
        main_log_file,
        maxBytes=config.max_file_size,
        backupCount=config.backup_count
    )
    main_handler.setLevel(getattr(logging, config.level.upper()))
    main_handler.setFormatter(formatter)
    handlers.append(main_handler)
    
    # Trading-specific log
    if config.enable_trading_logging:
        trading_log_file = config.log_dir / "trading.log"
        trading_handler = logging.handlers.RotatingFileHandler(
            trading_log_file,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        trading_handler.setLevel(LogLevel.TRADE.value)
        trading_handler.setFormatter(formatter)
        trading_handler.addFilter(CategoryFilter(LogCategory.TRADING))
        handlers.append(trading_handler)
    
    # Performance log
    if config.enable_performance_logging:
        perf_log_file = config.log_dir / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        perf_handler.setLevel(LogLevel.PERFORMANCE.value)
        perf_handler.setFormatter(formatter)
        perf_handler.addFilter(CategoryFilter(LogCategory.PERFORMANCE))
        handlers.append(perf_handler)
    
    # Audit log
    if config.enable_audit_logging:
        audit_log_file = config.log_dir / "audit.log"
        audit_handler = logging.handlers.RotatingFileHandler(
            audit_log_file,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        audit_handler.setLevel(LogLevel.AUDIT.value)
        audit_handler.setFormatter(formatter)
        audit_handler.addFilter(CategoryFilter(LogCategory.AUDIT))
        handlers.append(audit_handler)
    
    # Error log
    error_log_file = config.log_dir / "errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=config.max_file_size,
        backupCount=config.backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    handlers.append(error_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=LogLevel.TRACE.value,  # Set to lowest level, handlers will filter
        handlers=handlers,
        format="%(message)s" if config.format_type == "json" else None
    )
    
    # Store config globally for other components
    global _log_config
    _log_config = config


# Global config storage
_log_config: Optional[LogConfig] = None
_correlation_context = threading.local()


class CategoryFilter(logging.Filter):
    """Filter logs by category."""
    
    def __init__(self, category: LogCategory):
        super().__init__()
        self.category = category.value
    
    def filter(self, record):
        return getattr(record, 'category', None) == self.category


class EnhancedJsonFormatter(logging.Formatter):
    """Enhanced JSON formatter with structured metadata."""
    
    def __init__(self, config: LogConfig):
        super().__init__()
        self.config = config
    
    def format(self, record):
        # Base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'thread_id': threading.get_ident(),
            'process_id': os.getpid()
        }
        
        # Add correlation ID if enabled
        if self.config.correlation_id_enabled:
            correlation_id = getattr(_correlation_context, 'correlation_id', None)
            if correlation_id:
                log_entry['correlation_id'] = correlation_id
        
        # Add structured metadata
        if self.config.structured_metadata:
            # Add category if present
            if hasattr(record, 'category'):
                log_entry['category'] = record.category
            
            # Add performance metrics if present
            if hasattr(record, 'performance_metrics'):
                log_entry['performance'] = record.performance_metrics
            
            # Add trading context if present
            if hasattr(record, 'trading_context'):
                log_entry['trading'] = record.trading_context
            
            # Add error context if present
            if hasattr(record, 'error_context'):
                log_entry['error'] = record.error_context
            
            # Add custom fields
            if hasattr(record, 'extra_fields'):
                log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add file location info
        if record.pathname:
            log_entry['source'] = {
                'file': record.pathname,
                'line': record.lineno,
                'function': record.funcName
            }
        
        return json.dumps(log_entry, default=str)


class EnhancedTextFormatter(logging.Formatter):
    """Enhanced text formatter with additional context."""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record):
        formatted = super().format(record)
        
        # Add category if present
        if hasattr(record, 'category'):
            formatted = f"[{record.category}] {formatted}"
        
        # Add correlation ID if present
        correlation_id = getattr(_correlation_context, 'correlation_id', None)
        if correlation_id:
            formatted = f"[{correlation_id[:8]}] {formatted}"
        
        return formatted


class JsonFormatter(logging.Formatter):
    """Legacy JSON formatter for backward compatibility."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
        
        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger
    """
    return logging.getLogger(name)


def get_enhanced_logger(name: str, category: Optional[LogCategory] = None) -> 'EnhancedLogger':
    """
    Get an enhanced logger instance with additional functionality.
    
    Args:
        name: Logger name (typically __name__)
        category: Default log category
        
    Returns:
        Enhanced logger instance
    """
    return EnhancedLogger(name, category)


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current thread."""
    _correlation_context.correlation_id = correlation_id


def clear_correlation_id() -> None:
    """Clear correlation ID for current thread."""
    if hasattr(_correlation_context, 'correlation_id'):
        delattr(_correlation_context, 'correlation_id')


def performance_logging(include_args: bool = False):
    """
    Decorator for automatic performance logging.
    
    Args:
        include_args: Whether to include function arguments in logs
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_enhanced_logger(func.__module__, LogCategory.PERFORMANCE)
            
            # Capture start metrics
            start_time = time.time()
            process = psutil.Process()
            cpu_start = process.cpu_percent()
            memory_start = process.memory_info().rss
            
            # Generate args hash if requested
            args_hash = None
            if include_args:
                try:
                    args_hash = str(hash(str(args) + str(kwargs)))[:8]
                except:
                    args_hash = "unhashable"
            
            try:
                result = func(*args, **kwargs)
                
                # Capture end metrics
                end_time = time.time()
                cpu_end = process.cpu_percent()
                memory_end = process.memory_info().rss
                
                # Create performance metrics
                metrics = PerformanceMetrics(
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    cpu_usage_start=cpu_start,
                    cpu_usage_end=cpu_end,
                    memory_usage_start=memory_start,
                    memory_usage_end=memory_end,
                    function_name=func.__name__,
                    args_hash=args_hash
                )
                
                # Log performance
                logger.performance(
                    f"Function {func.__name__} completed",
                    performance_metrics=metrics.to_dict()
                )
                
                return result
                
            except Exception as e:
                # Log performance even on failure
                end_time = time.time()
                cpu_end = process.cpu_percent()
                memory_end = process.memory_info().rss
                
                metrics = PerformanceMetrics(
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    cpu_usage_start=cpu_start,
                    cpu_usage_end=cpu_end,
                    memory_usage_start=memory_start,
                    memory_usage_end=memory_end,
                    function_name=func.__name__,
                    args_hash=args_hash
                )
                
                logger.error(
                    f"Function {func.__name__} failed",
                    performance_metrics=metrics.to_dict(),
                    exception=str(e)
                )
                
                raise
        
        return wrapper
    return decorator


class EnhancedLogger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str, default_category: Optional[LogCategory] = None):
        self.logger = logging.getLogger(name)
        self.default_category = default_category
    
    def _log(
        self,
        level: int,
        message: str,
        category: Optional[LogCategory] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        trading_context: Optional[Dict[str, Any]] = None,
        error_context: Optional[Dict[str, Any]] = None,
        **extra_fields
    ):
        """Internal logging method with structured data."""
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        
        # Add structured data
        if category or self.default_category:
            record.category = (category or self.default_category).value
        
        if performance_metrics:
            record.performance_metrics = performance_metrics
        
        if trading_context:
            record.trading_context = trading_context
        
        if error_context:
            record.error_context = error_context
        
        if extra_fields:
            record.extra_fields = extra_fields
        
        self.logger.handle(record)
    
    def trace(self, message: str, **kwargs):
        """Log trace level message."""
        self._log(LogLevel.TRACE.value, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self._log(LogLevel.DEBUG.value, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info level message."""
        self._log(LogLevel.INFO.value, message, **kwargs)
    
    def audit(self, message: str, **kwargs):
        """Log audit level message."""
        self._log(LogLevel.AUDIT.value, message, category=LogCategory.AUDIT, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self._log(LogLevel.WARNING.value, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error level message."""
        self._log(LogLevel.ERROR.value, message, category=LogCategory.ERROR, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical level message."""
        self._log(LogLevel.CRITICAL.value, message, category=LogCategory.ERROR, **kwargs)
    
    def trade(self, message: str, **kwargs):
        """Log trading event."""
        self._log(LogLevel.TRADE.value, message, category=LogCategory.TRADING, **kwargs)
    
    def performance(self, message: str, **kwargs):
        """Log performance metrics."""
        self._log(LogLevel.PERFORMANCE.value, message, category=LogCategory.PERFORMANCE, **kwargs)
    
    def log_trade_execution(
        self,
        action: str,
        symbol: str,
        quantity: Union[int, float, Decimal],
        price: Union[float, Decimal],
        order_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        **metadata
    ):
        """Log trade execution with structured data."""
        trading_context = {
            'action': action,
            'symbol': symbol,
            'quantity': float(quantity),
            'price': float(price),
            'order_id': order_id,
            'strategy_id': strategy_id,
            'execution_time': datetime.utcnow().isoformat(),
            **metadata
        }
        
        self.trade(
            f"Trade executed: {action} {quantity} {symbol} @ {price}",
            trading_context=trading_context
        )
    
    def log_order_update(
        self,
        order_id: str,
        status: str,
        symbol: Optional[str] = None,
        **metadata
    ):
        """Log order status update."""
        trading_context = {
            'order_id': order_id,
            'status': status,
            'symbol': symbol,
            'update_time': datetime.utcnow().isoformat(),
            **metadata
        }
        
        self.trade(
            f"Order {order_id} status: {status}",
            trading_context=trading_context
        )
    
    def log_portfolio_update(
        self,
        total_value: Union[float, Decimal],
        daily_pnl: Union[float, Decimal],
        positions_count: Optional[int] = None,
        **metadata
    ):
        """Log portfolio value update."""
        trading_context = {
            'total_value': float(total_value),
            'daily_pnl': float(daily_pnl),
            'positions_count': positions_count,
            'update_time': datetime.utcnow().isoformat(),
            **metadata
        }
        
        self.info(
            f"Portfolio update: ${total_value:.2f} (PnL: ${daily_pnl:.2f})",
            category=LogCategory.PORTFOLIO,
            trading_context=trading_context
        )
    
    def log_strategy_signal(
        self,
        strategy_name: str,
        signal: str,
        symbol: str,
        confidence: Union[float, Decimal],
        **metadata
    ):
        """Log strategy signal generation."""
        trading_context = {
            'strategy': strategy_name,
            'signal': signal,
            'symbol': symbol,
            'confidence': float(confidence),
            'signal_time': datetime.utcnow().isoformat(),
            **metadata
        }
        
        self.info(
            f"Strategy {strategy_name} signal: {signal} for {symbol} (confidence: {confidence:.2%})",
            category=LogCategory.STRATEGY,
            trading_context=trading_context
        )
    
    def log_risk_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        affected_positions: Optional[List[str]] = None,
        **metadata
    ):
        """Log risk management events."""
        trading_context = {
            'event_type': event_type,
            'severity': severity,
            'description': description,
            'affected_positions': affected_positions or [],
            'event_time': datetime.utcnow().isoformat(),
            **metadata
        }
        
        # Map severity to log level
        severity_map = {
            'low': LogLevel.INFO.value,
            'medium': LogLevel.WARNING.value,
            'high': LogLevel.ERROR.value,
            'critical': LogLevel.CRITICAL.value
        }
        
        level = severity_map.get(severity.lower(), LogLevel.WARNING.value)
        
        self._log(
            level,
            f"Risk event: {event_type} - {description}",
            category=LogCategory.RISK,
            trading_context=trading_context
        )
    
    def log_error_with_context(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: str = "error"
    ):
        """Log error with additional context."""
        error_context = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        level_map = {
            'warning': LogLevel.WARNING.value,
            'error': LogLevel.ERROR.value,
            'critical': LogLevel.CRITICAL.value
        }
        
        level = level_map.get(severity.lower(), LogLevel.ERROR.value)
        
        self._log(
            level,
            f"Error occurred: {str(error)}",
            category=LogCategory.ERROR,
            error_context=error_context
        )


class TradingLogger:
    """
    Legacy trading logger for backward compatibility.
    Enhanced version available as EnhancedLogger.
    """
    
    def __init__(self, name: str):
        self.logger = get_enhanced_logger(name, LogCategory.TRADING)
    
    def log_trade(self, action: str, symbol: str, quantity: float, 
                  price: float, **kwargs):
        """Log a trading action."""
        self.logger.log_trade_execution(
            action=action,
            symbol=symbol,
            quantity=quantity,
            price=price,
            **kwargs
        )
    
    def log_order(self, order_id: str, status: str, **kwargs):
        """Log order status update."""
        self.logger.log_order_update(
            order_id=order_id,
            status=status,
            **kwargs
        )
    
    def log_portfolio_update(self, total_value: float, daily_pnl: float, **kwargs):
        """Log portfolio value update."""
        self.logger.log_portfolio_update(
            total_value=total_value,
            daily_pnl=daily_pnl,
            **kwargs
        )
    
    def log_strategy_signal(self, strategy: str, signal: str, 
                           symbol: str, confidence: float, **kwargs):
        """Log strategy signal generation."""
        self.logger.log_strategy_signal(
            strategy_name=strategy,
            signal=signal,
            symbol=symbol,
            confidence=confidence,
            **kwargs
        )
    
    def log_risk_event(self, event_type: str, severity: str, 
                       description: str, **kwargs):
        """Log risk management events."""
        self.logger.log_risk_event(
            event_type=event_type,
            severity=severity,
            description=description,
            **kwargs
        )


class LogManager:
    """
    Central log management for monitoring and analysis.
    """
    
    def __init__(self):
        self.config = _log_config or LogConfig()
        self._stats_lock = threading.Lock()
        self._log_stats = {
            'total_logs': 0,
            'logs_by_level': {},
            'logs_by_category': {},
            'errors_count': 0,
            'warnings_count': 0,
            'performance_logs': 0,
            'trading_logs': 0
        }
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self._stats_lock:
            return self._log_stats.copy()
    
    def rotate_logs(self) -> None:
        """Manually trigger log rotation."""
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.doRollover()
    
    def set_log_level(self, level: str) -> None:
        """Change log level dynamically."""
        log_level = getattr(logging, level.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)
        
        for handler in logging.getLogger().handlers:
            handler.setLevel(log_level)
    
    def analyze_log_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze log patterns for the specified time period."""
        # This would analyze log files for patterns
        # Implementation would read log files and extract metrics
        return {
            'analysis_period_hours': hours,
            'total_logs_analyzed': 0,
            'error_patterns': [],
            'performance_trends': {},
            'warning_frequency': 0
        }
    
    def export_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        categories: Optional[List[LogCategory]] = None,
        format: str = 'json'
    ) -> str:
        """Export logs for the specified criteria."""
        # Implementation would read and filter log files
        return f"Logs exported for period {start_time} to {end_time}"
    
    def cleanup_old_logs(self, days: int = 30) -> int:
        """Clean up log files older than specified days."""
        cleaned_count = 0
        log_dir = self.config.log_dir
        
        if log_dir.exists():
            cutoff_time = datetime.now() - timedelta(days=days)
            
            for log_file in log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_time.timestamp():
                    try:
                        log_file.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger = get_logger(__name__)
                        logger.warning(f"Failed to delete old log file {log_file}: {e}")
        
        return cleaned_count


# Global log manager instance
_log_manager = None


def get_log_manager() -> LogManager:
    """Get global log manager instance."""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager


# Initialize enhanced logging on module import
try:
    config = LogConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format_type=os.getenv("LOG_FORMAT", "json"),
        log_dir=Path(os.getenv("LOG_DIR", "logs")),
        enable_performance_logging=os.getenv("ENABLE_PERF_LOGGING", "true").lower() == "true",
        enable_audit_logging=os.getenv("ENABLE_AUDIT_LOGGING", "true").lower() == "true",
        enable_trading_logging=os.getenv("ENABLE_TRADING_LOGGING", "true").lower() == "true"
    )
    setup_logging(config)
except Exception as e:
    # Fallback to basic logging if enhanced setup fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger(__name__).warning(f"Enhanced logging setup failed, using basic logging: {e}") 