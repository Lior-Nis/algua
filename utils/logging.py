"""
Logging utilities for the Algua trading platform.
"""

import logging
import logging.config
import os
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None
) -> None:
    """
    Setup logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('json' or 'text')
        log_file: Optional log file path
    """
    
    # Configure standard logging
    handlers = []
    
    if format_type == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers,
        format="%(message)s" if format_type == "json" else None
    )


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
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


class TradingLogger:
    """
    Specialized logger for trading operations.
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def log_trade(self, action: str, symbol: str, quantity: float, 
                  price: float, **kwargs):
        """Log a trading action."""
        self.logger.info(
            "trade_executed",
            action=action,
            symbol=symbol,
            quantity=quantity,
            price=price,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
    
    def log_order(self, order_id: str, status: str, **kwargs):
        """Log order status update."""
        self.logger.info(
            "order_update",
            order_id=order_id,
            status=status,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
    
    def log_portfolio_update(self, total_value: float, daily_pnl: float, **kwargs):
        """Log portfolio value update."""
        self.logger.info(
            "portfolio_update",
            total_value=total_value,
            daily_pnl=daily_pnl,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
    
    def log_strategy_signal(self, strategy: str, signal: str, 
                           symbol: str, confidence: float, **kwargs):
        """Log strategy signal generation."""
        self.logger.info(
            "strategy_signal",
            strategy=strategy,
            signal=signal,
            symbol=symbol,
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )
    
    def log_risk_event(self, event_type: str, severity: str, 
                       description: str, **kwargs):
        """Log risk management events."""
        level_map = {
            "low": "info",
            "medium": "warning", 
            "high": "error",
            "critical": "critical"
        }
        
        log_method = getattr(self.logger, level_map.get(severity, "info"))
        log_method(
            "risk_event",
            event_type=event_type,
            severity=severity,
            description=description,
            timestamp=datetime.utcnow().isoformat(),
            **kwargs
        )


# Initialize logging on module import
setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format_type=os.getenv("LOG_FORMAT", "json"),
    log_file=os.getenv("LOG_FILE")
) 