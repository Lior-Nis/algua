"""
Risk event system for monitoring and handling risk events.
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import threading
from queue import Queue, Empty

from .interfaces import RiskEvent, RiskEventType, RiskLevel
from domain.value_objects import Symbol
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RiskEventStats:
    """Statistics for risk events."""
    total_events: int = 0
    events_by_type: Dict[RiskEventType, int] = field(default_factory=lambda: defaultdict(int))
    events_by_level: Dict[RiskLevel, int] = field(default_factory=lambda: defaultdict(int))
    events_by_symbol: Dict[Symbol, int] = field(default_factory=lambda: defaultdict(int))
    last_event_time: Optional[datetime] = None


class RiskEventHandler:
    """Base class for risk event handlers."""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.events_handled = 0
    
    def handle_event(self, event: RiskEvent) -> None:
        """Handle a risk event."""
        if not self.enabled:
            return
        
        try:
            self._handle_event_impl(event)
            self.events_handled += 1
            logger.info(f"Risk event handled by {self.name}: {event.event_type.value}")
        except Exception as e:
            logger.error(f"Error handling risk event in {self.name}: {e}")
    
    def _handle_event_impl(self, event: RiskEvent) -> None:
        """Implementation of event handling - override in subclasses."""
        pass


class LoggingRiskEventHandler(RiskEventHandler):
    """Handler that logs risk events."""
    
    def __init__(self):
        super().__init__("LoggingHandler")
    
    def _handle_event_impl(self, event: RiskEvent) -> None:
        """Log the risk event."""
        log_level = {
            RiskLevel.LOW: logger.info,
            RiskLevel.MEDIUM: logger.warning,
            RiskLevel.HIGH: logger.error,
            RiskLevel.EXTREME: logger.critical
        }.get(event.risk_level, logger.info)
        
        log_level(
            f"Risk Event: {event.event_type.value} | "
            f"Symbol: {event.symbol} | "
            f"Level: {event.risk_level.value} | "
            f"Message: {event.message}"
        )


class EmailRiskEventHandler(RiskEventHandler):
    """Handler that sends email alerts for risk events."""
    
    def __init__(self, email_config: Optional[Dict[str, Any]] = None):
        super().__init__("EmailHandler")
        self.email_config = email_config or {}
        self.min_level = RiskLevel.MEDIUM  # Only send emails for medium+ risk
    
    def _handle_event_impl(self, event: RiskEvent) -> None:
        """Send email alert for risk event."""
        if event.risk_level.value in ['low']:
            return  # Skip low-risk events
        
        # In a real implementation, you would integrate with an email service
        # For now, just log that an email would be sent
        logger.info(f"EMAIL ALERT: {event.event_type.value} - {event.message}")


class PositionCloseRiskEventHandler(RiskEventHandler):
    """Handler that closes positions in response to risk events."""
    
    def __init__(self, position_manager=None):
        super().__init__("PositionCloseHandler")
        self.position_manager = position_manager
        self.auto_close_events = {
            RiskEventType.STOP_LOSS_TRIGGERED,
            RiskEventType.DAILY_LOSS_EXCEEDED,
            RiskEventType.DRAWDOWN_LIMIT_REACHED
        }
    
    def _handle_event_impl(self, event: RiskEvent) -> None:
        """Close position if event requires it."""
        if event.event_type not in self.auto_close_events:
            return
        
        if self.position_manager is None:
            logger.warning(f"No position manager available to close position for {event.symbol}")
            return
        
        try:
            # In a real implementation, you would close the position
            logger.info(f"AUTO-CLOSING position for {event.symbol} due to {event.event_type.value}")
            # self.position_manager.close_position(event.symbol)
        except Exception as e:
            logger.error(f"Failed to auto-close position for {event.symbol}: {e}")


class RiskEventBus:
    """Event bus for risk events."""
    
    def __init__(self):
        self.handlers: List[RiskEventHandler] = []
        self.event_queue = Queue()
        self.stats = RiskEventStats()
        self.running = False
        self.worker_thread = None
        self._lock = threading.Lock()
    
    def add_handler(self, handler: RiskEventHandler) -> None:
        """Add a risk event handler."""
        with self._lock:
            self.handlers.append(handler)
            logger.info(f"Added risk event handler: {handler.name}")
    
    def remove_handler(self, handler: RiskEventHandler) -> None:
        """Remove a risk event handler."""
        with self._lock:
            if handler in self.handlers:
                self.handlers.remove(handler)
                logger.info(f"Removed risk event handler: {handler.name}")
    
    def publish_event(self, event: RiskEvent) -> None:
        """Publish a risk event."""
        self.event_queue.put(event)
        self._update_stats(event)
        logger.debug(f"Published risk event: {event.event_type.value}")
    
    def _update_stats(self, event: RiskEvent) -> None:
        """Update event statistics."""
        with self._lock:
            self.stats.total_events += 1
            self.stats.events_by_type[event.event_type] += 1
            self.stats.events_by_level[event.risk_level] += 1
            self.stats.events_by_symbol[event.symbol] += 1
            self.stats.last_event_time = event.timestamp
    
    def start(self) -> None:
        """Start the event processing thread."""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_events, daemon=True)
        self.worker_thread.start()
        logger.info("Risk event bus started")
    
    def stop(self) -> None:
        """Stop the event processing thread."""
        if not self.running:
            return
        
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("Risk event bus stopped")
    
    def _process_events(self) -> None:
        """Process events from the queue."""
        while self.running:
            try:
                event = self.event_queue.get(timeout=1.0)
                self._handle_event(event)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing risk event: {e}")
    
    def _handle_event(self, event: RiskEvent) -> None:
        """Handle an event with all registered handlers."""
        with self._lock:
            handlers = self.handlers.copy()
        
        for handler in handlers:
            try:
                handler.handle_event(event)
            except Exception as e:
                logger.error(f"Error in handler {handler.name}: {e}")
    
    def get_stats(self) -> RiskEventStats:
        """Get event statistics."""
        with self._lock:
            return self.stats
    
    def reset_stats(self) -> None:
        """Reset event statistics."""
        with self._lock:
            self.stats = RiskEventStats()


class RiskEventFactory:
    """Factory for creating risk events."""
    
    @staticmethod
    def create_position_size_exceeded_event(
        symbol: Symbol,
        position_size_pct: float,
        max_position_size_pct: float,
        action_taken: Optional[str] = None
    ) -> RiskEvent:
        """Create position size exceeded event."""
        return RiskEvent(
            event_type=RiskEventType.POSITION_SIZE_EXCEEDED,
            timestamp=datetime.now(),
            symbol=symbol,
            risk_level=RiskLevel.HIGH,
            message=f"Position size {position_size_pct:.2%} exceeds limit {max_position_size_pct:.2%}",
            data={
                'position_size_pct': position_size_pct,
                'max_position_size_pct': max_position_size_pct
            },
            action_taken=action_taken
        )
    
    @staticmethod
    def create_daily_loss_exceeded_event(
        symbol: Symbol,
        daily_loss_pct: float,
        max_daily_loss_pct: float,
        action_taken: Optional[str] = None
    ) -> RiskEvent:
        """Create daily loss exceeded event."""
        return RiskEvent(
            event_type=RiskEventType.DAILY_LOSS_EXCEEDED,
            timestamp=datetime.now(),
            symbol=symbol,
            risk_level=RiskLevel.EXTREME,
            message=f"Daily loss {daily_loss_pct:.2%} exceeds limit {max_daily_loss_pct:.2%}",
            data={
                'daily_loss_pct': daily_loss_pct,
                'max_daily_loss_pct': max_daily_loss_pct
            },
            action_taken=action_taken
        )
    
    @staticmethod
    def create_drawdown_limit_reached_event(
        symbol: Symbol,
        drawdown_pct: float,
        max_drawdown_pct: float,
        action_taken: Optional[str] = None
    ) -> RiskEvent:
        """Create drawdown limit reached event."""
        return RiskEvent(
            event_type=RiskEventType.DRAWDOWN_LIMIT_REACHED,
            timestamp=datetime.now(),
            symbol=symbol,
            risk_level=RiskLevel.EXTREME,
            message=f"Drawdown {drawdown_pct:.2%} reached limit {max_drawdown_pct:.2%}",
            data={
                'drawdown_pct': drawdown_pct,
                'max_drawdown_pct': max_drawdown_pct
            },
            action_taken=action_taken
        )
    
    @staticmethod
    def create_stop_loss_triggered_event(
        symbol: Symbol,
        current_price: float,
        stop_loss_price: float,
        action_taken: Optional[str] = None
    ) -> RiskEvent:
        """Create stop loss triggered event."""
        return RiskEvent(
            event_type=RiskEventType.STOP_LOSS_TRIGGERED,
            timestamp=datetime.now(),
            symbol=symbol,
            risk_level=RiskLevel.HIGH,
            message=f"Stop loss triggered at {current_price:.2f} (stop: {stop_loss_price:.2f})",
            data={
                'current_price': current_price,
                'stop_loss_price': stop_loss_price
            },
            action_taken=action_taken
        )


# Global event bus instance
_event_bus = None


def get_risk_event_bus() -> RiskEventBus:
    """Get global risk event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = RiskEventBus()
        
        # Add default handlers
        _event_bus.add_handler(LoggingRiskEventHandler())
        _event_bus.add_handler(EmailRiskEventHandler())
        
        # Start the event bus
        _event_bus.start()
    
    return _event_bus


def publish_risk_event(event: RiskEvent) -> None:
    """Publish a risk event to the global event bus."""
    event_bus = get_risk_event_bus()
    event_bus.publish_event(event)