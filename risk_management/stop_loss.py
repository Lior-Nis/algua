"""
Stop loss mechanisms for risk management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from decimal import Decimal
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

from domain.value_objects import Price, Quantity, Symbol
from .interfaces import StopLossProtocol
from .configuration import get_risk_config
from .event_system import publish_risk_event, RiskEventFactory
from utils.logging import get_logger

logger = get_logger(__name__)


class StopLossType(Enum):
    """Types of stop losses."""
    FIXED_PERCENTAGE = "fixed_percentage"
    TRAILING_PERCENTAGE = "trailing_percentage"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    TIME_BASED = "time_based"
    SUPPORT_RESISTANCE = "support_resistance"


@dataclass
class StopLossOrder:
    """Stop loss order information."""
    symbol: Symbol
    position_type: str  # 'long' or 'short'
    stop_price: Price
    stop_type: StopLossType
    created_at: datetime
    triggered: bool = False
    triggered_at: Optional[datetime] = None
    triggered_price: Optional[Price] = None
    metadata: Dict[str, Any] = None


class BaseStopLoss(ABC):
    """Base class for stop loss mechanisms."""
    
    @abstractmethod
    def calculate_stop_loss(
        self,
        symbol: Symbol,
        entry_price: Price,
        position_size: Quantity,
        position_type: str,
        market_data: Dict[str, Any]
    ) -> Price:
        """Calculate stop loss price."""
        pass
    
    @abstractmethod
    def should_trigger_stop(
        self,
        current_price: Price,
        stop_loss_price: Price,
        position_type: str
    ) -> bool:
        """Check if stop loss should trigger."""
        pass
    
    @abstractmethod
    def update_stop_loss(
        self,
        symbol: Symbol,
        current_price: Price,
        current_stop: Price,
        market_data: Dict[str, Any]
    ) -> Optional[Price]:
        """Update stop loss price (for trailing stops)."""
        pass


class FixedPercentageStopLoss(BaseStopLoss):
    """Fixed percentage stop loss."""
    
    def __init__(self, stop_percentage: Decimal = None, config=None):
        self.config = config or get_risk_config()
        self.stop_percentage = stop_percentage or self.config.default_stop_loss_pct
    
    def calculate_stop_loss(
        self,
        symbol: Symbol,
        entry_price: Price,
        position_size: Quantity,
        position_type: str,
        market_data: Dict[str, Any]
    ) -> Price:
        """Calculate fixed percentage stop loss."""
        if position_type.lower() == 'long':
            stop_price = entry_price.value * (1 - self.stop_percentage)
        else:  # short position
            stop_price = entry_price.value * (1 + self.stop_percentage)
        
        logger.info(
            f"Fixed stop loss for {symbol} ({position_type}): "
            f"entry={entry_price.value:.2f}, stop={stop_price:.2f} ({self.stop_percentage:.2%})"
        )
        
        return Price(stop_price)
    
    def should_trigger_stop(
        self,
        current_price: Price,
        stop_loss_price: Price,
        position_type: str
    ) -> bool:
        """Check if fixed stop loss should trigger."""
        if position_type.lower() == 'long':
            return current_price.value <= stop_loss_price.value
        else:  # short position
            return current_price.value >= stop_loss_price.value
    
    def update_stop_loss(
        self,
        symbol: Symbol,
        current_price: Price,
        current_stop: Price,
        market_data: Dict[str, Any]
    ) -> Optional[Price]:
        """Fixed stops don't update."""
        return None


class TrailingPercentageStopLoss(BaseStopLoss):
    """Trailing percentage stop loss."""
    
    def __init__(self, trail_percentage: Decimal = None, config=None):
        self.config = config or get_risk_config()
        self.trail_percentage = trail_percentage or self.config.trailing_stop_pct
        self.highest_price = {}  # Track highest prices for trailing
        self.lowest_price = {}   # Track lowest prices for short positions
    
    def calculate_stop_loss(
        self,
        symbol: Symbol,
        entry_price: Price,
        position_size: Quantity,
        position_type: str,
        market_data: Dict[str, Any]
    ) -> Price:
        """Calculate initial trailing stop loss."""
        if position_type.lower() == 'long':
            stop_price = entry_price.value * (1 - self.trail_percentage)
            self.highest_price[symbol] = entry_price.value
        else:  # short position
            stop_price = entry_price.value * (1 + self.trail_percentage)
            self.lowest_price[symbol] = entry_price.value
        
        logger.info(
            f"Trailing stop loss for {symbol} ({position_type}): "
            f"entry={entry_price.value:.2f}, initial_stop={stop_price:.2f} ({self.trail_percentage:.2%})"
        )
        
        return Price(stop_price)
    
    def should_trigger_stop(
        self,
        current_price: Price,
        stop_loss_price: Price,
        position_type: str
    ) -> bool:
        """Check if trailing stop loss should trigger."""
        return FixedPercentageStopLoss.should_trigger_stop(
            self, current_price, stop_loss_price, position_type
        )
    
    def update_stop_loss(
        self,
        symbol: Symbol,
        current_price: Price,
        current_stop: Price,
        market_data: Dict[str, Any]
    ) -> Optional[Price]:
        """Update trailing stop loss."""
        position_type = market_data.get('position_type', 'long').lower()
        
        if position_type == 'long':
            # Update highest price
            if symbol not in self.highest_price:
                self.highest_price[symbol] = current_price.value
            
            if current_price.value > self.highest_price[symbol]:
                self.highest_price[symbol] = current_price.value
                new_stop = current_price.value * (1 - self.trail_percentage)
                
                # Only move stop up, never down
                if new_stop > current_stop.value:
                    logger.info(
                        f"Trailing stop updated for {symbol}: "
                        f"{current_stop.value:.2f} -> {new_stop:.2f}"
                    )
                    return Price(new_stop)
        
        else:  # short position
            # Update lowest price
            if symbol not in self.lowest_price:
                self.lowest_price[symbol] = current_price.value
            
            if current_price.value < self.lowest_price[symbol]:
                self.lowest_price[symbol] = current_price.value
                new_stop = current_price.value * (1 + self.trail_percentage)
                
                # Only move stop down, never up
                if new_stop < current_stop.value:
                    logger.info(
                        f"Trailing stop updated for {symbol}: "
                        f"{current_stop.value:.2f} -> {new_stop:.2f}"
                    )
                    return Price(new_stop)
        
        return None


class ATRBasedStopLoss(BaseStopLoss):
    """ATR-based stop loss."""
    
    def __init__(self, atr_multiplier: Decimal = None, config=None):
        self.config = config or get_risk_config()
        self.atr_multiplier = atr_multiplier or self.config.volatility_multiplier
    
    def calculate_stop_loss(
        self,
        symbol: Symbol,
        entry_price: Price,
        position_size: Quantity,
        position_type: str,
        market_data: Dict[str, Any]
    ) -> Price:
        """Calculate ATR-based stop loss."""
        atr_value = market_data.get('atr', Decimal('1.0'))
        stop_distance = atr_value * self.atr_multiplier
        
        if position_type.lower() == 'long':
            stop_price = entry_price.value - stop_distance
        else:  # short position
            stop_price = entry_price.value + stop_distance
        
        logger.info(
            f"ATR stop loss for {symbol} ({position_type}): "
            f"entry={entry_price.value:.2f}, atr={atr_value:.2f}, "
            f"stop={stop_price:.2f} (distance={stop_distance:.2f})"
        )
        
        return Price(max(stop_price, Decimal('0.01')))  # Ensure positive price
    
    def should_trigger_stop(
        self,
        current_price: Price,
        stop_loss_price: Price,
        position_type: str
    ) -> bool:
        """Check if ATR stop loss should trigger."""
        return FixedPercentageStopLoss.should_trigger_stop(
            self, current_price, stop_loss_price, position_type
        )
    
    def update_stop_loss(
        self,
        symbol: Symbol,
        current_price: Price,
        current_stop: Price,
        market_data: Dict[str, Any]
    ) -> Optional[Price]:
        """ATR stops can be updated with new ATR values."""
        if 'atr' in market_data:
            position_type = market_data.get('position_type', 'long').lower()
            new_atr = market_data['atr']
            stop_distance = new_atr * self.atr_multiplier
            
            if position_type == 'long':
                new_stop = current_price.value - stop_distance
                # Only move stop up
                if new_stop > current_stop.value:
                    return Price(new_stop)
            else:  # short position
                new_stop = current_price.value + stop_distance
                # Only move stop down
                if new_stop < current_stop.value:
                    return Price(new_stop)
        
        return None


class TimeBasedStopLoss(BaseStopLoss):
    """Time-based stop loss."""
    
    def __init__(self, max_hold_days: int = None, config=None):
        self.config = config or get_risk_config()
        self.max_hold_days = max_hold_days or self.config.max_position_hold_days
        self.position_start_times = {}
    
    def calculate_stop_loss(
        self,
        symbol: Symbol,
        entry_price: Price,
        position_size: Quantity,
        position_type: str,
        market_data: Dict[str, Any]
    ) -> Price:
        """Time-based stops use current price as reference."""
        self.position_start_times[symbol] = datetime.now()
        
        logger.info(
            f"Time-based stop loss for {symbol}: max_hold_days={self.max_hold_days}"
        )
        
        return entry_price  # Will trigger based on time, not price
    
    def should_trigger_stop(
        self,
        current_price: Price,
        stop_loss_price: Price,
        position_type: str
    ) -> bool:
        """Time-based stops trigger based on time."""
        # This method is overridden in the manager to check time
        return False
    
    def should_trigger_time_stop(self, symbol: Symbol) -> bool:
        """Check if time-based stop should trigger."""
        if symbol not in self.position_start_times:
            return False
        
        position_age = datetime.now() - self.position_start_times[symbol]
        return position_age.days >= self.max_hold_days
    
    def update_stop_loss(
        self,
        symbol: Symbol,
        current_price: Price,
        current_stop: Price,
        market_data: Dict[str, Any]
    ) -> Optional[Price]:
        """Time-based stops don't update based on price."""
        return None


class StopLossManager:
    """Manager for all stop loss mechanisms."""
    
    def __init__(self, config=None):
        self.config = config or get_risk_config()
        self.active_stops: Dict[Symbol, StopLossOrder] = {}
        self.stop_loss_types = {
            StopLossType.FIXED_PERCENTAGE: FixedPercentageStopLoss(config=self.config),
            StopLossType.TRAILING_PERCENTAGE: TrailingPercentageStopLoss(config=self.config),
            StopLossType.ATR_BASED: ATRBasedStopLoss(config=self.config),
            StopLossType.TIME_BASED: TimeBasedStopLoss(config=self.config)
        }
    
    def create_stop_loss(
        self,
        symbol: Symbol,
        entry_price: Price,
        position_size: Quantity,
        position_type: str,
        stop_type: StopLossType = StopLossType.FIXED_PERCENTAGE,
        market_data: Dict[str, Any] = None
    ) -> StopLossOrder:
        """Create a stop loss order."""
        market_data = market_data or {}
        
        stop_loss_handler = self.stop_loss_types[stop_type]
        stop_price = stop_loss_handler.calculate_stop_loss(
            symbol, entry_price, position_size, position_type, market_data
        )
        
        stop_order = StopLossOrder(
            symbol=symbol,
            position_type=position_type,
            stop_price=stop_price,
            stop_type=stop_type,
            created_at=datetime.now(),
            metadata=market_data.copy()
        )
        
        self.active_stops[symbol] = stop_order
        
        logger.info(
            f"Created {stop_type.value} stop loss for {symbol}: "
            f"entry={entry_price.value:.2f}, stop={stop_price.value:.2f}"
        )
        
        return stop_order
    
    def check_stop_triggers(
        self,
        current_prices: Dict[Symbol, Price]
    ) -> List[StopLossOrder]:
        """Check all active stops for triggers."""
        triggered_stops = []
        
        for symbol, stop_order in self.active_stops.items():
            if stop_order.triggered:
                continue
            
            current_price = current_prices.get(symbol)
            if current_price is None:
                continue
            
            stop_handler = self.stop_loss_types[stop_order.stop_type]
            
            # Check price-based triggers
            should_trigger = stop_handler.should_trigger_stop(
                current_price, stop_order.stop_price, stop_order.position_type
            )
            
            # Check time-based triggers
            if stop_order.stop_type == StopLossType.TIME_BASED:
                time_handler = stop_handler
                should_trigger = time_handler.should_trigger_time_stop(symbol)
            
            if should_trigger:
                stop_order.triggered = True
                stop_order.triggered_at = datetime.now()
                stop_order.triggered_price = current_price
                triggered_stops.append(stop_order)
                
                # Publish risk event
                event = RiskEventFactory.create_stop_loss_triggered_event(
                    symbol=symbol,
                    current_price=float(current_price.value),
                    stop_loss_price=float(stop_order.stop_price.value),
                    action_taken="position_close_required"
                )
                publish_risk_event(event)
                
                logger.warning(
                    f"Stop loss triggered for {symbol}: "
                    f"type={stop_order.stop_type.value}, "
                    f"trigger_price={current_price.value:.2f}, "
                    f"stop_price={stop_order.stop_price.value:.2f}"
                )
        
        return triggered_stops
    
    def update_trailing_stops(
        self,
        current_prices: Dict[Symbol, Price],
        market_data: Dict[Symbol, Dict[str, Any]] = None
    ) -> Dict[Symbol, Price]:
        """Update trailing stop losses."""
        market_data = market_data or {}
        updated_stops = {}
        
        for symbol, stop_order in self.active_stops.items():
            if stop_order.triggered:
                continue
            
            current_price = current_prices.get(symbol)
            if current_price is None:
                continue
            
            stop_handler = self.stop_loss_types[stop_order.stop_type]
            symbol_market_data = market_data.get(symbol, {})
            symbol_market_data['position_type'] = stop_order.position_type
            
            new_stop_price = stop_handler.update_stop_loss(
                symbol, current_price, stop_order.stop_price, symbol_market_data
            )
            
            if new_stop_price is not None:
                stop_order.stop_price = new_stop_price
                updated_stops[symbol] = new_stop_price
                
                logger.info(
                    f"Updated stop loss for {symbol}: new_stop={new_stop_price.value:.2f}"
                )
        
        return updated_stops
    
    def remove_stop_loss(self, symbol: Symbol) -> Optional[StopLossOrder]:
        """Remove stop loss for a symbol."""
        return self.active_stops.pop(symbol, None)
    
    def get_active_stops(self) -> Dict[Symbol, StopLossOrder]:
        """Get all active stop losses."""
        return {symbol: stop for symbol, stop in self.active_stops.items() 
                if not stop.triggered}
    
    def get_stop_loss_info(self, symbol: Symbol) -> Optional[StopLossOrder]:
        """Get stop loss information for a symbol."""
        return self.active_stops.get(symbol)


class StopLossFactory:
    """Factory for creating stop loss handlers."""
    
    @staticmethod
    def create_stop_loss_handler(
        stop_type: StopLossType,
        config=None,
        **kwargs
    ) -> BaseStopLoss:
        """Create a stop loss handler."""
        config = config or get_risk_config()
        
        if stop_type == StopLossType.FIXED_PERCENTAGE:
            return FixedPercentageStopLoss(
                stop_percentage=kwargs.get('stop_percentage'),
                config=config
            )
        elif stop_type == StopLossType.TRAILING_PERCENTAGE:
            return TrailingPercentageStopLoss(
                trail_percentage=kwargs.get('trail_percentage'),
                config=config
            )
        elif stop_type == StopLossType.ATR_BASED:
            return ATRBasedStopLoss(
                atr_multiplier=kwargs.get('atr_multiplier'),
                config=config
            )
        elif stop_type == StopLossType.TIME_BASED:
            return TimeBasedStopLoss(
                max_hold_days=kwargs.get('max_hold_days'),
                config=config
            )
        else:
            raise ValueError(f"Unknown stop loss type: {stop_type}")


# Global stop loss manager
_stop_loss_manager = None


def get_stop_loss_manager() -> StopLossManager:
    """Get global stop loss manager."""
    global _stop_loss_manager
    if _stop_loss_manager is None:
        _stop_loss_manager = StopLossManager()
    return _stop_loss_manager