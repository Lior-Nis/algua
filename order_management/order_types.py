"""
Order types and order data structures for the order management system.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from decimal import Decimal
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import uuid

from domain.value_objects import Symbol, Price, Quantity, Money
from utils.logging import get_logger

logger = get_logger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class OrderSide(Enum):
    """Order side (buy/sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"                    # Order created but not submitted
    SUBMITTED = "submitted"                # Order submitted to broker/exchange
    ACKNOWLEDGED = "acknowledged"          # Order acknowledged by broker/exchange
    PARTIALLY_FILLED = "partially_filled" # Order partially executed
    FILLED = "filled"                     # Order completely executed
    CANCELED = "canceled"                 # Order canceled
    REJECTED = "rejected"                 # Order rejected
    EXPIRED = "expired"                   # Order expired
    REPLACED = "replaced"                 # Order replaced/modified


class TimeInForce(Enum):
    """Time in force for orders."""
    DAY = "day"           # Good for day
    GTC = "gtc"           # Good till canceled
    IOC = "ioc"           # Immediate or cancel
    FOK = "fok"           # Fill or kill
    GTD = "gtd"           # Good till date


@dataclass
class OrderMetadata:
    """Order metadata and context."""
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    risk_params: Dict[str, Any] = field(default_factory=dict)


class Order(ABC):
    """Base order class."""
    
    def __init__(
        self,
        symbol: Symbol,
        side: OrderSide,
        quantity: Quantity,
        order_type: OrderType,
        time_in_force: TimeInForce = TimeInForce.DAY,
        metadata: Optional[OrderMetadata] = None
    ):
        self.order_id = str(uuid.uuid4())
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.time_in_force = time_in_force
        self.metadata = metadata or OrderMetadata()
        
        # Order lifecycle
        self.status = OrderStatus.PENDING
        self.created_at = datetime.now()
        self.submitted_at: Optional[datetime] = None
        self.filled_at: Optional[datetime] = None
        self.canceled_at: Optional[datetime] = None
        
        # Execution details
        self.filled_quantity = Quantity(Decimal('0'))
        self.remaining_quantity = quantity
        self.average_fill_price: Optional[Price] = None
        self.commission: Optional[Money] = None
        self.fills: List['Fill'] = []
        
        # Broker/exchange details
        self.broker_order_id: Optional[str] = None
        self.exchange_order_id: Optional[str] = None
        self.rejection_reason: Optional[str] = None
        
        # Expiration
        self.expires_at: Optional[datetime] = None
        if time_in_force == TimeInForce.DAY:
            # Set expiration to end of trading day
            self.expires_at = self._get_end_of_trading_day()
        elif time_in_force == TimeInForce.GTD:
            # Will be set by user when creating order
            pass
    
    def _get_end_of_trading_day(self) -> datetime:
        """Get end of trading day for expiration."""
        # Simplified: assume market closes at 4 PM ET
        now = datetime.now()
        end_of_day = now.replace(hour=16, minute=0, second=0, microsecond=0)
        if now > end_of_day:
            end_of_day = end_of_day + timedelta(days=1)
        return end_of_day
    
    @abstractmethod
    def is_executable(self, current_price: Price) -> bool:
        """Check if order can be executed at current price."""
        pass
    
    @abstractmethod
    def get_execution_price(self, current_price: Price) -> Price:
        """Get price at which order should execute."""
        pass
    
    def add_fill(self, fill: 'Fill') -> None:
        """Add a fill to the order."""
        self.fills.append(fill)
        self.filled_quantity = Quantity(self.filled_quantity.value + fill.quantity.value)
        self.remaining_quantity = Quantity(self.quantity.value - self.filled_quantity.value)
        
        # Update average fill price
        if self.fills:
            total_value = sum(fill.quantity.value * fill.price.value for fill in self.fills)
            total_quantity = sum(fill.quantity.value for fill in self.fills)
            self.average_fill_price = Price(total_value / total_quantity)
        
        # Update status
        if self.remaining_quantity.value <= 0:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.now()
        elif self.filled_quantity.value > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        logger.info(
            f"Fill added to order {self.order_id}: {fill.quantity.value} @ {fill.price.value} "
            f"(filled: {self.filled_quantity.value}/{self.quantity.value})"
        )
    
    def cancel(self, reason: Optional[str] = None) -> None:
        """Cancel the order."""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
            logger.warning(f"Cannot cancel order {self.order_id} in status {self.status.value}")
            return
        
        self.status = OrderStatus.CANCELED
        self.canceled_at = datetime.now()
        if reason:
            self.rejection_reason = reason
        
        logger.info(f"Order {self.order_id} canceled: {reason or 'No reason provided'}")
    
    def reject(self, reason: str) -> None:
        """Reject the order."""
        self.status = OrderStatus.REJECTED
        self.rejection_reason = reason
        
        logger.warning(f"Order {self.order_id} rejected: {reason}")
    
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED
        ]
    
    def is_complete(self) -> bool:
        """Check if order is complete (filled, canceled, etc.)."""
        return not self.is_active()
    
    def get_order_value(self) -> Optional[Money]:
        """Get total order value if determinable."""
        if hasattr(self, 'price') and self.price:
            return Money(self.quantity.value * self.price.value)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary representation."""
        return {
            'order_id': self.order_id,
            'symbol': str(self.symbol),
            'side': self.side.value,
            'quantity': float(self.quantity.value),
            'order_type': self.order_type.value,
            'time_in_force': self.time_in_force.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'canceled_at': self.canceled_at.isoformat() if self.canceled_at else None,
            'filled_quantity': float(self.filled_quantity.value),
            'remaining_quantity': float(self.remaining_quantity.value),
            'average_fill_price': float(self.average_fill_price.value) if self.average_fill_price else None,
            'commission': float(self.commission.amount) if self.commission else None,
            'fills_count': len(self.fills),
            'broker_order_id': self.broker_order_id,
            'exchange_order_id': self.exchange_order_id,
            'rejection_reason': self.rejection_reason,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'metadata': {
                'strategy_id': self.metadata.strategy_id,
                'signal_id': self.metadata.signal_id,
                'parent_order_id': self.metadata.parent_order_id,
                'tags': self.metadata.tags,
                'notes': self.metadata.notes
            }
        }


class MarketOrder(Order):
    """Market order - executes immediately at current market price."""
    
    def __init__(
        self,
        symbol: Symbol,
        side: OrderSide,
        quantity: Quantity,
        time_in_force: TimeInForce = TimeInForce.IOC,  # Market orders typically IOC
        metadata: Optional[OrderMetadata] = None
    ):
        super().__init__(symbol, side, quantity, OrderType.MARKET, time_in_force, metadata)
    
    def is_executable(self, current_price: Price) -> bool:
        """Market orders are always executable if market is open."""
        return True
    
    def get_execution_price(self, current_price: Price) -> Price:
        """Market orders execute at current market price."""
        return current_price


class LimitOrder(Order):
    """Limit order - executes only at specified price or better."""
    
    def __init__(
        self,
        symbol: Symbol,
        side: OrderSide,
        quantity: Quantity,
        price: Price,
        time_in_force: TimeInForce = TimeInForce.GTC,
        metadata: Optional[OrderMetadata] = None
    ):
        super().__init__(symbol, side, quantity, OrderType.LIMIT, time_in_force, metadata)
        self.price = price
    
    def is_executable(self, current_price: Price) -> bool:
        """Limit order executes when price condition is met."""
        if self.side == OrderSide.BUY:
            # Buy limit: execute when market price <= limit price
            return current_price.value <= self.price.value
        else:
            # Sell limit: execute when market price >= limit price
            return current_price.value >= self.price.value
    
    def get_execution_price(self, current_price: Price) -> Price:
        """Limit orders execute at limit price or better."""
        if self.side == OrderSide.BUY:
            # Buy at the better of current price or limit price
            return Price(min(current_price.value, self.price.value))
        else:
            # Sell at the better of current price or limit price
            return Price(max(current_price.value, self.price.value))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert limit order to dictionary."""
        data = super().to_dict()
        data['price'] = float(self.price.value)
        return data


class StopOrder(Order):
    """Stop order - becomes market order when stop price is hit."""
    
    def __init__(
        self,
        symbol: Symbol,
        side: OrderSide,
        quantity: Quantity,
        stop_price: Price,
        time_in_force: TimeInForce = TimeInForce.GTC,
        metadata: Optional[OrderMetadata] = None
    ):
        super().__init__(symbol, side, quantity, OrderType.STOP, time_in_force, metadata)
        self.stop_price = stop_price
        self.triggered = False
    
    def is_executable(self, current_price: Price) -> bool:
        """Stop order executes when stop price is hit."""
        if not self.triggered:
            if self.side == OrderSide.BUY:
                # Buy stop: trigger when price rises above stop price
                self.triggered = current_price.value >= self.stop_price.value
            else:
                # Sell stop: trigger when price falls below stop price
                self.triggered = current_price.value <= self.stop_price.value
            
            if self.triggered:
                logger.info(f"Stop order {self.order_id} triggered at price {current_price.value}")
        
        return self.triggered
    
    def get_execution_price(self, current_price: Price) -> Price:
        """Stop orders execute at current market price once triggered."""
        return current_price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stop order to dictionary."""
        data = super().to_dict()
        data['stop_price'] = float(self.stop_price.value)
        data['triggered'] = self.triggered
        return data


class StopLimitOrder(Order):
    """Stop-limit order - becomes limit order when stop price is hit."""
    
    def __init__(
        self,
        symbol: Symbol,
        side: OrderSide,
        quantity: Quantity,
        stop_price: Price,
        limit_price: Price,
        time_in_force: TimeInForce = TimeInForce.GTC,
        metadata: Optional[OrderMetadata] = None
    ):
        super().__init__(symbol, side, quantity, OrderType.STOP_LIMIT, time_in_force, metadata)
        self.stop_price = stop_price
        self.limit_price = limit_price
        self.triggered = False
    
    def is_executable(self, current_price: Price) -> bool:
        """Stop-limit order executes when both stop and limit conditions are met."""
        # First check if stop is triggered
        if not self.triggered:
            if self.side == OrderSide.BUY:
                self.triggered = current_price.value >= self.stop_price.value
            else:
                self.triggered = current_price.value <= self.stop_price.value
            
            if self.triggered:
                logger.info(f"Stop-limit order {self.order_id} triggered at price {current_price.value}")
        
        # Then check limit condition if triggered
        if self.triggered:
            if self.side == OrderSide.BUY:
                return current_price.value <= self.limit_price.value
            else:
                return current_price.value >= self.limit_price.value
        
        return False
    
    def get_execution_price(self, current_price: Price) -> Price:
        """Stop-limit orders execute at limit price or better."""
        if self.side == OrderSide.BUY:
            return Price(min(current_price.value, self.limit_price.value))
        else:
            return Price(max(current_price.value, self.limit_price.value))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stop-limit order to dictionary."""
        data = super().to_dict()
        data['stop_price'] = float(self.stop_price.value)
        data['limit_price'] = float(self.limit_price.value)
        data['triggered'] = self.triggered
        return data


class OrderFactory:
    """Factory for creating orders."""
    
    @staticmethod
    def create_market_order(
        symbol: Symbol,
        side: OrderSide,
        quantity: Quantity,
        time_in_force: TimeInForce = TimeInForce.IOC,
        metadata: Optional[OrderMetadata] = None
    ) -> MarketOrder:
        """Create a market order."""
        return MarketOrder(symbol, side, quantity, time_in_force, metadata)
    
    @staticmethod
    def create_limit_order(
        symbol: Symbol,
        side: OrderSide,
        quantity: Quantity,
        price: Price,
        time_in_force: TimeInForce = TimeInForce.GTC,
        metadata: Optional[OrderMetadata] = None
    ) -> LimitOrder:
        """Create a limit order."""
        return LimitOrder(symbol, side, quantity, price, time_in_force, metadata)
    
    @staticmethod
    def create_stop_order(
        symbol: Symbol,
        side: OrderSide,
        quantity: Quantity,
        stop_price: Price,
        time_in_force: TimeInForce = TimeInForce.GTC,
        metadata: Optional[OrderMetadata] = None
    ) -> StopOrder:
        """Create a stop order."""
        return StopOrder(symbol, side, quantity, stop_price, time_in_force, metadata)
    
    @staticmethod
    def create_stop_limit_order(
        symbol: Symbol,
        side: OrderSide,
        quantity: Quantity,
        stop_price: Price,
        limit_price: Price,
        time_in_force: TimeInForce = TimeInForce.GTC,
        metadata: Optional[OrderMetadata] = None
    ) -> StopLimitOrder:
        """Create a stop-limit order."""
        return StopLimitOrder(symbol, side, quantity, stop_price, limit_price, time_in_force, metadata)
    
    @staticmethod
    def create_order_from_dict(order_data: Dict[str, Any]) -> Order:
        """Create order from dictionary representation."""
        order_type = OrderType(order_data['order_type'])
        symbol = Symbol(order_data['symbol'])
        side = OrderSide(order_data['side'])
        quantity = Quantity(Decimal(str(order_data['quantity'])))
        time_in_force = TimeInForce(order_data.get('time_in_force', 'day'))
        
        # Create metadata if present
        metadata = None
        if 'metadata' in order_data:
            meta_data = order_data['metadata']
            metadata = OrderMetadata(
                strategy_id=meta_data.get('strategy_id'),
                signal_id=meta_data.get('signal_id'),
                parent_order_id=meta_data.get('parent_order_id'),
                tags=meta_data.get('tags', []),
                notes=meta_data.get('notes')
            )
        
        # Create appropriate order type
        if order_type == OrderType.MARKET:
            return OrderFactory.create_market_order(symbol, side, quantity, time_in_force, metadata)
        elif order_type == OrderType.LIMIT:
            price = Price(Decimal(str(order_data['price'])))
            return OrderFactory.create_limit_order(symbol, side, quantity, price, time_in_force, metadata)
        elif order_type == OrderType.STOP:
            stop_price = Price(Decimal(str(order_data['stop_price'])))
            return OrderFactory.create_stop_order(symbol, side, quantity, stop_price, time_in_force, metadata)
        elif order_type == OrderType.STOP_LIMIT:
            stop_price = Price(Decimal(str(order_data['stop_price'])))
            limit_price = Price(Decimal(str(order_data['limit_price'])))
            return OrderFactory.create_stop_limit_order(symbol, side, quantity, stop_price, limit_price, time_in_force, metadata)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")


# Import Fill here to avoid circular imports
from .fill_handler import Fill