"""
Order entity - Represents a trading order.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from ..value_objects import Symbol, Quantity, Price


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order entity."""
    
    id: str
    symbol: Symbol
    side: OrderSide
    order_type: OrderType
    quantity: Quantity
    price: Optional[Price] = None  # For limit orders
    stop_price: Optional[Price] = None  # For stop orders
    filled_quantity: Optional[Quantity] = None
    avg_fill_price: Optional[Price] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate order data."""
        if self.quantity.value <= 0:
            raise ValueError("Order quantity must be positive")
        
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError("Limit orders require a price")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError("Stop orders require a stop price")
        
        if self.filled_quantity is None:
            self.filled_quantity = Quantity(Decimal('0'))
        
        if self.created_at is None:
            self.created_at = datetime.now()
    
    @property
    def is_buy(self) -> bool:
        """Check if order is a buy order."""
        return self.side == OrderSide.BUY
    
    @property
    def is_sell(self) -> bool:
        """Check if order is a sell order."""
        return self.side == OrderSide.SELL
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return self.status == OrderStatus.PARTIALLY_FILLED
    
    @property
    def remaining_quantity(self) -> Quantity:
        """Calculate remaining quantity to fill."""
        return self.quantity - self.filled_quantity
    
    @property
    def fill_percentage(self) -> Decimal:
        """Calculate fill percentage."""
        if self.quantity.value == 0:
            return Decimal('0')
        return (self.filled_quantity.value / self.quantity.value) * 100
    
    @property
    def total_value(self) -> Optional[Decimal]:
        """Calculate total order value."""
        if self.avg_fill_price and self.filled_quantity:
            return self.avg_fill_price.value * self.filled_quantity.value
        elif self.price:
            return self.price.value * self.quantity.value
        return None
    
    def fill(self, quantity: Quantity, price: Price, fill_time: datetime = None) -> None:
        """
        Fill the order partially or completely.
        
        Args:
            quantity: Quantity being filled
            price: Fill price
            fill_time: Time of fill
        """
        if fill_time is None:
            fill_time = datetime.now()
        
        if quantity.value <= 0:
            raise ValueError("Fill quantity must be positive")
        
        if self.filled_quantity.value + quantity.value > self.quantity.value:
            raise ValueError("Cannot fill more than order quantity")
        
        # Update average fill price
        if self.filled_quantity.value == 0:
            self.avg_fill_price = price
        else:
            total_filled_value = (self.avg_fill_price.value * self.filled_quantity.value + 
                                price.value * quantity.value)
            new_filled_quantity = self.filled_quantity.value + quantity.value
            self.avg_fill_price = Price(total_filled_value / new_filled_quantity)
        
        # Update quantities and status
        self.filled_quantity = Quantity(self.filled_quantity.value + quantity.value)
        
        if self.filled_quantity.value >= self.quantity.value:
            self.status = OrderStatus.FILLED
            self.filled_at = fill_time
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.updated_at = fill_time
    
    def cancel(self, cancel_time: datetime = None) -> None:
        """Cancel the order."""
        if cancel_time is None:
            cancel_time = datetime.now()
        
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise ValueError(f"Cannot cancel order with status {self.status.value}")
        
        self.status = OrderStatus.CANCELLED
        self.updated_at = cancel_time
    
    def reject(self, reject_time: datetime = None) -> None:
        """Reject the order."""
        if reject_time is None:
            reject_time = datetime.now()
        
        self.status = OrderStatus.REJECTED
        self.updated_at = reject_time