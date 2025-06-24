"""
Position entity - Represents a trading position.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from ..value_objects import Symbol, Quantity, Price


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


@dataclass
class Position:
    """Trading position entity."""
    
    id: str
    symbol: Symbol
    side: PositionSide
    quantity: Quantity
    avg_price: Price
    current_price: Optional[Price] = None
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    status: PositionStatus = PositionStatus.OPEN
    
    def __post_init__(self):
        """Validate position data."""
        if self.quantity.value <= 0:
            raise ValueError("Position quantity must be positive")
        if self.avg_price.value <= 0:
            raise ValueError("Average price must be positive")
    
    @property
    def market_value(self) -> Decimal:
        """Calculate current market value."""
        if self.current_price is None:
            return self.avg_price.value * self.quantity.value
        return self.current_price.value * self.quantity.value
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L."""
        if self.current_price is None:
            return Decimal('0')
        
        price_diff = self.current_price.value - self.avg_price.value
        if self.side == PositionSide.SHORT:
            price_diff = -price_diff
        
        return price_diff * self.quantity.value
    
    @property
    def unrealized_pnl_percent(self) -> Decimal:
        """Calculate unrealized P&L percentage."""
        if self.current_price is None or self.avg_price.value == 0:
            return Decimal('0')
        
        cost_basis = self.avg_price.value * self.quantity.value
        return (self.unrealized_pnl / cost_basis) * 100
    
    def update_price(self, price: Price) -> None:
        """Update current market price."""
        self.current_price = price
    
    def close_position(self, close_price: Price, close_time: datetime) -> None:
        """Close the position."""
        self.current_price = close_price
        self.closed_at = close_time
        self.status = PositionStatus.CLOSED
    
    def reduce_quantity(self, reduction: Quantity) -> None:
        """Reduce position quantity (partial close)."""
        if reduction.value >= self.quantity.value:
            raise ValueError("Reduction cannot be greater than current quantity")
        
        self.quantity = Quantity(self.quantity.value - reduction.value)
        if self.quantity.value == 0:
            self.status = PositionStatus.CLOSED
        else:
            self.status = PositionStatus.PARTIAL