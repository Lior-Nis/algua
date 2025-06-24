"""
Portfolio entity - Represents a trading portfolio.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from ..value_objects import Money, Symbol
from .position import Position


@dataclass
class Portfolio:
    """Portfolio entity containing positions and cash."""
    
    id: str
    name: str
    initial_capital: Money
    cash: Money
    positions: Dict[str, Position] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate portfolio data."""
        if self.initial_capital.amount <= 0:
            raise ValueError("Initial capital must be positive")
        
        if self.cash.currency != self.initial_capital.currency:
            raise ValueError("Cash and initial capital must have same currency")
    
    @property
    def total_value(self) -> Money:
        """Calculate total portfolio value."""
        positions_value = sum(
            position.market_value 
            for position in self.positions.values()
        )
        return Money(self.cash.amount + positions_value, self.cash.currency)
    
    @property
    def invested_value(self) -> Money:
        """Calculate total invested value."""
        positions_value = sum(
            position.market_value 
            for position in self.positions.values()
        )
        return Money(positions_value, self.cash.currency)
    
    @property
    def unrealized_pnl(self) -> Money:
        """Calculate total unrealized P&L."""
        total_pnl = sum(
            position.unrealized_pnl 
            for position in self.positions.values()
        )
        return Money(total_pnl, self.cash.currency)
    
    @property
    def total_return(self) -> Decimal:
        """Calculate total return percentage."""
        if self.initial_capital.amount == 0:
            return Decimal('0')
        
        current_value = self.total_value.amount
        return ((current_value - self.initial_capital.amount) / self.initial_capital.amount) * 100
    
    @property
    def cash_percentage(self) -> Decimal:
        """Calculate cash allocation percentage."""
        total_value = self.total_value.amount
        if total_value == 0:
            return Decimal('100')
        return (self.cash.amount / total_value) * 100
    
    def add_position(self, position: Position) -> None:
        """Add or update a position in the portfolio."""
        symbol_key = str(position.symbol)
        
        if symbol_key in self.positions:
            # Merge with existing position
            existing = self.positions[symbol_key]
            if existing.side != position.side:
                raise ValueError("Cannot merge positions with different sides")
            
            # Calculate new weighted average price
            total_cost = (existing.avg_price.value * existing.quantity.value + 
                         position.avg_price.value * position.quantity.value)
            total_quantity = existing.quantity.value + position.quantity.value
            
            from ..value_objects import Price, Quantity
            new_avg_price = Price(total_cost / total_quantity)
            new_quantity = Quantity(total_quantity)
            
            existing.avg_price = new_avg_price
            existing.quantity = new_quantity
        else:
            self.positions[symbol_key] = position
    
    def remove_position(self, symbol: Symbol) -> Optional[Position]:
        """Remove a position from the portfolio."""
        symbol_key = str(symbol)
        return self.positions.pop(symbol_key, None)
    
    def get_position(self, symbol: Symbol) -> Optional[Position]:
        """Get a position for a symbol."""
        symbol_key = str(symbol)
        return self.positions.get(symbol_key)
    
    def get_allocation_by_symbol(self) -> Dict[str, Decimal]:
        """Get allocation percentages by symbol."""
        total_value = self.total_value.amount
        if total_value == 0:
            return {"CASH": Decimal('100')}
        
        allocations = {}
        
        # Position allocations
        for symbol_key, position in self.positions.items():
            percentage = (position.market_value / total_value) * 100
            allocations[symbol_key] = percentage
        
        # Cash allocation
        cash_percentage = (self.cash.amount / total_value) * 100
        allocations['CASH'] = cash_percentage
        
        return allocations
    
    def update_cash(self, amount: Money) -> None:
        """Update cash balance."""
        if amount.currency != self.cash.currency:
            raise ValueError("Currency mismatch")
        
        new_amount = self.cash.amount + amount.amount
        if new_amount < 0:
            raise ValueError("Insufficient cash")
        
        self.cash = Money(new_amount, self.cash.currency)
    
    def get_position_summary(self) -> List[Dict]:
        """Get summary of all positions."""
        summary = []
        
        for symbol_key, position in self.positions.items():
            summary.append({
                'symbol': symbol_key,
                'side': position.side.value,
                'quantity': float(position.quantity.value),
                'avg_price': float(position.avg_price.value),
                'current_price': float(position.current_price.value) if position.current_price else None,
                'market_value': float(position.market_value),
                'unrealized_pnl': float(position.unrealized_pnl),
                'unrealized_pnl_percent': float(position.unrealized_pnl_percent),
                'status': position.status.value
            })
        
        return summary