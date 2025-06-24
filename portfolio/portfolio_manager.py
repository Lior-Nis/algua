"""
Portfolio management and tracking.
"""

from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass, field

from domain.entities import Position
from domain.value_objects import Symbol, Money, Price
from configs.settings import get_settings


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""
    timestamp: datetime
    total_value: Money
    cash: Money
    positions: List[Position]
    daily_pnl: Money
    total_pnl: Money
    
    @property
    def invested_value(self) -> Money:
        """Calculate total invested value."""
        total = Decimal('0')
        for position in self.positions:
            total += position.market_value
        return Money(total, self.total_value.currency)
    
    @property
    def allocation_percentages(self) -> Dict[str, Decimal]:
        """Calculate allocation percentages by symbol."""
        if self.total_value.amount == 0:
            return {}
        
        allocations = {}
        for position in self.positions:
            symbol_str = str(position.symbol)
            percentage = (position.market_value / self.total_value.amount) * 100
            allocations[symbol_str] = percentage
        
        # Add cash allocation
        cash_percentage = (self.cash.amount / self.total_value.amount) * 100
        allocations['CASH'] = cash_percentage
        
        return allocations


class PortfolioManager:
    """Manage portfolio positions and calculations."""
    
    def __init__(self, initial_cash: Money):
        self.settings = get_settings()
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[dict] = []
        self._snapshots: List[PortfolioSnapshot] = []
    
    def add_position(self, position: Position) -> None:
        """Add or update a position."""
        symbol_key = str(position.symbol)
        
        if symbol_key in self.positions:
            # Merge with existing position
            existing = self.positions[symbol_key]
            if existing.side != position.side:
                raise ValueError("Cannot merge positions with different sides")
            
            # Calculate new average price
            total_cost = (existing.avg_price.value * existing.quantity.value + 
                         position.avg_price.value * position.quantity.value)
            total_quantity = existing.quantity.value + position.quantity.value
            new_avg_price = Price(total_cost / total_quantity)
            
            # Update existing position
            existing.quantity = existing.quantity + position.quantity
            existing.avg_price = new_avg_price
        else:
            self.positions[symbol_key] = position
    
    def remove_position(self, symbol: Symbol) -> Optional[Position]:
        """Remove and return a position."""
        symbol_key = str(symbol)
        return self.positions.pop(symbol_key, None)
    
    def get_position(self, symbol: Symbol) -> Optional[Position]:
        """Get position for a symbol."""
        symbol_key = str(symbol)
        return self.positions.get(symbol_key)
    
    def update_market_prices(self, prices: Dict[Symbol, Price]) -> None:
        """Update market prices for all positions."""
        for symbol, price in prices.items():
            position = self.get_position(symbol)
            if position:
                position.update_price(price)
    
    def calculate_total_value(self) -> Money:
        """Calculate total portfolio value."""
        total = self.cash.amount
        
        for position in self.positions.values():
            total += position.market_value
        
        return Money(total, self.cash.currency)
    
    def calculate_unrealized_pnl(self) -> Money:
        """Calculate total unrealized P&L."""
        total_pnl = Decimal('0')
        
        for position in self.positions.values():
            total_pnl += position.unrealized_pnl
        
        return Money(total_pnl, self.cash.currency)
    
    def get_allocation_by_symbol(self) -> Dict[str, Decimal]:
        """Get current allocation percentages by symbol."""
        total_value = self.calculate_total_value()
        if total_value.amount == 0:
            return {}
        
        allocations = {}
        for symbol_key, position in self.positions.items():
            percentage = (position.market_value / total_value.amount) * 100
            allocations[symbol_key] = percentage
        
        # Add cash allocation
        cash_percentage = (self.cash.amount / total_value.amount) * 100
        allocations['CASH'] = cash_percentage
        
        return allocations
    
    def check_risk_limits(self) -> List[str]:
        """Check if portfolio violates risk limits."""
        violations = []
        total_value = self.calculate_total_value()
        
        if total_value.amount == 0:
            return violations
        
        # Check maximum position size
        max_position_size = Decimal(str(self.settings.max_position_size))
        for symbol_key, position in self.positions.items():
            position_percentage = position.market_value / total_value.amount
            if position_percentage > max_position_size:
                violations.append(
                    f"Position {symbol_key} exceeds maximum size: "
                    f"{position_percentage:.2%} > {max_position_size:.2%}"
                )
        
        # Check maximum drawdown (requires historical snapshots)
        if len(self._snapshots) > 1:
            peak_value = max(snapshot.total_value.amount for snapshot in self._snapshots)
            current_value = total_value.amount
            drawdown = (peak_value - current_value) / peak_value
            
            max_drawdown = Decimal(str(self.settings.max_drawdown))
            if drawdown > max_drawdown:
                violations.append(
                    f"Portfolio drawdown exceeds limit: "
                    f"{drawdown:.2%} > {max_drawdown:.2%}"
                )
        
        return violations
    
    def take_snapshot(self) -> PortfolioSnapshot:
        """Take a snapshot of current portfolio state."""
        total_value = self.calculate_total_value()
        unrealized_pnl = self.calculate_unrealized_pnl()
        
        # Calculate daily P&L (requires previous snapshot)
        daily_pnl = Money(Decimal('0'), self.cash.currency)
        if self._snapshots:
            last_snapshot = self._snapshots[-1]
            daily_pnl = Money(
                total_value.amount - last_snapshot.total_value.amount,
                self.cash.currency
            )
        
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value=total_value,
            cash=self.cash,
            positions=list(self.positions.values()),
            daily_pnl=daily_pnl,
            total_pnl=unrealized_pnl
        )
        
        self._snapshots.append(snapshot)
        return snapshot
    
    def get_historical_snapshots(self, days: int = 30) -> List[PortfolioSnapshot]:
        """Get historical portfolio snapshots."""
        if days <= 0:
            return self._snapshots
        
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
        
        return [
            snapshot for snapshot in self._snapshots 
            if snapshot.timestamp >= cutoff_date
        ]
    
    def rebalance_to_target_allocation(
        self, 
        target_allocations: Dict[str, Decimal]
    ) -> List[dict]:
        """
        Generate rebalancing orders to achieve target allocation.
        
        Args:
            target_allocations: Target allocation percentages by symbol
            
        Returns:
            List of order instructions
        """
        current_allocations = self.get_allocation_by_symbol()
        total_value = self.calculate_total_value()
        orders = []
        
        for symbol_str, target_pct in target_allocations.items():
            if symbol_str == 'CASH':
                continue
                
            current_pct = current_allocations.get(symbol_str, Decimal('0'))
            difference_pct = target_pct - current_pct
            
            if abs(difference_pct) > Decimal('0.01'):  # 1% threshold
                symbol = Symbol.from_string(symbol_str)
                position = self.get_position(symbol)
                
                # Calculate target value and current value
                target_value = total_value.amount * (target_pct / 100)
                current_value = position.market_value if position else Decimal('0')
                
                difference_value = target_value - current_value
                
                if position and position.current_price:
                    quantity_change = abs(difference_value) / position.current_price.value
                    side = "buy" if difference_value > 0 else "sell"
                    
                    orders.append({
                        "symbol": symbol_str,
                        "side": side,
                        "quantity": float(quantity_change),
                        "type": "market",
                        "reason": f"Rebalance: {current_pct:.1f}% -> {target_pct:.1f}%"
                    })
        
        return orders