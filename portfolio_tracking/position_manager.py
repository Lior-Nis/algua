"""
Position management and tracking system.
"""

from typing import Dict, List, Optional, Tuple, Set
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading

from domain.value_objects import Symbol, Price, Quantity, Money
from order_management import Order, Fill, OrderSide
from utils.logging import get_logger

logger = get_logger(__name__)


class PositionType(Enum):
    """Types of positions."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class PositionStatus(Enum):
    """Position status."""
    OPENING = "opening"        # Position being established
    OPEN = "open"             # Active position
    CLOSING = "closing"       # Position being closed
    CLOSED = "closed"         # Position fully closed
    PARTIALLY_CLOSED = "partially_closed"  # Position partially closed


@dataclass
class Position:
    """Individual position in a security."""
    symbol: Symbol
    quantity: Quantity
    average_price: Price
    current_price: Price
    position_type: PositionType
    status: PositionStatus
    
    # Timestamps
    opened_at: datetime
    last_updated: datetime
    closed_at: Optional[datetime] = None
    
    # Financial details
    total_cost: Money = field(init=False)
    market_value: Money = field(init=False)
    unrealized_pnl: Money = field(init=False)
    realized_pnl: Money = field(default_factory=lambda: Money(Decimal('0')))
    
    # Trading details
    total_commission: Money = field(default_factory=lambda: Money(Decimal('0')))
    total_fees: Money = field(default_factory=lambda: Money(Decimal('0')))
    
    # Tracking
    fills: List[Fill] = field(default_factory=list)
    orders: List[str] = field(default_factory=list)  # Order IDs
    
    # Risk metrics
    stop_loss_price: Optional[Price] = None
    take_profit_price: Optional[Price] = None
    max_unrealized_pnl: Money = field(default_factory=lambda: Money(Decimal('0')))
    max_unrealized_loss: Money = field(default_factory=lambda: Money(Decimal('0')))
    
    # Metadata
    strategy_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    def __post_init__(self):
        self._calculate_derived_values()
    
    def _calculate_derived_values(self):
        """Calculate derived financial values."""
        # Total cost (negative for short positions)
        if self.position_type == PositionType.LONG:
            self.total_cost = Money(self.quantity.value * self.average_price.value)
            self.market_value = Money(self.quantity.value * self.current_price.value)
            self.unrealized_pnl = Money(self.market_value.amount - self.total_cost.amount)
        elif self.position_type == PositionType.SHORT:
            self.total_cost = Money(-(self.quantity.value * self.average_price.value))
            self.market_value = Money(-(self.quantity.value * self.current_price.value))
            self.unrealized_pnl = Money(self.total_cost.amount - (-self.market_value.amount))
        else:  # FLAT
            self.total_cost = Money(Decimal('0'))
            self.market_value = Money(Decimal('0'))
            self.unrealized_pnl = Money(Decimal('0'))
    
    def update_price(self, new_price: Price, timestamp: datetime = None) -> None:
        """Update current price and recalculate P&L."""
        self.current_price = new_price
        self.last_updated = timestamp or datetime.now()
        self._calculate_derived_values()
        
        # Update max P&L tracking
        if self.unrealized_pnl.amount > self.max_unrealized_pnl.amount:
            self.max_unrealized_pnl = Money(self.unrealized_pnl.amount)
        
        if self.unrealized_pnl.amount < self.max_unrealized_loss.amount:
            self.max_unrealized_loss = Money(self.unrealized_pnl.amount)
    
    def add_fill(self, fill: Fill, order_id: str) -> None:
        """Add a fill to the position."""
        self.fills.append(fill)
        
        if order_id not in self.orders:
            self.orders.append(order_id)
        
        # Update commission
        self.total_commission = Money(self.total_commission.amount + fill.commission.amount)
        
        # Recalculate average price and quantity
        self._recalculate_position_from_fills()
        
        logger.debug(
            f"Added fill to position {self.symbol}: {fill.quantity.value} @ {fill.price.value}"
        )
    
    def _recalculate_position_from_fills(self) -> None:
        """Recalculate position from all fills."""
        if not self.fills:
            return
        
        # Separate buys and sells
        buy_fills = [f for f in self.fills if self._is_buy_fill(f)]
        sell_fills = [f for f in self.fills if not self._is_buy_fill(f)]
        
        # Calculate net position
        total_bought = sum(f.quantity.value for f in buy_fills)
        total_sold = sum(f.quantity.value for f in sell_fills)
        net_quantity = total_bought - total_sold
        
        if net_quantity > 0:
            # Long position
            self.position_type = PositionType.LONG
            self.quantity = Quantity(net_quantity)
            
            # Calculate average buy price
            if buy_fills:
                total_cost = sum(f.quantity.value * f.price.value for f in buy_fills)
                self.average_price = Price(total_cost / total_bought)
            
        elif net_quantity < 0:
            # Short position
            self.position_type = PositionType.SHORT
            self.quantity = Quantity(abs(net_quantity))
            
            # Calculate average short price
            if sell_fills:
                total_proceeds = sum(f.quantity.value * f.price.value for f in sell_fills)
                self.average_price = Price(total_proceeds / total_sold)
        
        else:
            # Flat position
            self.position_type = PositionType.FLAT
            self.quantity = Quantity(Decimal('0'))
            self.status = PositionStatus.CLOSED
            self.closed_at = datetime.now()
        
        # Recalculate derived values
        self._calculate_derived_values()
    
    def _is_buy_fill(self, fill: Fill) -> bool:
        """Check if fill is a buy based on order side."""
        # This would need to be enhanced to look up the actual order
        # For now, assume fill metadata or order lookup
        return hasattr(fill, 'side') and fill.side == OrderSide.BUY
    
    def get_total_pnl(self) -> Money:
        """Get total P&L (realized + unrealized)."""
        return Money(self.realized_pnl.amount + self.unrealized_pnl.amount)
    
    def get_pnl_percentage(self) -> Decimal:
        """Get P&L as percentage of cost basis."""
        if self.total_cost.amount == 0:
            return Decimal('0')
        
        return self.get_total_pnl().amount / abs(self.total_cost.amount)
    
    def get_holding_period(self) -> timedelta:
        """Get holding period."""
        end_time = self.closed_at or datetime.now()
        return end_time - self.opened_at
    
    def is_profitable(self) -> bool:
        """Check if position is currently profitable."""
        return self.get_total_pnl().amount > 0
    
    def get_position_size_pct(self, portfolio_value: Money) -> Decimal:
        """Get position size as percentage of portfolio."""
        if portfolio_value.amount == 0:
            return Decimal('0')
        
        return abs(self.market_value.amount) / portfolio_value.amount
    
    def to_dict(self) -> Dict[str, any]:
        """Convert position to dictionary."""
        return {
            'symbol': str(self.symbol),
            'quantity': float(self.quantity.value),
            'average_price': float(self.average_price.value),
            'current_price': float(self.current_price.value),
            'position_type': self.position_type.value,
            'status': self.status.value,
            'opened_at': self.opened_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'closed_at': self.closed_at.isoformat() if self.closed_at else None,
            'total_cost': float(self.total_cost.amount),
            'market_value': float(self.market_value.amount),
            'unrealized_pnl': float(self.unrealized_pnl.amount),
            'realized_pnl': float(self.realized_pnl.amount),
            'total_pnl': float(self.get_total_pnl().amount),
            'pnl_percentage': float(self.get_pnl_percentage()),
            'total_commission': float(self.total_commission.amount),
            'fills_count': len(self.fills),
            'orders_count': len(self.orders),
            'holding_period_days': self.get_holding_period().days,
            'is_profitable': self.is_profitable(),
            'strategy_id': self.strategy_id,
            'tags': self.tags,
            'notes': self.notes
        }


@dataclass
class PositionSnapshot:
    """Point-in-time snapshot of a position."""
    timestamp: datetime
    symbol: Symbol
    quantity: Quantity
    price: Price
    market_value: Money
    unrealized_pnl: Money
    position_type: PositionType
    notes: Optional[str] = None


@dataclass
class PositionHistory:
    """Complete history of a position."""
    position: Position
    snapshots: List[PositionSnapshot] = field(default_factory=list)
    price_updates: List[Tuple[datetime, Price]] = field(default_factory=list)
    
    def add_snapshot(self, snapshot: PositionSnapshot) -> None:
        """Add position snapshot."""
        self.snapshots.append(snapshot)
    
    def add_price_update(self, timestamp: datetime, price: Price) -> None:
        """Add price update."""
        self.price_updates.append((timestamp, price))
    
    def get_price_series(self) -> List[Tuple[datetime, Decimal]]:
        """Get price time series."""
        return [(ts, price.value) for ts, price in self.price_updates]
    
    def get_pnl_series(self) -> List[Tuple[datetime, Decimal]]:
        """Get P&L time series."""
        return [(snap.timestamp, snap.unrealized_pnl.amount) for snap in self.snapshots]


class PositionManager:
    """Manager for all positions."""
    
    def __init__(self):
        self.positions: Dict[Symbol, Position] = {}
        self.position_histories: Dict[Symbol, PositionHistory] = {}
        self.closed_positions: List[Position] = []
        
        # Configuration
        self.auto_create_positions = True
        self.snapshot_interval_seconds = 60  # 1 minute snapshots
        self.last_snapshot_time = datetime.now()
        
        # Statistics
        self.position_stats = {
            'total_positions_opened': 0,
            'total_positions_closed': 0,
            'active_positions_count': 0,
            'total_realized_pnl': Money(Decimal('0')),
            'total_unrealized_pnl': Money(Decimal('0')),
            'profitable_positions': 0,
            'losing_positions': 0
        }
        
        self._lock = threading.Lock()
    
    def create_position(
        self,
        symbol: Symbol,
        initial_fill: Fill,
        order_id: str,
        strategy_id: Optional[str] = None
    ) -> Position:
        """Create new position from initial fill."""
        with self._lock:
            # Determine position type from fill
            position_type = PositionType.LONG  # Would need order lookup for actual side
            
            position = Position(
                symbol=symbol,
                quantity=initial_fill.quantity,
                average_price=initial_fill.price,
                current_price=initial_fill.price,
                position_type=position_type,
                status=PositionStatus.OPENING,
                opened_at=initial_fill.timestamp,
                last_updated=initial_fill.timestamp,
                strategy_id=strategy_id
            )
            
            position.add_fill(initial_fill, order_id)
            position.status = PositionStatus.OPEN
            
            self.positions[symbol] = position
            self.position_histories[symbol] = PositionHistory(position=position)
            
            # Update statistics
            self.position_stats['total_positions_opened'] += 1
            self.position_stats['active_positions_count'] += 1
            
            # Take initial snapshot
            self._take_snapshot(symbol)
            
            logger.info(
                f"Created new position: {symbol} {position_type.value} "
                f"{position.quantity.value} @ {position.average_price.value}"
            )
            
            return position
    
    def update_position(
        self,
        symbol: Symbol,
        fill: Fill,
        order_id: str
    ) -> Optional[Position]:
        """Update existing position with new fill."""
        with self._lock:
            if symbol not in self.positions:
                if self.auto_create_positions:
                    return self.create_position(symbol, fill, order_id)
                else:
                    logger.warning(f"No existing position for {symbol} and auto-create disabled")
                    return None
            
            position = self.positions[symbol]
            old_quantity = position.quantity.value
            
            position.add_fill(fill, order_id)
            position.last_updated = fill.timestamp
            
            # Check if position was closed
            if position.quantity.value == 0:
                position.status = PositionStatus.CLOSED
                position.closed_at = fill.timestamp
                
                # Move to closed positions
                self.closed_positions.append(position)
                del self.positions[symbol]
                
                # Update statistics
                self.position_stats['active_positions_count'] -= 1
                self.position_stats['total_positions_closed'] += 1
                self.position_stats['total_realized_pnl'] = Money(
                    self.position_stats['total_realized_pnl'].amount + position.realized_pnl.amount
                )
                
                if position.is_profitable():
                    self.position_stats['profitable_positions'] += 1
                else:
                    self.position_stats['losing_positions'] += 1
                
                logger.info(f"Position closed: {symbol} final P&L: ${position.get_total_pnl().amount:.2f}")
            
            else:
                # Position still active
                if old_quantity > position.quantity.value:
                    position.status = PositionStatus.PARTIALLY_CLOSED
                else:
                    position.status = PositionStatus.OPEN
            
            # Take snapshot
            self._take_snapshot(symbol)
            
            logger.debug(f"Updated position: {symbol} new quantity: {position.quantity.value}")
            
            return position
    
    def update_market_prices(self, prices: Dict[Symbol, Price], timestamp: datetime = None) -> None:
        """Update market prices for all positions."""
        timestamp = timestamp or datetime.now()
        
        with self._lock:
            for symbol, position in self.positions.items():
                if symbol in prices:
                    old_pnl = position.unrealized_pnl.amount
                    position.update_price(prices[symbol], timestamp)
                    
                    # Record price update in history
                    if symbol in self.position_histories:
                        self.position_histories[symbol].add_price_update(timestamp, prices[symbol])
                    
                    logger.debug(
                        f"Price update {symbol}: {prices[symbol].value:.2f} "
                        f"P&L: ${old_pnl:.2f} -> ${position.unrealized_pnl.amount:.2f}"
                    )
            
            # Update total unrealized P&L
            total_unrealized = sum(pos.unrealized_pnl.amount for pos in self.positions.values())
            self.position_stats['total_unrealized_pnl'] = Money(total_unrealized)
            
            # Take periodic snapshots
            if (timestamp - self.last_snapshot_time).total_seconds() >= self.snapshot_interval_seconds:
                for symbol in self.positions.keys():
                    self._take_snapshot(symbol)
                self.last_snapshot_time = timestamp
    
    def close_position(
        self,
        symbol: Symbol,
        price: Price,
        timestamp: datetime = None,
        reason: str = "Manual close"
    ) -> Optional[Position]:
        """Manually close a position."""
        with self._lock:
            if symbol not in self.positions:
                logger.warning(f"Cannot close non-existent position: {symbol}")
                return None
            
            position = self.positions[symbol]
            timestamp = timestamp or datetime.now()
            
            # Update to current price
            position.update_price(price, timestamp)
            
            # Calculate realized P&L
            position.realized_pnl = Money(position.unrealized_pnl.amount)
            position.unrealized_pnl = Money(Decimal('0'))
            
            # Mark as closed
            position.status = PositionStatus.CLOSED
            position.closed_at = timestamp
            position.notes = f"{position.notes or ''} {reason}".strip()
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[symbol]
            
            # Update statistics
            self.position_stats['active_positions_count'] -= 1
            self.position_stats['total_positions_closed'] += 1
            self.position_stats['total_realized_pnl'] = Money(
                self.position_stats['total_realized_pnl'].amount + position.realized_pnl.amount
            )
            
            if position.is_profitable():
                self.position_stats['profitable_positions'] += 1
            else:
                self.position_stats['losing_positions'] += 1
            
            logger.info(f"Position manually closed: {symbol} P&L: ${position.realized_pnl.amount:.2f}")
            
            return position
    
    def get_position(self, symbol: Symbol) -> Optional[Position]:
        """Get position for symbol."""
        with self._lock:
            return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[Symbol, Position]:
        """Get all active positions."""
        with self._lock:
            return self.positions.copy()
    
    def get_positions_by_strategy(self, strategy_id: str) -> List[Position]:
        """Get positions for specific strategy."""
        with self._lock:
            return [pos for pos in self.positions.values() if pos.strategy_id == strategy_id]
    
    def get_profitable_positions(self) -> List[Position]:
        """Get all profitable positions."""
        with self._lock:
            return [pos for pos in self.positions.values() if pos.is_profitable()]
    
    def get_losing_positions(self) -> List[Position]:
        """Get all losing positions."""
        with self._lock:
            return [pos for pos in self.positions.values() if not pos.is_profitable()]
    
    def get_position_history(self, symbol: Symbol) -> Optional[PositionHistory]:
        """Get position history."""
        with self._lock:
            return self.position_histories.get(symbol)
    
    def get_total_market_value(self) -> Money:
        """Get total market value of all positions."""
        with self._lock:
            total = sum(abs(pos.market_value.amount) for pos in self.positions.values())
            return Money(total)
    
    def get_total_unrealized_pnl(self) -> Money:
        """Get total unrealized P&L."""
        with self._lock:
            total = sum(pos.unrealized_pnl.amount for pos in self.positions.values())
            return Money(total)
    
    def get_total_realized_pnl(self) -> Money:
        """Get total realized P&L from closed positions."""
        return self.position_stats['total_realized_pnl']
    
    def get_position_count(self) -> int:
        """Get count of active positions."""
        with self._lock:
            return len(self.positions)
    
    def _take_snapshot(self, symbol: Symbol) -> None:
        """Take snapshot of position."""
        if symbol not in self.positions or symbol not in self.position_histories:
            return
        
        position = self.positions[symbol]
        snapshot = PositionSnapshot(
            timestamp=datetime.now(),
            symbol=symbol,
            quantity=position.quantity,
            price=position.current_price,
            market_value=position.market_value,
            unrealized_pnl=position.unrealized_pnl,
            position_type=position.position_type
        )
        
        self.position_histories[symbol].add_snapshot(snapshot)
    
    def cleanup_old_histories(self, retention_days: int = 30) -> int:
        """Clean up old position histories."""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        cleaned_count = 0
        
        with self._lock:
            # Clean up closed position histories
            symbols_to_remove = []
            
            for symbol, history in self.position_histories.items():
                if (symbol not in self.positions and  # Position is closed
                    history.position.closed_at and
                    history.position.closed_at < cutoff_time):
                    symbols_to_remove.append(symbol)
            
            for symbol in symbols_to_remove:
                del self.position_histories[symbol]
                cleaned_count += 1
            
            # Clean up old closed positions
            old_closed_count = len(self.closed_positions)
            self.closed_positions = [
                pos for pos in self.closed_positions
                if pos.closed_at and pos.closed_at >= cutoff_time
            ]
            cleaned_count += old_closed_count - len(self.closed_positions)
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old position records")
        
        return cleaned_count
    
    def get_position_statistics(self) -> Dict[str, any]:
        """Get position management statistics."""
        with self._lock:
            stats = self.position_stats.copy()
            
            # Add current metrics
            stats['current_positions'] = len(self.positions)
            stats['total_market_value'] = float(self.get_total_market_value().amount)
            stats['total_unrealized_pnl'] = float(self.get_total_unrealized_pnl().amount)
            stats['total_realized_pnl'] = float(self.position_stats['total_realized_pnl'].amount)
            
            # Calculate win rate
            total_closed = stats['profitable_positions'] + stats['losing_positions']
            if total_closed > 0:
                stats['win_rate'] = stats['profitable_positions'] / total_closed
            else:
                stats['win_rate'] = 0
            
            # Position distribution
            position_types = defaultdict(int)
            for position in self.positions.values():
                position_types[position.position_type.value] += 1
            stats['position_type_distribution'] = dict(position_types)
            
            return stats


# Global position manager instance
_position_manager = None


def get_position_manager() -> PositionManager:
    """Get global position manager."""
    global _position_manager
    if _position_manager is None:
        _position_manager = PositionManager()
    return _position_manager