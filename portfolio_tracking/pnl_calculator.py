"""
P&L calculation and attribution system.
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading

from domain.value_objects import Symbol, Price, Quantity, Money
from .position_manager import Position, PositionManager, get_position_manager
from order_management import Fill
from utils.logging import get_logger

logger = get_logger(__name__)


class PnLType(Enum):
    """Types of P&L."""
    REALIZED = "realized"
    UNREALIZED = "unrealized"
    TOTAL = "total"


class AttributionType(Enum):
    """P&L attribution types."""
    SYMBOL = "symbol"
    STRATEGY = "strategy"
    SECTOR = "sector"
    ASSET_CLASS = "asset_class"
    TIME_PERIOD = "time_period"


@dataclass
class RealizedPnL:
    """Realized P&L from closed positions."""
    symbol: Symbol
    realized_amount: Money
    realized_date: datetime
    holding_period: timedelta
    quantity_closed: Quantity
    average_open_price: Price
    average_close_price: Price
    commission_paid: Money
    strategy_id: Optional[str] = None
    
    def get_return_percentage(self) -> Decimal:
        """Get return percentage."""
        if self.average_open_price.value == 0:
            return Decimal('0')
        
        price_diff = self.average_close_price.value - self.average_open_price.value
        return price_diff / self.average_open_price.value
    
    def get_annualized_return(self) -> Decimal:
        """Get annualized return percentage."""
        if self.holding_period.days == 0:
            return Decimal('0')
        
        daily_return = self.get_return_percentage()
        return daily_return * Decimal('365') / Decimal(str(self.holding_period.days))


@dataclass
class UnrealizedPnL:
    """Unrealized P&L from open positions."""
    symbol: Symbol
    unrealized_amount: Money
    current_price: Price
    cost_basis: Money
    quantity: Quantity
    position_value: Money
    last_updated: datetime
    strategy_id: Optional[str] = None
    
    def get_return_percentage(self) -> Decimal:
        """Get unrealized return percentage."""
        if self.cost_basis.amount == 0:
            return Decimal('0')
        
        return self.unrealized_amount.amount / abs(self.cost_basis.amount)


@dataclass
class TotalPnL:
    """Total P&L combining realized and unrealized."""
    realized_pnl: Money
    unrealized_pnl: Money
    total_pnl: Money
    total_cost_basis: Money
    total_market_value: Money
    calculation_time: datetime
    
    def get_total_return_percentage(self) -> Decimal:
        """Get total return percentage."""
        if self.total_cost_basis.amount == 0:
            return Decimal('0')
        
        return self.total_pnl.amount / abs(self.total_cost_basis.amount)


@dataclass
class PnLSnapshot:
    """Point-in-time P&L snapshot."""
    timestamp: datetime
    realized_pnl: Money
    unrealized_pnl: Money
    total_pnl: Money
    daily_pnl_change: Money
    positions_count: int
    market_value: Money
    cash_balance: Money
    total_portfolio_value: Money
    
    # Attribution breakdowns
    pnl_by_symbol: Dict[Symbol, Money] = field(default_factory=dict)
    pnl_by_strategy: Dict[str, Money] = field(default_factory=dict)
    pnl_by_sector: Dict[str, Money] = field(default_factory=dict)


@dataclass
class PnLHistory:
    """Historical P&L data."""
    snapshots: List[PnLSnapshot] = field(default_factory=list)
    daily_returns: List[Tuple[date, Decimal]] = field(default_factory=list)
    realized_trades: List[RealizedPnL] = field(default_factory=list)
    
    def add_snapshot(self, snapshot: PnLSnapshot) -> None:
        """Add P&L snapshot."""
        self.snapshots.append(snapshot)
        
        # Calculate daily return if we have previous snapshot
        if len(self.snapshots) >= 2:
            prev_snapshot = self.snapshots[-2]
            if prev_snapshot.total_portfolio_value.amount > 0:
                daily_return = (
                    (snapshot.total_portfolio_value.amount - prev_snapshot.total_portfolio_value.amount) /
                    prev_snapshot.total_portfolio_value.amount
                )
                self.daily_returns.append((snapshot.timestamp.date(), daily_return))
    
    def add_realized_trade(self, realized_pnl: RealizedPnL) -> None:
        """Add realized trade."""
        self.realized_trades.append(realized_pnl)
    
    def get_return_series(self, start_date: date = None, end_date: date = None) -> List[Tuple[date, Decimal]]:
        """Get return series for date range."""
        if not start_date and not end_date:
            return self.daily_returns
        
        filtered_returns = []
        for return_date, return_value in self.daily_returns:
            if start_date and return_date < start_date:
                continue
            if end_date and return_date > end_date:
                continue
            filtered_returns.append((return_date, return_value))
        
        return filtered_returns


@dataclass
class PnLAttribution:
    """P&L attribution analysis."""
    attribution_type: AttributionType
    attributions: Dict[str, Money]
    total_attributed: Money
    unattributed: Money
    calculation_time: datetime
    
    def get_top_contributors(self, limit: int = 10) -> List[Tuple[str, Money]]:
        """Get top P&L contributors."""
        sorted_items = sorted(
            self.attributions.items(),
            key=lambda x: x[1].amount,
            reverse=True
        )
        return sorted_items[:limit]
    
    def get_worst_performers(self, limit: int = 10) -> List[Tuple[str, Money]]:
        """Get worst P&L performers."""
        sorted_items = sorted(
            self.attributions.items(),
            key=lambda x: x[1].amount
        )
        return sorted_items[:limit]


class PnLCalculator:
    """P&L calculation engine."""
    
    def __init__(self, position_manager: PositionManager = None):
        self.position_manager = position_manager or get_position_manager()
        self.pnl_history = PnLHistory()
        self.cash_balance = Money(Decimal('0'))
        
        # Configuration
        self.snapshot_interval_minutes = 5
        self.last_snapshot_time = datetime.now()
        
        # Sector and asset class mappings for attribution
        self.sector_mappings: Dict[Symbol, str] = {}
        self.asset_class_mappings: Dict[Symbol, str] = {}
        
        # Cache for performance
        self._last_calculation_time = None
        self._cached_total_pnl = None
        
        self._lock = threading.Lock()
    
    def set_cash_balance(self, balance: Money) -> None:
        """Set cash balance."""
        with self._lock:
            self.cash_balance = balance
            logger.debug(f"Cash balance updated: ${balance.amount:.2f}")
    
    def add_cash(self, amount: Money, reason: str = "Cash deposit") -> None:
        """Add cash to portfolio."""
        with self._lock:
            self.cash_balance = Money(self.cash_balance.amount + amount.amount)
            logger.info(f"Cash added: ${amount.amount:.2f} ({reason})")
    
    def subtract_cash(self, amount: Money, reason: str = "Cash withdrawal") -> None:
        """Subtract cash from portfolio."""
        with self._lock:
            self.cash_balance = Money(self.cash_balance.amount - amount.amount)
            logger.info(f"Cash subtracted: ${amount.amount:.2f} ({reason})")
    
    def add_sector_mapping(self, symbol: Symbol, sector: str) -> None:
        """Add sector mapping for attribution."""
        self.sector_mappings[symbol] = sector
    
    def add_asset_class_mapping(self, symbol: Symbol, asset_class: str) -> None:
        """Add asset class mapping for attribution."""
        self.asset_class_mappings[symbol] = asset_class
    
    def calculate_realized_pnl(self, start_date: date = None, end_date: date = None) -> List[RealizedPnL]:
        """Calculate realized P&L for date range."""
        with self._lock:
            realized_trades = []
            
            # Get realized P&L from position manager's closed positions
            for position in self.position_manager.closed_positions:
                if start_date and position.closed_at.date() < start_date:
                    continue
                if end_date and position.closed_at.date() > end_date:
                    continue
                
                if position.realized_pnl.amount != 0:
                    realized_pnl = RealizedPnL(
                        symbol=position.symbol,
                        realized_amount=position.realized_pnl,
                        realized_date=position.closed_at,
                        holding_period=position.get_holding_period(),
                        quantity_closed=position.quantity,
                        average_open_price=position.average_price,
                        average_close_price=position.current_price,
                        commission_paid=position.total_commission,
                        strategy_id=position.strategy_id
                    )
                    realized_trades.append(realized_pnl)
            
            return realized_trades
    
    def calculate_unrealized_pnl(self) -> List[UnrealizedPnL]:
        """Calculate unrealized P&L for all open positions."""
        with self._lock:
            unrealized_positions = []
            
            for position in self.position_manager.get_all_positions().values():
                if position.quantity.value > 0:
                    unrealized_pnl = UnrealizedPnL(
                        symbol=position.symbol,
                        unrealized_amount=position.unrealized_pnl,
                        current_price=position.current_price,
                        cost_basis=position.total_cost,
                        quantity=position.quantity,
                        position_value=position.market_value,
                        last_updated=position.last_updated,
                        strategy_id=position.strategy_id
                    )
                    unrealized_positions.append(unrealized_pnl)
            
            return unrealized_positions
    
    def calculate_total_pnl(self) -> TotalPnL:
        """Calculate total P&L (realized + unrealized)."""
        with self._lock:
            calculation_time = datetime.now()
            
            # Check cache
            if (self._last_calculation_time and
                (calculation_time - self._last_calculation_time).total_seconds() < 30 and
                self._cached_total_pnl):
                return self._cached_total_pnl
            
            # Calculate realized P&L
            total_realized = Money(Decimal('0'))
            for position in self.position_manager.closed_positions:
                total_realized = Money(total_realized.amount + position.realized_pnl.amount)
            
            # Calculate unrealized P&L
            total_unrealized = self.position_manager.get_total_unrealized_pnl()
            
            # Calculate totals
            total_pnl = Money(total_realized.amount + total_unrealized.amount)
            
            # Calculate cost basis and market value
            total_cost_basis = Money(Decimal('0'))
            total_market_value = Money(Decimal('0'))
            
            for position in self.position_manager.get_all_positions().values():
                total_cost_basis = Money(total_cost_basis.amount + abs(position.total_cost.amount))
                total_market_value = Money(total_market_value.amount + abs(position.market_value.amount))
            
            total_pnl_result = TotalPnL(
                realized_pnl=total_realized,
                unrealized_pnl=total_unrealized,
                total_pnl=total_pnl,
                total_cost_basis=total_cost_basis,
                total_market_value=total_market_value,
                calculation_time=calculation_time
            )
            
            # Cache result
            self._last_calculation_time = calculation_time
            self._cached_total_pnl = total_pnl_result
            
            return total_pnl_result
    
    def calculate_daily_pnl(self, target_date: date = None) -> Money:
        """Calculate P&L for specific day."""
        target_date = target_date or date.today()
        
        # Find snapshots for the day
        day_snapshots = [
            snapshot for snapshot in self.pnl_history.snapshots
            if snapshot.timestamp.date() == target_date
        ]
        
        if not day_snapshots:
            return Money(Decimal('0'))
        
        # Get first and last snapshots of the day
        first_snapshot = min(day_snapshots, key=lambda x: x.timestamp)
        last_snapshot = max(day_snapshots, key=lambda x: x.timestamp)
        
        daily_change = Money(
            last_snapshot.total_portfolio_value.amount - first_snapshot.total_portfolio_value.amount
        )
        
        return daily_change
    
    def take_snapshot(self, timestamp: datetime = None) -> PnLSnapshot:
        """Take P&L snapshot."""
        timestamp = timestamp or datetime.now()
        
        with self._lock:
            total_pnl = self.calculate_total_pnl()
            
            # Calculate daily change
            daily_change = Money(Decimal('0'))
            if self.pnl_history.snapshots:
                last_snapshot = self.pnl_history.snapshots[-1]
                if last_snapshot.timestamp.date() == timestamp.date():
                    # Same day - calculate change from first snapshot of day
                    day_snapshots = [
                        s for s in self.pnl_history.snapshots
                        if s.timestamp.date() == timestamp.date()
                    ]
                    if day_snapshots:
                        first_today = min(day_snapshots, key=lambda x: x.timestamp)
                        daily_change = Money(
                            total_pnl.total_market_value.amount + self.cash_balance.amount -
                            first_today.total_portfolio_value.amount
                        )
                else:
                    # Different day - change from last snapshot
                    current_portfolio_value = total_pnl.total_market_value.amount + self.cash_balance.amount
                    daily_change = Money(current_portfolio_value - last_snapshot.total_portfolio_value.amount)
            
            # Calculate attribution breakdowns
            pnl_by_symbol = {}
            pnl_by_strategy = defaultdict(lambda: Money(Decimal('0')))
            pnl_by_sector = defaultdict(lambda: Money(Decimal('0')))
            
            for position in self.position_manager.get_all_positions().values():
                # Symbol attribution
                pnl_by_symbol[position.symbol] = position.unrealized_pnl
                
                # Strategy attribution
                if position.strategy_id:
                    pnl_by_strategy[position.strategy_id] = Money(
                        pnl_by_strategy[position.strategy_id].amount + position.unrealized_pnl.amount
                    )
                
                # Sector attribution
                sector = self.sector_mappings.get(position.symbol, 'Unknown')
                pnl_by_sector[sector] = Money(
                    pnl_by_sector[sector].amount + position.unrealized_pnl.amount
                )
            
            snapshot = PnLSnapshot(
                timestamp=timestamp,
                realized_pnl=total_pnl.realized_pnl,
                unrealized_pnl=total_pnl.unrealized_pnl,
                total_pnl=total_pnl.total_pnl,
                daily_pnl_change=daily_change,
                positions_count=self.position_manager.get_position_count(),
                market_value=total_pnl.total_market_value,
                cash_balance=self.cash_balance,
                total_portfolio_value=Money(total_pnl.total_market_value.amount + self.cash_balance.amount),
                pnl_by_symbol=pnl_by_symbol,
                pnl_by_strategy=dict(pnl_by_strategy),
                pnl_by_sector=dict(pnl_by_sector)
            )
            
            self.pnl_history.add_snapshot(snapshot)
            self.last_snapshot_time = timestamp
            
            logger.debug(
                f"P&L snapshot taken: Total P&L ${total_pnl.total_pnl.amount:.2f}, "
                f"Daily change ${daily_change.amount:.2f}"
            )
            
            return snapshot
    
    def calculate_attribution(
        self,
        attribution_type: AttributionType,
        start_date: date = None,
        end_date: date = None
    ) -> PnLAttribution:
        """Calculate P&L attribution."""
        calculation_time = datetime.now()
        attributions = defaultdict(lambda: Money(Decimal('0')))
        
        # Get relevant time period
        if not start_date:
            start_date = date.today() - timedelta(days=30)  # Default to last 30 days
        if not end_date:
            end_date = date.today()
        
        if attribution_type == AttributionType.SYMBOL:
            # Symbol attribution from current positions
            for position in self.position_manager.get_all_positions().values():
                attributions[str(position.symbol)] = position.unrealized_pnl
            
            # Add realized P&L from closed positions in period
            for position in self.position_manager.closed_positions:
                if (position.closed_at and
                    start_date <= position.closed_at.date() <= end_date):
                    symbol_key = str(position.symbol)
                    attributions[symbol_key] = Money(
                        attributions[symbol_key].amount + position.realized_pnl.amount
                    )
        
        elif attribution_type == AttributionType.STRATEGY:
            # Strategy attribution
            for position in self.position_manager.get_all_positions().values():
                strategy = position.strategy_id or 'Unknown'
                attributions[strategy] = Money(
                    attributions[strategy].amount + position.unrealized_pnl.amount
                )
            
            # Add realized P&L
            for position in self.position_manager.closed_positions:
                if (position.closed_at and
                    start_date <= position.closed_at.date() <= end_date):
                    strategy = position.strategy_id or 'Unknown'
                    attributions[strategy] = Money(
                        attributions[strategy].amount + position.realized_pnl.amount
                    )
        
        elif attribution_type == AttributionType.SECTOR:
            # Sector attribution
            for position in self.position_manager.get_all_positions().values():
                sector = self.sector_mappings.get(position.symbol, 'Unknown')
                attributions[sector] = Money(
                    attributions[sector].amount + position.unrealized_pnl.amount
                )
            
            # Add realized P&L
            for position in self.position_manager.closed_positions:
                if (position.closed_at and
                    start_date <= position.closed_at.date() <= end_date):
                    sector = self.sector_mappings.get(position.symbol, 'Unknown')
                    attributions[sector] = Money(
                        attributions[sector].amount + position.realized_pnl.amount
                    )
        
        # Calculate totals
        total_attributed = Money(sum(attr.amount for attr in attributions.values()))
        total_pnl = self.calculate_total_pnl()
        unattributed = Money(total_pnl.total_pnl.amount - total_attributed.amount)
        
        return PnLAttribution(
            attribution_type=attribution_type,
            attributions=dict(attributions),
            total_attributed=total_attributed,
            unattributed=unattributed,
            calculation_time=calculation_time
        )
    
    def get_pnl_statistics(self) -> Dict[str, any]:
        """Get P&L statistics."""
        with self._lock:
            total_pnl = self.calculate_total_pnl()
            
            # Calculate win/loss metrics
            profitable_positions = len(self.position_manager.get_profitable_positions())
            losing_positions = len(self.position_manager.get_losing_positions())
            total_positions = profitable_positions + losing_positions
            
            win_rate = profitable_positions / total_positions if total_positions > 0 else 0
            
            # Calculate average trade metrics
            realized_trades = self.calculate_realized_pnl()
            if realized_trades:
                avg_winning_trade = Money(
                    sum(trade.realized_amount.amount for trade in realized_trades 
                        if trade.realized_amount.amount > 0) / 
                    len([t for t in realized_trades if t.realized_amount.amount > 0])
                ) if any(trade.realized_amount.amount > 0 for trade in realized_trades) else Money(Decimal('0'))
                
                avg_losing_trade = Money(
                    sum(trade.realized_amount.amount for trade in realized_trades 
                        if trade.realized_amount.amount < 0) / 
                    len([t for t in realized_trades if t.realized_amount.amount < 0])
                ) if any(trade.realized_amount.amount < 0 for trade in realized_trades) else Money(Decimal('0'))
            else:
                avg_winning_trade = Money(Decimal('0'))
                avg_losing_trade = Money(Decimal('0'))
            
            # Portfolio value breakdown
            portfolio_value = total_pnl.total_market_value.amount + self.cash_balance.amount
            
            return {
                'total_realized_pnl': float(total_pnl.realized_pnl.amount),
                'total_unrealized_pnl': float(total_pnl.unrealized_pnl.amount),
                'total_pnl': float(total_pnl.total_pnl.amount),
                'total_return_percentage': float(total_pnl.get_total_return_percentage()),
                'portfolio_value': float(portfolio_value),
                'cash_balance': float(self.cash_balance.amount),
                'market_value': float(total_pnl.total_market_value.amount),
                'cost_basis': float(total_pnl.total_cost_basis.amount),
                'win_rate': win_rate,
                'profitable_positions': profitable_positions,
                'losing_positions': losing_positions,
                'total_closed_positions': len(realized_trades),
                'avg_winning_trade': float(avg_winning_trade.amount),
                'avg_losing_trade': float(avg_losing_trade.amount),
                'snapshots_taken': len(self.pnl_history.snapshots),
                'calculation_time': datetime.now().isoformat()
            }
    
    def get_pnl_history(self) -> PnLHistory:
        """Get P&L history."""
        return self.pnl_history
    
    def cleanup_old_snapshots(self, retention_days: int = 90) -> int:
        """Clean up old P&L snapshots."""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        with self._lock:
            old_count = len(self.pnl_history.snapshots)
            self.pnl_history.snapshots = [
                snapshot for snapshot in self.pnl_history.snapshots
                if snapshot.timestamp >= cutoff_time
            ]
            
            # Also clean up daily returns
            cutoff_date = cutoff_time.date()
            self.pnl_history.daily_returns = [
                (return_date, return_value) for return_date, return_value in self.pnl_history.daily_returns
                if return_date >= cutoff_date
            ]
            
            cleaned_count = old_count - len(self.pnl_history.snapshots)
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old P&L snapshots")
            
            return cleaned_count


# Global P&L calculator instance
_pnl_calculator = None


def get_pnl_calculator() -> PnLCalculator:
    """Get global P&L calculator."""
    global _pnl_calculator
    if _pnl_calculator is None:
        _pnl_calculator = PnLCalculator()
    return _pnl_calculator