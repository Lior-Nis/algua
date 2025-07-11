"""
Portfolio management and coordination system.
"""

from typing import Dict, List, Optional, Tuple, Set
from decimal import Decimal
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from enum import Enum
import threading
import json
from pathlib import Path

from domain.value_objects import Symbol, Price, Quantity, Money
from .position_manager import PositionManager, Position, get_position_manager
from .pnl_calculator import PnLCalculator, PnLSnapshot, get_pnl_calculator
from order_management import Fill, Order
from risk_management import get_portfolio_limiter, get_drawdown_controller
from utils.logging import get_logger

logger = get_logger(__name__)


class PortfolioStatus(Enum):
    """Portfolio status."""
    ACTIVE = "active"
    PAUSED = "paused"
    LIQUIDATING = "liquidating"
    CLOSED = "closed"


@dataclass
class PortfolioConfiguration:
    """Portfolio configuration settings."""
    name: str
    initial_capital: Money
    currency: str = "USD"
    benchmark_symbol: Optional[Symbol] = None
    
    # Risk settings
    max_positions: int = 20
    max_portfolio_leverage: Decimal = Decimal('1.0')
    max_position_size_pct: Decimal = Decimal('0.10')  # 10%
    
    # P&L settings
    take_profit_threshold: Optional[Decimal] = None
    stop_loss_threshold: Optional[Decimal] = None
    
    # Rebalancing
    rebalance_frequency_days: int = 30
    rebalance_threshold_pct: Decimal = Decimal('0.05')  # 5%
    
    # Reporting
    performance_benchmark: Optional[str] = None
    snapshot_frequency_minutes: int = 5
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'initial_capital': float(self.initial_capital.amount),
            'currency': self.currency,
            'benchmark_symbol': str(self.benchmark_symbol) if self.benchmark_symbol else None,
            'max_positions': self.max_positions,
            'max_portfolio_leverage': float(self.max_portfolio_leverage),
            'max_position_size_pct': float(self.max_position_size_pct),
            'take_profit_threshold': float(self.take_profit_threshold) if self.take_profit_threshold else None,
            'stop_loss_threshold': float(self.stop_loss_threshold) if self.stop_loss_threshold else None,
            'rebalance_frequency_days': self.rebalance_frequency_days,
            'rebalance_threshold_pct': float(self.rebalance_threshold_pct),
            'performance_benchmark': self.performance_benchmark,
            'snapshot_frequency_minutes': self.snapshot_frequency_minutes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> 'PortfolioConfiguration':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            initial_capital=Money(Decimal(str(data['initial_capital']))),
            currency=data.get('currency', 'USD'),
            benchmark_symbol=Symbol(data['benchmark_symbol']) if data.get('benchmark_symbol') else None,
            max_positions=data.get('max_positions', 20),
            max_portfolio_leverage=Decimal(str(data.get('max_portfolio_leverage', '1.0'))),
            max_position_size_pct=Decimal(str(data.get('max_position_size_pct', '0.10'))),
            take_profit_threshold=Decimal(str(data['take_profit_threshold'])) if data.get('take_profit_threshold') else None,
            stop_loss_threshold=Decimal(str(data['stop_loss_threshold'])) if data.get('stop_loss_threshold') else None,
            rebalance_frequency_days=data.get('rebalance_frequency_days', 30),
            rebalance_threshold_pct=Decimal(str(data.get('rebalance_threshold_pct', '0.05'))),
            performance_benchmark=data.get('performance_benchmark'),
            snapshot_frequency_minutes=data.get('snapshot_frequency_minutes', 5)
        )


@dataclass
class Portfolio:
    """Portfolio data structure."""
    name: str
    configuration: PortfolioConfiguration
    status: PortfolioStatus
    created_at: datetime
    
    # Current state
    cash_balance: Money
    total_value: Money
    market_value: Money
    positions_count: int
    
    # Performance metrics
    total_return: Decimal
    daily_return: Decimal
    unrealized_pnl: Money
    realized_pnl: Money
    
    # Risk metrics
    current_leverage: Decimal
    max_drawdown: Decimal
    volatility: Decimal
    
    # Timestamps
    last_updated: datetime
    last_rebalance: Optional[datetime] = None
    
    def get_portfolio_value(self) -> Money:
        """Get total portfolio value."""
        return Money(self.cash_balance.amount + self.market_value.amount)
    
    def get_cash_percentage(self) -> Decimal:
        """Get cash as percentage of portfolio."""
        total_value = self.get_portfolio_value()
        if total_value.amount == 0:
            return Decimal('1.0')
        return self.cash_balance.amount / total_value.amount


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot for historical tracking."""
    timestamp: datetime
    portfolio: Portfolio
    positions: Dict[Symbol, Position]
    pnl_snapshot: PnLSnapshot
    
    # Performance metrics
    sharpe_ratio: Optional[Decimal] = None
    sortino_ratio: Optional[Decimal] = None
    max_drawdown_pct: Optional[Decimal] = None
    
    # Risk metrics
    var_1d: Optional[Money] = None
    portfolio_beta: Optional[Decimal] = None
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': float(self.portfolio.get_portfolio_value().amount),
            'cash_balance': float(self.portfolio.cash_balance.amount),
            'market_value': float(self.portfolio.market_value.amount),
            'positions_count': self.portfolio.positions_count,
            'total_return': float(self.portfolio.total_return),
            'daily_return': float(self.portfolio.daily_return),
            'unrealized_pnl': float(self.portfolio.unrealized_pnl.amount),
            'realized_pnl': float(self.portfolio.realized_pnl.amount),
            'current_leverage': float(self.portfolio.current_leverage),
            'max_drawdown': float(self.portfolio.max_drawdown),
            'sharpe_ratio': float(self.sharpe_ratio) if self.sharpe_ratio else None,
            'sortino_ratio': float(self.sortino_ratio) if self.sortino_ratio else None,
            'portfolio_beta': float(self.portfolio_beta) if self.portfolio_beta else None
        }


@dataclass
class PortfolioHistory:
    """Portfolio historical data."""
    snapshots: List[PortfolioSnapshot] = field(default_factory=list)
    rebalance_history: List[Dict[str, any]] = field(default_factory=list)
    
    def add_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Add portfolio snapshot."""
        self.snapshots.append(snapshot)
    
    def add_rebalance_event(self, event: Dict[str, any]) -> None:
        """Add rebalance event."""
        self.rebalance_history.append(event)
    
    def get_value_series(self) -> List[Tuple[datetime, Decimal]]:
        """Get portfolio value time series."""
        return [(snap.timestamp, snap.portfolio.get_portfolio_value().amount) for snap in self.snapshots]
    
    def get_return_series(self) -> List[Tuple[datetime, Decimal]]:
        """Get return time series."""
        return [(snap.timestamp, snap.portfolio.daily_return) for snap in self.snapshots]


class PortfolioManager:
    """Main portfolio management system."""
    
    def __init__(
        self,
        configuration: PortfolioConfiguration,
        position_manager: PositionManager = None,
        pnl_calculator: PnLCalculator = None
    ):
        self.configuration = configuration
        self.position_manager = position_manager or get_position_manager()
        self.pnl_calculator = pnl_calculator or get_pnl_calculator()
        
        # Portfolio state
        self.portfolio = Portfolio(
            name=configuration.name,
            configuration=configuration,
            status=PortfolioStatus.ACTIVE,
            created_at=datetime.now(),
            cash_balance=configuration.initial_capital,
            total_value=configuration.initial_capital,
            market_value=Money(Decimal('0')),
            positions_count=0,
            total_return=Decimal('0'),
            daily_return=Decimal('0'),
            unrealized_pnl=Money(Decimal('0')),
            realized_pnl=Money(Decimal('0')),
            current_leverage=Decimal('0'),
            max_drawdown=Decimal('0'),
            volatility=Decimal('0'),
            last_updated=datetime.now()
        )
        
        # History tracking
        self.portfolio_history = PortfolioHistory()
        
        # Risk management integration
        self.portfolio_limiter = get_portfolio_limiter()
        self.drawdown_controller = get_drawdown_controller()
        
        # Auto-snapshot configuration
        self.auto_snapshot_enabled = True
        self.last_snapshot_time = datetime.now()
        
        # Initialize P&L calculator with initial cash
        self.pnl_calculator.set_cash_balance(configuration.initial_capital)
        
        self._lock = threading.Lock()
        
        logger.info(f"Portfolio '{configuration.name}' initialized with ${configuration.initial_capital.amount:.2f}")
    
    def process_fill(self, fill: Fill, order: Order) -> None:
        """Process a trade fill and update portfolio."""
        with self._lock:
            # Update position manager
            position = self.position_manager.update_position(order.symbol, fill, order.order_id)
            
            # Update cash balance (simplified - assumes cash settlement)
            if order.side.value == 'buy':
                cash_change = -(fill.quantity.value * fill.price.value + fill.commission.amount)
            else:
                cash_change = fill.quantity.value * fill.price.value - fill.commission.amount
            
            self.portfolio.cash_balance = Money(self.portfolio.cash_balance.amount + cash_change)
            self.pnl_calculator.add_cash(Money(cash_change), f"Trade: {order.symbol}")
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            # Check for auto-snapshot
            if self._should_take_snapshot():
                self.take_snapshot()
            
            logger.info(
                f"Processed fill: {fill.quantity.value} {order.symbol} @ {fill.price.value:.2f}, "
                f"Cash: ${self.portfolio.cash_balance.amount:.2f}"
            )
    
    def update_market_prices(self, prices: Dict[Symbol, Price], timestamp: datetime = None) -> None:
        """Update market prices and recalculate portfolio value."""
        timestamp = timestamp or datetime.now()
        
        with self._lock:
            # Update position manager
            self.position_manager.update_market_prices(prices, timestamp)
            
            # Update portfolio metrics
            self._update_portfolio_metrics(timestamp)
            
            # Update drawdown controller
            current_portfolio_value = self.portfolio.get_portfolio_value()
            self.drawdown_controller.update_portfolio_value(current_portfolio_value, timestamp)
            
            # Check for auto-snapshot
            if self._should_take_snapshot():
                self.take_snapshot(timestamp)
            
            logger.debug(f"Updated portfolio prices: {len(prices)} symbols")
    
    def _update_portfolio_metrics(self, timestamp: datetime = None) -> None:
        """Update portfolio metrics from current positions."""
        timestamp = timestamp or datetime.now()
        
        # Get current P&L
        total_pnl = self.pnl_calculator.calculate_total_pnl()
        
        # Update portfolio values
        self.portfolio.market_value = total_pnl.total_market_value
        self.portfolio.total_value = Money(
            self.portfolio.cash_balance.amount + self.portfolio.market_value.amount
        )
        self.portfolio.positions_count = self.position_manager.get_position_count()
        self.portfolio.unrealized_pnl = total_pnl.unrealized_pnl
        self.portfolio.realized_pnl = total_pnl.realized_pnl
        
        # Calculate returns
        if self.configuration.initial_capital.amount > 0:
            self.portfolio.total_return = (
                (self.portfolio.total_value.amount - self.configuration.initial_capital.amount) /
                self.configuration.initial_capital.amount
            )
        
        # Calculate daily return
        if self.portfolio_history.snapshots:
            last_snapshot = self.portfolio_history.snapshots[-1]
            if last_snapshot.timestamp.date() == timestamp.date():
                # Same day - calculate from first snapshot of day
                day_snapshots = [
                    s for s in self.portfolio_history.snapshots
                    if s.timestamp.date() == timestamp.date()
                ]
                if day_snapshots:
                    first_today = min(day_snapshots, key=lambda x: x.timestamp)
                    if first_today.portfolio.total_value.amount > 0:
                        self.portfolio.daily_return = (
                            (self.portfolio.total_value.amount - first_today.portfolio.total_value.amount) /
                            first_today.portfolio.total_value.amount
                        )
            else:
                # Different day
                if last_snapshot.portfolio.total_value.amount > 0:
                    self.portfolio.daily_return = (
                        (self.portfolio.total_value.amount - last_snapshot.portfolio.total_value.amount) /
                        last_snapshot.portfolio.total_value.amount
                    )
        
        # Calculate leverage
        if self.portfolio.total_value.amount > 0:
            self.portfolio.current_leverage = (
                self.portfolio.market_value.amount / self.portfolio.total_value.amount
            )
        
        # Update max drawdown from drawdown controller
        drawdown_metrics = self.drawdown_controller.update_portfolio_value(self.portfolio.total_value)
        self.portfolio.max_drawdown = drawdown_metrics.max_drawdown_pct
        
        self.portfolio.last_updated = timestamp
    
    def _should_take_snapshot(self) -> bool:
        """Check if we should take an automatic snapshot."""
        if not self.auto_snapshot_enabled:
            return False
        
        time_since_last = (datetime.now() - self.last_snapshot_time).total_seconds()
        return time_since_last >= (self.configuration.snapshot_frequency_minutes * 60)
    
    def take_snapshot(self, timestamp: datetime = None) -> PortfolioSnapshot:
        """Take portfolio snapshot."""
        timestamp = timestamp or datetime.now()
        
        with self._lock:
            # Take P&L snapshot
            pnl_snapshot = self.pnl_calculator.take_snapshot(timestamp)
            
            # Get current positions
            current_positions = self.position_manager.get_all_positions()
            
            # Calculate additional metrics
            sharpe_ratio = self._calculate_sharpe_ratio()
            sortino_ratio = self._calculate_sortino_ratio()
            
            # Create portfolio snapshot
            snapshot = PortfolioSnapshot(
                timestamp=timestamp,
                portfolio=Portfolio(
                    name=self.portfolio.name,
                    configuration=self.portfolio.configuration,
                    status=self.portfolio.status,
                    created_at=self.portfolio.created_at,
                    cash_balance=self.portfolio.cash_balance,
                    total_value=self.portfolio.total_value,
                    market_value=self.portfolio.market_value,
                    positions_count=self.portfolio.positions_count,
                    total_return=self.portfolio.total_return,
                    daily_return=self.portfolio.daily_return,
                    unrealized_pnl=self.portfolio.unrealized_pnl,
                    realized_pnl=self.portfolio.realized_pnl,
                    current_leverage=self.portfolio.current_leverage,
                    max_drawdown=self.portfolio.max_drawdown,
                    volatility=self.portfolio.volatility,
                    last_updated=self.portfolio.last_updated,
                    last_rebalance=self.portfolio.last_rebalance
                ),
                positions=current_positions.copy(),
                pnl_snapshot=pnl_snapshot,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio
            )
            
            self.portfolio_history.add_snapshot(snapshot)
            self.last_snapshot_time = timestamp
            
            logger.debug(f"Portfolio snapshot taken: Value ${self.portfolio.total_value.amount:.2f}")
            
            return snapshot
    
    def _calculate_sharpe_ratio(self, risk_free_rate: Decimal = Decimal('0.02')) -> Optional[Decimal]:
        """Calculate Sharpe ratio."""
        if len(self.portfolio_history.snapshots) < 30:  # Need at least 30 days
            return None
        
        # Get daily returns
        returns = [snap.portfolio.daily_return for snap in self.portfolio_history.snapshots[-30:]]
        
        if not returns:
            return None
        
        # Calculate excess returns
        daily_risk_free = risk_free_rate / Decimal('252')  # Annualized to daily
        excess_returns = [r - daily_risk_free for r in returns]
        
        # Calculate Sharpe ratio
        if len(excess_returns) > 1:
            mean_excess = sum(excess_returns) / len(excess_returns)
            variance = sum((r - mean_excess) ** 2 for r in excess_returns) / (len(excess_returns) - 1)
            
            if variance > 0:
                std_dev = variance ** Decimal('0.5')
                sharpe = mean_excess / std_dev * Decimal('252') ** Decimal('0.5')  # Annualized
                return sharpe
        
        return None
    
    def _calculate_sortino_ratio(self, risk_free_rate: Decimal = Decimal('0.02')) -> Optional[Decimal]:
        """Calculate Sortino ratio (downside deviation)."""
        if len(self.portfolio_history.snapshots) < 30:
            return None
        
        returns = [snap.portfolio.daily_return for snap in self.portfolio_history.snapshots[-30:]]
        
        if not returns:
            return None
        
        daily_risk_free = risk_free_rate / Decimal('252')
        excess_returns = [r - daily_risk_free for r in returns]
        
        # Calculate downside deviation
        negative_returns = [r for r in excess_returns if r < 0]
        
        if len(negative_returns) > 1:
            mean_excess = sum(excess_returns) / len(excess_returns)
            downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
            
            if downside_variance > 0:
                downside_deviation = downside_variance ** Decimal('0.5')
                sortino = mean_excess / downside_deviation * Decimal('252') ** Decimal('0.5')
                return sortino
        
        return None
    
    def check_rebalance_needed(self) -> bool:
        """Check if portfolio needs rebalancing."""
        if not self.portfolio.last_rebalance:
            # Never rebalanced
            days_since_creation = (datetime.now() - self.portfolio.created_at).days
            return days_since_creation >= self.configuration.rebalance_frequency_days
        
        days_since_rebalance = (datetime.now() - self.portfolio.last_rebalance).days
        return days_since_rebalance >= self.configuration.rebalance_frequency_days
    
    def suggest_rebalancing(self) -> List[Dict[str, any]]:
        """Suggest rebalancing actions."""
        suggestions = []
        
        positions = self.position_manager.get_all_positions()
        portfolio_value = self.portfolio.get_portfolio_value()
        
        for symbol, position in positions.items():
            position_pct = position.get_position_size_pct(portfolio_value)
            
            # Check if position exceeds size limits
            if position_pct > self.configuration.max_position_size_pct:
                target_pct = self.configuration.max_position_size_pct * Decimal('0.9')  # 10% buffer
                reduction_needed = position_pct - target_pct
                
                suggestions.append({
                    'action': 'reduce',
                    'symbol': str(symbol),
                    'current_size_pct': float(position_pct),
                    'target_size_pct': float(target_pct),
                    'reduction_pct': float(reduction_needed),
                    'reason': 'Position size exceeds limit'
                })
        
        return suggestions
    
    def add_cash(self, amount: Money, reason: str = "Cash deposit") -> None:
        """Add cash to portfolio."""
        with self._lock:
            self.portfolio.cash_balance = Money(self.portfolio.cash_balance.amount + amount.amount)
            self.pnl_calculator.add_cash(amount, reason)
            self._update_portfolio_metrics()
            
            logger.info(f"Added ${amount.amount:.2f} to portfolio: {reason}")
    
    def withdraw_cash(self, amount: Money, reason: str = "Cash withdrawal") -> bool:
        """Withdraw cash from portfolio."""
        with self._lock:
            if amount.amount > self.portfolio.cash_balance.amount:
                logger.warning(f"Insufficient cash for withdrawal: ${amount.amount:.2f} requested, ${self.portfolio.cash_balance.amount:.2f} available")
                return False
            
            self.portfolio.cash_balance = Money(self.portfolio.cash_balance.amount - amount.amount)
            self.pnl_calculator.subtract_cash(amount, reason)
            self._update_portfolio_metrics()
            
            logger.info(f"Withdrew ${amount.amount:.2f} from portfolio: {reason}")
            return True
    
    def get_portfolio_summary(self) -> Dict[str, any]:
        """Get portfolio summary."""
        with self._lock:
            positions = self.position_manager.get_all_positions()
            pnl_stats = self.pnl_calculator.get_pnl_statistics()
            
            return {
                'name': self.portfolio.name,
                'status': self.portfolio.status.value,
                'created_at': self.portfolio.created_at.isoformat(),
                'last_updated': self.portfolio.last_updated.isoformat(),
                'portfolio_value': float(self.portfolio.get_portfolio_value().amount),
                'cash_balance': float(self.portfolio.cash_balance.amount),
                'cash_percentage': float(self.portfolio.get_cash_percentage()),
                'market_value': float(self.portfolio.market_value.amount),
                'positions_count': self.portfolio.positions_count,
                'total_return': float(self.portfolio.total_return),
                'daily_return': float(self.portfolio.daily_return),
                'unrealized_pnl': float(self.portfolio.unrealized_pnl.amount),
                'realized_pnl': float(self.portfolio.realized_pnl.amount),
                'current_leverage': float(self.portfolio.current_leverage),
                'max_drawdown': float(self.portfolio.max_drawdown),
                'win_rate': pnl_stats.get('win_rate', 0),
                'profitable_positions': len(self.position_manager.get_profitable_positions()),
                'losing_positions': len(self.position_manager.get_losing_positions()),
                'largest_position': self._get_largest_position_info(),
                'top_performers': self._get_top_performers(5),
                'worst_performers': self._get_worst_performers(5)
            }
    
    def _get_largest_position_info(self) -> Optional[Dict[str, any]]:
        """Get largest position information."""
        positions = self.position_manager.get_all_positions()
        if not positions:
            return None
        
        largest_position = max(positions.values(), key=lambda p: abs(p.market_value.amount))
        portfolio_value = self.portfolio.get_portfolio_value()
        
        return {
            'symbol': str(largest_position.symbol),
            'market_value': float(largest_position.market_value.amount),
            'percentage': float(largest_position.get_position_size_pct(portfolio_value)),
            'unrealized_pnl': float(largest_position.unrealized_pnl.amount)
        }
    
    def _get_top_performers(self, limit: int) -> List[Dict[str, any]]:
        """Get top performing positions."""
        positions = self.position_manager.get_profitable_positions()
        sorted_positions = sorted(positions, key=lambda p: p.unrealized_pnl.amount, reverse=True)
        
        return [
            {
                'symbol': str(pos.symbol),
                'unrealized_pnl': float(pos.unrealized_pnl.amount),
                'return_pct': float(pos.get_pnl_percentage())
            }
            for pos in sorted_positions[:limit]
        ]
    
    def _get_worst_performers(self, limit: int) -> List[Dict[str, any]]:
        """Get worst performing positions."""
        positions = self.position_manager.get_losing_positions()
        sorted_positions = sorted(positions, key=lambda p: p.unrealized_pnl.amount)
        
        return [
            {
                'symbol': str(pos.symbol),
                'unrealized_pnl': float(pos.unrealized_pnl.amount),
                'return_pct': float(pos.get_pnl_percentage())
            }
            for pos in sorted_positions[:limit]
        ]
    
    def save_configuration(self, file_path: Path) -> None:
        """Save portfolio configuration."""
        config_data = self.configuration.to_dict()
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Portfolio configuration saved to {file_path}")
    
    @classmethod
    def load_configuration(cls, file_path: Path) -> PortfolioConfiguration:
        """Load portfolio configuration."""
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
        return PortfolioConfiguration.from_dict(config_data)
    
    def get_portfolio_history(self) -> PortfolioHistory:
        """Get portfolio history."""
        return self.portfolio_history


# Global portfolio manager instance
_portfolio_manager = None


def get_portfolio_manager() -> Optional[PortfolioManager]:
    """Get global portfolio manager."""
    global _portfolio_manager
    return _portfolio_manager


def initialize_portfolio_manager(configuration: PortfolioConfiguration) -> PortfolioManager:
    """Initialize global portfolio manager."""
    global _portfolio_manager
    _portfolio_manager = PortfolioManager(configuration)
    return _portfolio_manager