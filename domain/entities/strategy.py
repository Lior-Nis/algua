"""
Strategy entity - Represents a trading strategy.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any

from ..value_objects import Money


class StrategyStatus(Enum):
    INACTIVE = "inactive"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StrategyPerformance:
    """Strategy performance metrics."""
    total_return: Decimal = Decimal('0')
    sharpe_ratio: Decimal = Decimal('0')
    max_drawdown: Decimal = Decimal('0')
    win_rate: Decimal = Decimal('0')
    profit_factor: Decimal = Decimal('0')
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: Decimal = Decimal('0')
    avg_loss: Decimal = Decimal('0')
    largest_win: Decimal = Decimal('0')
    largest_loss: Decimal = Decimal('0')
    
    @property
    def expectancy(self) -> Decimal:
        """Calculate expectancy per trade."""
        if self.total_trades == 0:
            return Decimal('0')
        
        win_prob = self.win_rate / 100
        loss_prob = Decimal('1') - win_prob
        
        return (win_prob * self.avg_win) - (loss_prob * abs(self.avg_loss))


@dataclass
class Strategy:
    """Trading strategy entity."""
    
    id: str
    name: str
    description: str
    strategy_type: str  # e.g., "mean_reversion", "momentum", "pairs_trading"
    parameters: Dict[str, Any] = field(default_factory=dict)
    allocated_capital: Money = field(default_factory=lambda: Money(Decimal('0')))
    status: StrategyStatus = StrategyStatus.INACTIVE
    performance: StrategyPerformance = field(default_factory=StrategyPerformance)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    last_signal_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate strategy data."""
        if not self.name.strip():
            raise ValueError("Strategy name cannot be empty")
        
        if self.allocated_capital.amount < 0:
            raise ValueError("Allocated capital cannot be negative")
    
    @property
    def is_active(self) -> bool:
        """Check if strategy is actively running."""
        return self.status == StrategyStatus.RUNNING
    
    @property
    def runtime_hours(self) -> Optional[Decimal]:
        """Calculate strategy runtime in hours."""
        if self.started_at is None:
            return None
        
        end_time = self.stopped_at or datetime.now()
        runtime = end_time - self.started_at
        return Decimal(str(runtime.total_seconds() / 3600))
    
    @property
    def roi_percentage(self) -> Decimal:
        """Calculate return on investment percentage."""
        if self.allocated_capital.amount == 0:
            return Decimal('0')
        
        return self.performance.total_return
    
    def start(self, start_time: datetime = None) -> None:
        """Start the strategy."""
        if start_time is None:
            start_time = datetime.now()
        
        if self.status == StrategyStatus.RUNNING:
            raise ValueError("Strategy is already running")
        
        self.status = StrategyStatus.RUNNING
        self.started_at = start_time
        self.stopped_at = None
    
    def stop(self, stop_time: datetime = None) -> None:
        """Stop the strategy."""
        if stop_time is None:
            stop_time = datetime.now()
        
        if self.status not in [StrategyStatus.RUNNING, StrategyStatus.PAUSED]:
            raise ValueError("Strategy is not running or paused")
        
        self.status = StrategyStatus.STOPPED
        self.stopped_at = stop_time
    
    def pause(self, pause_time: datetime = None) -> None:
        """Pause the strategy."""
        if pause_time is None:
            pause_time = datetime.now()
        
        if self.status != StrategyStatus.RUNNING:
            raise ValueError("Strategy is not running")
        
        self.status = StrategyStatus.PAUSED
    
    def resume(self, resume_time: datetime = None) -> None:
        """Resume the strategy."""
        if resume_time is None:
            resume_time = datetime.now()
        
        if self.status != StrategyStatus.PAUSED:
            raise ValueError("Strategy is not paused")
        
        self.status = StrategyStatus.RUNNING
    
    def set_error(self, error_time: datetime = None) -> None:
        """Set strategy to error state."""
        if error_time is None:
            error_time = datetime.now()
        
        self.status = StrategyStatus.ERROR
        self.stopped_at = error_time
    
    def update_performance(self, performance: StrategyPerformance) -> None:
        """Update strategy performance metrics."""
        self.performance = performance
    
    def record_signal(self, signal_time: datetime = None) -> None:
        """Record when a trading signal was generated."""
        if signal_time is None:
            signal_time = datetime.now()
        
        self.last_signal_at = signal_time
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get a strategy parameter."""
        return self.parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """Set a strategy parameter."""
        self.parameters[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'strategy_type': self.strategy_type,
            'parameters': self.parameters,
            'allocated_capital': {
                'amount': float(self.allocated_capital.amount),
                'currency': self.allocated_capital.currency
            },
            'status': self.status.value,
            'performance': {
                'total_return': float(self.performance.total_return),
                'sharpe_ratio': float(self.performance.sharpe_ratio),
                'max_drawdown': float(self.performance.max_drawdown),
                'win_rate': float(self.performance.win_rate),
                'profit_factor': float(self.performance.profit_factor),
                'total_trades': self.performance.total_trades,
                'winning_trades': self.performance.winning_trades,
                'losing_trades': self.performance.losing_trades,
                'expectancy': float(self.performance.expectancy)
            },
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'stopped_at': self.stopped_at.isoformat() if self.stopped_at else None,
            'last_signal_at': self.last_signal_at.isoformat() if self.last_signal_at else None,
            'runtime_hours': float(self.runtime_hours) if self.runtime_hours else None,
            'roi_percentage': float(self.roi_percentage)
        }