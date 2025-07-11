"""
Risk management interfaces and abstractions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from domain.value_objects import Money, Price, Quantity, Symbol


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class RiskEventType(Enum):
    """Risk event types."""
    POSITION_SIZE_EXCEEDED = "position_size_exceeded"
    DAILY_LOSS_EXCEEDED = "daily_loss_exceeded"
    DRAWDOWN_LIMIT_REACHED = "drawdown_limit_reached"
    CONCENTRATION_RISK = "concentration_risk"
    CORRELATION_RISK = "correlation_risk"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TIME_STOP_TRIGGERED = "time_stop_triggered"
    VOLATILITY_SPIKE = "volatility_spike"


@dataclass
class RiskEvent:
    """Risk event data structure."""
    event_type: RiskEventType
    timestamp: datetime
    symbol: Symbol
    risk_level: RiskLevel
    message: str
    data: Dict[str, Any]
    action_taken: Optional[str] = None


@dataclass
class RiskMetrics:
    """Risk metrics data structure."""
    portfolio_value: Money
    total_exposure: Money
    max_position_size: Decimal
    current_drawdown: Decimal
    max_drawdown: Decimal
    var_1d: Money  # Value at Risk 1 day
    var_5d: Money  # Value at Risk 5 days
    risk_score: Decimal
    last_updated: datetime


@dataclass
class PositionRisk:
    """Position-specific risk metrics."""
    symbol: Symbol
    position_size: Quantity
    market_value: Money
    unrealized_pnl: Money
    risk_percentage: Decimal
    stop_loss_price: Optional[Price]
    risk_per_share: Optional[Money]
    time_in_position: int  # days
    volatility: Decimal


class RiskCalculatorProtocol(Protocol):
    """Protocol for risk calculators."""
    
    def calculate_position_risk(
        self,
        symbol: Symbol,
        position_size: Quantity,
        entry_price: Price,
        current_price: Price,
        stop_loss_price: Optional[Price] = None
    ) -> PositionRisk:
        """Calculate risk for a single position."""
        ...
    
    def calculate_portfolio_risk(
        self,
        positions: List[PositionRisk],
        portfolio_value: Money
    ) -> RiskMetrics:
        """Calculate overall portfolio risk."""
        ...


class PositionSizerProtocol(Protocol):
    """Protocol for position sizers."""
    
    def calculate_position_size(
        self,
        portfolio_value: Money,
        entry_price: Price,
        stop_loss_price: Price,
        risk_per_trade: Decimal
    ) -> Quantity:
        """Calculate optimal position size."""
        ...


class StopLossProtocol(Protocol):
    """Protocol for stop loss mechanisms."""
    
    def calculate_stop_loss(
        self,
        symbol: Symbol,
        entry_price: Price,
        position_size: Quantity,
        market_data: Dict[str, Any]
    ) -> Price:
        """Calculate stop loss price."""
        ...
    
    def should_trigger_stop(
        self,
        current_price: Price,
        stop_loss_price: Price,
        position_type: str  # 'long' or 'short'
    ) -> bool:
        """Check if stop loss should trigger."""
        ...


class RiskMonitorProtocol(Protocol):
    """Protocol for risk monitors."""
    
    def monitor_risk(
        self,
        positions: List[PositionRisk],
        portfolio_value: Money
    ) -> List[RiskEvent]:
        """Monitor risk and generate events."""
        ...


class RiskManagerInterface(ABC):
    """Abstract base class for risk managers."""
    
    @abstractmethod
    def assess_trade_risk(
        self,
        symbol: Symbol,
        position_size: Quantity,
        entry_price: Price,
        stop_loss_price: Optional[Price],
        portfolio_value: Money
    ) -> bool:
        """Assess if a trade meets risk criteria."""
        pass
    
    @abstractmethod
    def calculate_optimal_position_size(
        self,
        symbol: Symbol,
        entry_price: Price,
        stop_loss_price: Price,
        portfolio_value: Money
    ) -> Quantity:
        """Calculate optimal position size."""
        pass
    
    @abstractmethod
    def monitor_portfolio_risk(
        self,
        positions: List[PositionRisk],
        portfolio_value: Money
    ) -> List[RiskEvent]:
        """Monitor portfolio risk and generate events."""
        pass
    
    @abstractmethod
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        pass


class RiskEventHandlerProtocol(Protocol):
    """Protocol for risk event handlers."""
    
    def handle_risk_event(self, event: RiskEvent) -> None:
        """Handle a risk event."""
        ...


class RiskConfigurationProtocol(Protocol):
    """Protocol for risk configuration."""
    
    @property
    def max_position_size(self) -> Decimal:
        """Maximum position size as percentage of portfolio."""
        ...
    
    @property
    def max_daily_loss(self) -> Decimal:
        """Maximum daily loss as percentage of portfolio."""
        ...
    
    @property
    def max_drawdown(self) -> Decimal:
        """Maximum drawdown as percentage of portfolio."""
        ...
    
    @property
    def risk_free_rate(self) -> Decimal:
        """Risk-free rate for calculations."""
        ...
    
    @property
    def var_confidence_level(self) -> Decimal:
        """Confidence level for VaR calculations."""
        ...