"""
Position sizing calculations.
"""

from decimal import Decimal
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from domain.value_objects import Price, Quantity, Money, Symbol
from configs.settings import get_settings
from .interfaces import PositionSizerProtocol
from .configuration import get_risk_config
from utils.logging import get_logger

logger = get_logger(__name__)


class PositionSizingMethod(Enum):
    """Position sizing methods."""
    FIXED_RISK = "fixed_risk"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_TARGET = "volatility_target"
    FIXED_FRACTION = "fixed_fraction"
    ATR_BASED = "atr_based"


@dataclass
class PositionSizeParams:
    """Parameters for position sizing."""
    portfolio_value: Money
    risk_per_trade: Decimal  # As percentage (e.g., 0.02 for 2%)
    entry_price: Price
    stop_loss_price: Optional[Price] = None
    max_position_size: Optional[Decimal] = None  # As percentage of portfolio


class BasePositionSizer(ABC):
    """Base class for position sizers."""
    
    @abstractmethod
    def calculate_position_size(
        self,
        portfolio_value: Money,
        entry_price: Price,
        stop_loss_price: Price,
        risk_per_trade: Decimal
    ) -> Quantity:
        """Calculate position size."""
        pass


class FixedRiskPositionSizer(BasePositionSizer):
    """Fixed risk position sizer."""
    
    def __init__(self, config=None):
        self.config = config or get_risk_config()
    
    def calculate_position_size(
        self,
        portfolio_value: Money,
        entry_price: Price,
        stop_loss_price: Price,
        risk_per_trade: Decimal
    ) -> Quantity:
        """Calculate position size based on fixed risk per trade."""
        risk_per_share = abs(entry_price.value - stop_loss_price.value)
        if risk_per_share == 0:
            raise ValueError("Entry price and stop loss price cannot be the same")
        
        total_risk_amount = portfolio_value.amount * risk_per_trade
        position_size = total_risk_amount / risk_per_share
        
        # Apply position size limits
        max_position_value = portfolio_value.amount * self.config.max_position_size
        max_position_size = max_position_value / entry_price.value
        
        return Quantity(min(position_size, max_position_size))


class KellyPositionSizer(BasePositionSizer):
    """Kelly Criterion position sizer."""
    
    def __init__(self, win_rate: Decimal, avg_win: Decimal, avg_loss: Decimal, config=None):
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.config = config or get_risk_config()
    
    def calculate_position_size(
        self,
        portfolio_value: Money,
        entry_price: Price,
        stop_loss_price: Price,
        risk_per_trade: Decimal
    ) -> Quantity:
        """Calculate position size using Kelly Criterion."""
        kelly_fraction = self._calculate_kelly_fraction()
        
        # Apply Kelly fraction to portfolio
        kelly_position_value = portfolio_value.amount * kelly_fraction
        
        # Apply position size limits
        max_position_value = portfolio_value.amount * self.config.max_position_size
        position_value = min(kelly_position_value, max_position_value)
        
        return Quantity(position_value / entry_price.value)
    
    def _calculate_kelly_fraction(self) -> Decimal:
        """Calculate Kelly fraction."""
        b = self.avg_win / self.avg_loss
        p = self.win_rate
        q = Decimal('1') - p
        
        kelly_fraction = (b * p - q) / b
        
        # Cap at 25% for safety
        kelly_fraction = min(kelly_fraction, Decimal('0.25'))
        return max(kelly_fraction, Decimal('0'))


class VolatilityTargetPositionSizer(BasePositionSizer):
    """Volatility target position sizer."""
    
    def __init__(self, target_volatility: Decimal, asset_volatility: Decimal, config=None):
        self.target_volatility = target_volatility
        self.asset_volatility = asset_volatility
        self.config = config or get_risk_config()
    
    def calculate_position_size(
        self,
        portfolio_value: Money,
        entry_price: Price,
        stop_loss_price: Price,
        risk_per_trade: Decimal
    ) -> Quantity:
        """Calculate position size based on volatility targeting."""
        if self.asset_volatility <= 0:
            raise ValueError("Asset volatility must be positive")
        
        position_fraction = self.target_volatility / self.asset_volatility
        
        # Apply position size limits
        max_fraction = self.config.max_position_size
        position_fraction = min(position_fraction, max_fraction)
        
        position_value = position_fraction * portfolio_value.amount
        return Quantity(position_value / entry_price.value)


class ATRBasedPositionSizer(BasePositionSizer):
    """ATR-based position sizer."""
    
    def __init__(self, atr_value: Decimal, atr_multiplier: Decimal = None, config=None):
        self.atr_value = atr_value
        self.config = config or get_risk_config()
        self.atr_multiplier = atr_multiplier or self.config.volatility_multiplier
    
    def calculate_position_size(
        self,
        portfolio_value: Money,
        entry_price: Price,
        stop_loss_price: Price,
        risk_per_trade: Decimal
    ) -> Quantity:
        """Calculate position size based on ATR."""
        # Use ATR to determine stop distance
        atr_stop_distance = self.atr_value * self.atr_multiplier
        
        total_risk_amount = portfolio_value.amount * risk_per_trade
        position_size = total_risk_amount / atr_stop_distance
        
        # Apply position size limits
        max_position_value = portfolio_value.amount * self.config.max_position_size
        max_position_size = max_position_value / entry_price.value
        
        return Quantity(min(position_size, max_position_size))


class PositionSizer:
    """Calculate position sizes based on risk management rules."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def calculate_position_size(self, params: PositionSizeParams) -> Quantity:
        """
        Calculate position size based on risk parameters.
        
        Uses the following formula:
        Position Size = (Portfolio Value * Risk Per Trade) / (Entry Price - Stop Loss Price)
        
        Args:
            params: Position sizing parameters
            
        Returns:
            Calculated position quantity
        """
        # Maximum position size as percentage of portfolio
        max_size_pct = params.max_position_size or Decimal(str(self.settings.max_position_size))
        max_position_value = params.portfolio_value.amount * max_size_pct
        
        # If no stop loss provided, use maximum position size
        if params.stop_loss_price is None:
            position_value = min(max_position_value, params.portfolio_value.amount * params.risk_per_trade)
            return Quantity(position_value / params.entry_price.value)
        
        # Calculate risk per share
        risk_per_share = abs(params.entry_price.value - params.stop_loss_price.value)
        if risk_per_share == 0:
            raise ValueError("Entry price and stop loss price cannot be the same")
        
        # Calculate total risk amount
        total_risk_amount = params.portfolio_value.amount * params.risk_per_trade
        
        # Calculate position size based on risk
        risk_based_quantity = total_risk_amount / risk_per_share
        
        # Calculate maximum quantity based on position size limit
        max_quantity = max_position_value / params.entry_price.value
        
        # Return the smaller of the two
        final_quantity = min(risk_based_quantity, max_quantity)
        
        return Quantity(max(Decimal('0'), final_quantity))
    
    def calculate_kelly_position_size(
        self, 
        win_rate: Decimal, 
        avg_win: Decimal, 
        avg_loss: Decimal,
        portfolio_value: Money
    ) -> Decimal:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly % = (bp - q) / b
        where:
        b = odds of winning (avg_win / avg_loss)
        p = probability of winning (win_rate)
        q = probability of losing (1 - win_rate)
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive number)
            portfolio_value: Current portfolio value
            
        Returns:
            Percentage of portfolio to risk (0-1)
        """
        if avg_loss <= 0:
            raise ValueError("Average loss must be positive")
        
        if not (0 <= win_rate <= 1):
            raise ValueError("Win rate must be between 0 and 1")
        
        # Kelly calculation
        b = avg_win / avg_loss  # Odds ratio
        p = win_rate  # Win probability
        q = Decimal('1') - p  # Loss probability
        
        kelly_fraction = (b * p - q) / b
        
        # Cap Kelly at 25% for safety (Kelly can suggest very high leverage)
        kelly_fraction = min(kelly_fraction, Decimal('0.25'))
        
        # Ensure non-negative
        kelly_fraction = max(kelly_fraction, Decimal('0'))
        
        return kelly_fraction
    
    def calculate_volatility_position_size(
        self,
        portfolio_value: Money,
        target_volatility: Decimal,
        asset_volatility: Decimal
    ) -> Decimal:
        """
        Calculate position size based on volatility targeting.
        
        Position Size = (Target Vol / Asset Vol) * Portfolio Value
        
        Args:
            portfolio_value: Current portfolio value
            target_volatility: Target portfolio volatility (annualized)
            asset_volatility: Asset's historical volatility (annualized)
            
        Returns:
            Position value to achieve target volatility
        """
        if asset_volatility <= 0:
            raise ValueError("Asset volatility must be positive")
        
        position_fraction = target_volatility / asset_volatility
        
        # Cap at maximum position size
        max_fraction = Decimal(str(self.settings.max_position_size))
        position_fraction = min(position_fraction, max_fraction)
        
        return position_fraction * portfolio_value.amount