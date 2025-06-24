"""
Risk management orchestration.
"""

from typing import List, Dict, Any
from decimal import Decimal
from dataclasses import dataclass

from .position_sizer import PositionSizer, PositionSizeParams
from .risk_calculator import RiskCalculator
from domain.value_objects import Money, Price, Quantity
from configs.settings import get_settings


@dataclass
class RiskAssessment:
    """Risk assessment result."""
    approved: bool
    risk_score: Decimal
    warnings: List[str]
    recommendations: List[str]


class RiskManager:
    """Central risk management orchestrator."""
    
    def __init__(self):
        self.settings = get_settings()
        self.position_sizer = PositionSizer()
        self.risk_calculator = RiskCalculator()
    
    def assess_trade_risk(
        self,
        portfolio_value: Money,
        entry_price: Price,
        stop_loss_price: Price,
        position_size: Quantity
    ) -> RiskAssessment:
        """
        Assess risk for a proposed trade.
        
        Args:
            portfolio_value: Current portfolio value
            entry_price: Proposed entry price
            stop_loss_price: Stop loss price
            position_size: Proposed position size
            
        Returns:
            Risk assessment
        """
        warnings = []
        recommendations = []
        risk_score = Decimal('0')
        
        # Calculate position value
        position_value = entry_price.value * position_size.value
        position_percentage = position_value / portfolio_value.amount
        
        # Check position size limits
        max_position_size = Decimal(str(self.settings.max_position_size))
        if position_percentage > max_position_size:
            warnings.append(
                f"Position size {position_percentage:.2%} exceeds limit {max_position_size:.2%}"
            )
            risk_score += Decimal('30')
        
        # Calculate risk per trade
        risk_per_share = abs(entry_price.value - stop_loss_price.value)
        total_risk = risk_per_share * position_size.value
        risk_percentage = total_risk / portfolio_value.amount
        
        max_daily_loss = Decimal(str(self.settings.max_daily_loss))
        if risk_percentage > max_daily_loss:
            warnings.append(
                f"Trade risk {risk_percentage:.2%} exceeds daily limit {max_daily_loss:.2%}"
            )
            risk_score += Decimal('40')
        
        # Risk/reward ratio
        if stop_loss_price.value != 0:
            risk_reward_ratio = risk_per_share / abs(entry_price.value - stop_loss_price.value)
            if risk_reward_ratio < 1:
                recommendations.append("Consider better risk/reward ratio")
                risk_score += Decimal('10')
        
        # Determine approval
        approved = risk_score < 50 and len(warnings) == 0
        
        return RiskAssessment(
            approved=approved,
            risk_score=risk_score,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def calculate_optimal_position_size(
        self,
        portfolio_value: Money,
        entry_price: Price,
        stop_loss_price: Price,
        risk_per_trade: Decimal = None
    ) -> Quantity:
        """
        Calculate optimal position size based on risk management rules.
        
        Args:
            portfolio_value: Current portfolio value
            entry_price: Entry price
            stop_loss_price: Stop loss price
            risk_per_trade: Risk per trade (default from settings)
            
        Returns:
            Optimal position size
        """
        if risk_per_trade is None:
            risk_per_trade = Decimal(str(self.settings.max_daily_loss))
        
        params = PositionSizeParams(
            portfolio_value=portfolio_value,
            risk_per_trade=risk_per_trade,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price
        )
        
        return self.position_sizer.calculate_position_size(params)