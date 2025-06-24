"""
Risk calculation utilities.
"""

from decimal import Decimal
from typing import List, Optional
import pandas as pd
import numpy as np

from domain.value_objects import Money, Price


class RiskCalculator:
    """Calculate various risk metrics."""
    
    def calculate_var(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.95
    ) -> Decimal:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (0.95 = 95%)
            
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return Decimal('0')
        
        # Calculate percentile
        percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, percentile)
        
        return Decimal(str(abs(var)))
    
    def calculate_expected_shortfall(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.95
    ) -> Decimal:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level
            
        Returns:
            Expected Shortfall value
        """
        if len(returns) == 0:
            return Decimal('0')
        
        var = float(self.calculate_var(returns, confidence_level))
        
        # Calculate mean of returns below VaR
        tail_returns = returns[returns <= -var]
        
        if len(tail_returns) == 0:
            return Decimal(str(var))
        
        expected_shortfall = abs(tail_returns.mean())
        return Decimal(str(expected_shortfall))
    
    def calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = 0.02
    ) -> Decimal:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return Decimal('0')
        
        # Annualize returns and volatility
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        sharpe = (annual_return - risk_free_rate) / annual_volatility
        return Decimal(str(sharpe))
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Decimal:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Portfolio value over time
            
        Returns:
            Maximum drawdown as percentage
        """
        if len(equity_curve) == 0:
            return Decimal('0')
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        max_drawdown = abs(drawdown.min())
        return Decimal(str(max_drawdown))
    
    def calculate_portfolio_beta(
        self, 
        portfolio_returns: pd.Series, 
        market_returns: pd.Series
    ) -> Decimal:
        """
        Calculate portfolio beta relative to market.
        
        Args:
            portfolio_returns: Portfolio returns
            market_returns: Market returns (benchmark)
            
        Returns:
            Portfolio beta
        """
        if len(portfolio_returns) == 0 or len(market_returns) == 0:
            return Decimal('0')
        
        # Align data
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned_data) < 2:
            return Decimal('0')
        
        # Calculate covariance and variance
        covariance = aligned_data['portfolio'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()
        
        if market_variance == 0:
            return Decimal('0')
        
        beta = covariance / market_variance
        return Decimal(str(beta))