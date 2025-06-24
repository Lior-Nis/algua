"""
Portfolio performance calculation utilities.
"""

import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from domain.value_objects import Money
from domain.entities import Portfolio


class PerformanceCalculator:
    """Calculate portfolio performance metrics."""
    
    def __init__(self):
        pass
    
    def calculate_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """
        Calculate portfolio returns from value series.
        
        Args:
            portfolio_values: Time series of portfolio values
            
        Returns:
            Series of returns
        """
        if len(portfolio_values) < 2:
            return pd.Series(dtype=float)
        
        return portfolio_values.pct_change().dropna()
    
    def calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: Decimal = Decimal('0.02')
    ) -> Decimal:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return Decimal('0')
        
        # Annualize returns and volatility (assuming daily returns)
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        sharpe = (annual_return - float(risk_free_rate)) / annual_volatility
        return Decimal(str(sharpe))
    
    def calculate_sortino_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: Decimal = Decimal('0.02')
    ) -> Decimal:
        """
        Calculate Sortino ratio (only considers downside volatility).
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return Decimal('0')
        
        # Calculate downside deviation
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return Decimal('999')  # Very high ratio if no negative returns
        
        annual_return = returns.mean() * 252
        downside_volatility = negative_returns.std() * np.sqrt(252)
        
        if downside_volatility == 0:
            return Decimal('999')
        
        sortino = (annual_return - float(risk_free_rate)) / downside_volatility
        return Decimal(str(sortino))
    
    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> Tuple[Decimal, datetime, datetime]:
        """
        Calculate maximum drawdown and its dates.
        
        Args:
            portfolio_values: Time series of portfolio values
            
        Returns:
            Tuple of (max_drawdown_pct, peak_date, trough_date)
        """
        if len(portfolio_values) == 0:
            return Decimal('0'), datetime.now(), datetime.now()
        
        # Calculate running maximum
        running_max = portfolio_values.expanding().max()
        
        # Calculate drawdown
        drawdown = (portfolio_values - running_max) / running_max
        
        # Find maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_drawdown = abs(drawdown.min())
        
        # Find peak before max drawdown
        peak_idx = running_max.loc[:max_dd_idx].idxmax()
        
        peak_date = portfolio_values.index[peak_idx] if hasattr(portfolio_values.index[peak_idx], 'date') else datetime.now()
        trough_date = portfolio_values.index[max_dd_idx] if hasattr(portfolio_values.index[max_dd_idx], 'date') else datetime.now()
        
        return Decimal(str(max_drawdown)), peak_date, trough_date
    
    def calculate_calmar_ratio(self, returns: pd.Series, portfolio_values: pd.Series) -> Decimal:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            returns: Series of returns
            portfolio_values: Time series of portfolio values
            
        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return Decimal('0')
        
        annual_return = returns.mean() * 252
        max_drawdown, _, _ = self.calculate_max_drawdown(portfolio_values)
        
        if max_drawdown == 0:
            return Decimal('999')
        
        calmar = annual_return / float(max_drawdown)
        return Decimal(str(calmar))
    
    def calculate_var(
        self, 
        returns: pd.Series, 
        confidence_level: Decimal = Decimal('0.95')
    ) -> Decimal:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return Decimal('0')
        
        percentile = (1 - float(confidence_level)) * 100
        var = np.percentile(returns, percentile)
        
        return Decimal(str(abs(var)))
    
    def calculate_beta(
        self, 
        portfolio_returns: pd.Series, 
        benchmark_returns: pd.Series
    ) -> Decimal:
        """
        Calculate portfolio beta relative to benchmark.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Beta value
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return Decimal('0')
        
        # Align data
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) < 2:
            return Decimal('0')
        
        covariance = aligned_data['portfolio'].cov(aligned_data['benchmark'])
        benchmark_variance = aligned_data['benchmark'].var()
        
        if benchmark_variance == 0:
            return Decimal('0')
        
        beta = covariance / benchmark_variance
        return Decimal(str(beta))
    
    def calculate_alpha(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: Decimal = Decimal('0.02')
    ) -> Decimal:
        """
        Calculate portfolio alpha (Jensen's alpha).
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Alpha value
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return Decimal('0')
        
        # Calculate beta
        beta = self.calculate_beta(portfolio_returns, benchmark_returns)
        
        # Calculate average returns
        portfolio_return = portfolio_returns.mean() * 252
        benchmark_return = benchmark_returns.mean() * 252
        
        # Jensen's alpha formula
        alpha = portfolio_return - (float(risk_free_rate) + float(beta) * (benchmark_return - float(risk_free_rate)))
        
        return Decimal(str(alpha))
    
    def calculate_information_ratio(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Decimal:
        """
        Calculate Information Ratio.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return Decimal('0')
        
        # Align data
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned_data) < 2:
            return Decimal('0')
        
        # Calculate excess returns
        excess_returns = aligned_data['portfolio'] - aligned_data['benchmark']
        
        if excess_returns.std() == 0:
            return Decimal('0')
        
        # Information ratio = excess return / tracking error
        ir = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
        
        return Decimal(str(ir))
    
    def calculate_win_rate(self, returns: pd.Series) -> Decimal:
        """
        Calculate win rate (percentage of positive returns).
        
        Args:
            returns: Series of returns
            
        Returns:
            Win rate as percentage
        """
        if len(returns) == 0:
            return Decimal('0')
        
        positive_returns = returns > 0
        win_rate = positive_returns.mean() * 100
        
        return Decimal(str(win_rate))
    
    def calculate_profit_factor(self, returns: pd.Series) -> Decimal:
        """
        Calculate profit factor (gross profits / gross losses).
        
        Args:
            returns: Series of returns
            
        Returns:
            Profit factor
        """
        if len(returns) == 0:
            return Decimal('0')
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        gross_profits = positive_returns.sum()
        gross_losses = abs(negative_returns.sum())
        
        if gross_losses == 0:
            return Decimal('999') if gross_profits > 0 else Decimal('0')
        
        profit_factor = gross_profits / gross_losses
        return Decimal(str(profit_factor))
    
    def generate_performance_report(
        self,
        portfolio_values: pd.Series,
        benchmark_values: Optional[pd.Series] = None,
        risk_free_rate: Decimal = Decimal('0.02')
    ) -> Dict[str, Decimal]:
        """
        Generate comprehensive performance report.
        
        Args:
            portfolio_values: Time series of portfolio values
            benchmark_values: Optional benchmark values
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of performance metrics
        """
        if len(portfolio_values) == 0:
            return {}
        
        # Calculate returns
        returns = self.calculate_returns(portfolio_values)
        
        # Basic metrics
        total_return = ((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1) * 100
        annual_return = returns.mean() * 252 * 100
        annual_volatility = returns.std() * np.sqrt(252) * 100
        
        # Risk metrics
        sharpe_ratio = self.calculate_sharpe_ratio(returns, risk_free_rate)
        sortino_ratio = self.calculate_sortino_ratio(returns, risk_free_rate)
        max_drawdown, peak_date, trough_date = self.calculate_max_drawdown(portfolio_values)
        calmar_ratio = self.calculate_calmar_ratio(returns, portfolio_values)
        var_95 = self.calculate_var(returns, Decimal('0.95'))
        
        # Trading metrics
        win_rate = self.calculate_win_rate(returns)
        profit_factor = self.calculate_profit_factor(returns)
        
        report = {
            'total_return_pct': Decimal(str(total_return)),
            'annual_return_pct': Decimal(str(annual_return)),
            'annual_volatility_pct': Decimal(str(annual_volatility)),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'calmar_ratio': calmar_ratio,
            'var_95_pct': var_95 * 100,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(returns),
            'winning_trades': len(returns[returns > 0]),
            'losing_trades': len(returns[returns < 0])
        }
        
        # Add benchmark comparison if provided
        if benchmark_values is not None and len(benchmark_values) > 0:
            benchmark_returns = self.calculate_returns(benchmark_values)
            beta = self.calculate_beta(returns, benchmark_returns)
            alpha = self.calculate_alpha(returns, benchmark_returns, risk_free_rate)
            information_ratio = self.calculate_information_ratio(returns, benchmark_returns)
            
            report.update({
                'beta': beta,
                'alpha_pct': alpha * 100,
                'information_ratio': information_ratio
            })
        
        return report