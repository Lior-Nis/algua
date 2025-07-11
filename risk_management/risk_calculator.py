"""
Risk calculation utilities.
"""

from decimal import Decimal
from typing import List, Optional, Tuple
import statistics
import math

from domain.value_objects import Money, Price


class RiskCalculator:
    """Calculate various risk metrics."""
    
    def calculate_var(
        self, 
        returns: List[Decimal], 
        confidence_level: float = 0.95
    ) -> Decimal:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: List of returns
            confidence_level: Confidence level (0.95 = 95%)
            
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return Decimal('0')
        
        # Sort returns
        sorted_returns = sorted([float(r) for r in returns])
        
        # Calculate percentile index
        percentile = (1 - confidence_level)
        index = int(percentile * len(sorted_returns))
        
        if index >= len(sorted_returns):
            index = len(sorted_returns) - 1
        
        var = abs(sorted_returns[index])
        return Decimal(str(var))
    
    def calculate_expected_shortfall(
        self, 
        returns: List[Decimal], 
        confidence_level: float = 0.95
    ) -> Decimal:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: List of returns
            confidence_level: Confidence level
            
        Returns:
            Expected Shortfall value
        """
        if len(returns) == 0:
            return Decimal('0')
        
        var_value = float(self.calculate_var(returns, confidence_level))
        
        # Calculate mean of returns below VaR
        tail_returns = [float(r) for r in returns if float(r) <= -var_value]
        
        if len(tail_returns) == 0:
            return Decimal(str(var_value))
        
        expected_shortfall = abs(statistics.mean(tail_returns))
        return Decimal(str(expected_shortfall))
    
    def calculate_sharpe_ratio(
        self, 
        returns: List[Decimal], 
        risk_free_rate: float = 0.02
    ) -> Decimal:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: List of returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return Decimal('0')
        
        float_returns = [float(r) for r in returns]
        
        if len(float_returns) < 2:
            return Decimal('0')
        
        try:
            mean_return = statistics.mean(float_returns)
            std_return = statistics.stdev(float_returns)
            
            if std_return == 0:
                return Decimal('0')
            
            # Annualize returns and volatility
            annual_return = mean_return * 252
            annual_volatility = std_return * math.sqrt(252)
            
            sharpe = (annual_return - risk_free_rate) / annual_volatility
            return Decimal(str(sharpe))
            
        except Exception:
            return Decimal('0')
    
    def calculate_max_drawdown(self, equity_curve: List[Tuple[int, Decimal]]) -> Decimal:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: List of (timestamp, value) tuples
            
        Returns:
            Maximum drawdown as percentage
        """
        if len(equity_curve) == 0:
            return Decimal('0')
        
        values = [float(value) for _, value in equity_curve]
        
        if len(values) == 0:
            return Decimal('0')
        
        max_drawdown = 0.0
        peak = values[0]
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return Decimal(str(max_drawdown))
    
    def calculate_portfolio_beta(
        self, 
        portfolio_returns: List[Decimal], 
        market_returns: List[Decimal]
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
        
        # Align data to same length
        min_length = min(len(portfolio_returns), len(market_returns))
        portfolio_data = [float(portfolio_returns[i]) for i in range(min_length)]
        market_data = [float(market_returns[i]) for i in range(min_length)]
        
        if len(portfolio_data) < 2:
            return Decimal('0')
        
        try:
            # Calculate means
            portfolio_mean = statistics.mean(portfolio_data)
            market_mean = statistics.mean(market_data)
            
            # Calculate covariance and variance
            covariance = sum(
                (p - portfolio_mean) * (m - market_mean) 
                for p, m in zip(portfolio_data, market_data)
            ) / (len(portfolio_data) - 1)
            
            market_variance = statistics.variance(market_data)
            
            if market_variance == 0:
                return Decimal('0')
            
            beta = covariance / market_variance
            return Decimal(str(beta))
            
        except Exception:
            return Decimal('0')
    
    def calculate_volatility(self, returns: List[Decimal], annualize: bool = True) -> Decimal:
        """
        Calculate volatility (standard deviation of returns).
        
        Args:
            returns: List of returns
            annualize: Whether to annualize the volatility
            
        Returns:
            Volatility
        """
        if len(returns) < 2:
            return Decimal('0')
        
        try:
            float_returns = [float(r) for r in returns]
            std_dev = statistics.stdev(float_returns)
            
            if annualize:
                # Annualize assuming daily returns
                std_dev *= math.sqrt(252)
            
            return Decimal(str(std_dev))
            
        except Exception:
            return Decimal('0')
    
    def calculate_correlation(
        self, 
        returns1: List[Decimal], 
        returns2: List[Decimal]
    ) -> Decimal:
        """
        Calculate correlation between two return series.
        
        Args:
            returns1: First return series
            returns2: Second return series
            
        Returns:
            Correlation coefficient
        """
        if len(returns1) == 0 or len(returns2) == 0:
            return Decimal('0')
        
        # Align data
        min_length = min(len(returns1), len(returns2))
        data1 = [float(returns1[i]) for i in range(min_length)]
        data2 = [float(returns2[i]) for i in range(min_length)]
        
        if len(data1) < 2:
            return Decimal('0')
        
        try:
            correlation = statistics.correlation(data1, data2)
            return Decimal(str(correlation))
        except Exception:
            return Decimal('0')
    
    def calculate_information_ratio(
        self,
        portfolio_returns: List[Decimal],
        benchmark_returns: List[Decimal]
    ) -> Decimal:
        """
        Calculate information ratio.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return Decimal('0')
        
        # Calculate excess returns
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        excess_returns = [
            float(portfolio_returns[i]) - float(benchmark_returns[i])
            for i in range(min_length)
        ]
        
        if len(excess_returns) < 2:
            return Decimal('0')
        
        try:
            mean_excess = statistics.mean(excess_returns)
            tracking_error = statistics.stdev(excess_returns)
            
            if tracking_error == 0:
                return Decimal('0')
            
            # Annualize
            annual_mean_excess = mean_excess * 252
            annual_tracking_error = tracking_error * math.sqrt(252)
            
            information_ratio = annual_mean_excess / annual_tracking_error
            return Decimal(str(information_ratio))
            
        except Exception:
            return Decimal('0')