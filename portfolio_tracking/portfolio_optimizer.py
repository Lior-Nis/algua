"""
Portfolio optimization system for Algua trading platform.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from decimal import Decimal
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from enum import Enum
import math
from abc import ABC, abstractmethod
import threading
from collections import defaultdict

from domain.value_objects import Symbol, Price, Quantity, Money
from .portfolio_manager import PortfolioManager, PortfolioConfiguration
from .position_manager import PositionManager, Position
from .performance_analytics import PerformanceAnalyzer, PerformancePeriod
from utils.logging import get_logger

logger = get_logger(__name__)


class OptimizationObjective(Enum):
    """Portfolio optimization objectives."""
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_VARIANCE = "minimize_variance"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    MINIMUM_CORRELATION = "minimum_correlation"


class ConstraintType(Enum):
    """Types of optimization constraints."""
    WEIGHT_BOUNDS = "weight_bounds"
    SECTOR_LIMITS = "sector_limits"
    POSITION_LIMITS = "position_limits"
    TURNOVER_LIMITS = "turnover_limits"
    LEVERAGE_LIMITS = "leverage_limits"
    CONCENTRATION_LIMITS = "concentration_limits"
    TRACKING_ERROR_LIMITS = "tracking_error_limits"


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints."""
    # Weight constraints
    min_weight: Decimal = Decimal('0.0')
    max_weight: Decimal = Decimal('1.0')
    weight_bounds: Dict[Symbol, Tuple[Decimal, Decimal]] = field(default_factory=dict)
    
    # Portfolio constraints
    max_positions: Optional[int] = None
    min_positions: Optional[int] = None
    max_turnover: Optional[Decimal] = None
    max_leverage: Decimal = Decimal('1.0')
    
    # Risk constraints
    max_portfolio_volatility: Optional[Decimal] = None
    max_tracking_error: Optional[Decimal] = None
    max_concentration: Decimal = Decimal('0.20')  # 20% max in single position
    
    # Sector constraints
    sector_limits: Dict[str, Tuple[Decimal, Decimal]] = field(default_factory=dict)
    
    # Transaction constraints
    min_trade_size: Money = field(default_factory=lambda: Money(Decimal('100')))
    max_trade_size: Optional[Money] = None
    
    # Time constraints
    rebalance_frequency_days: int = 30
    max_holding_period_days: Optional[int] = None
    min_holding_period_days: int = 1
    
    def is_valid_allocation(self, allocation: Dict[Symbol, Decimal]) -> bool:
        """Check if allocation satisfies constraints."""
        total_weight = sum(allocation.values())
        
        # Check total weight
        if abs(total_weight - Decimal('1.0')) > Decimal('0.01'):
            return False
        
        # Check individual weight bounds
        for symbol, weight in allocation.items():
            if weight < self.min_weight or weight > self.max_weight:
                return False
            
            # Check symbol-specific bounds
            if symbol in self.weight_bounds:
                min_bound, max_bound = self.weight_bounds[symbol]
                if weight < min_bound or weight > max_bound:
                    return False
        
        # Check concentration
        if max(allocation.values()) > self.max_concentration:
            return False
        
        # Check position count
        active_positions = sum(1 for w in allocation.values() if w > Decimal('0.01'))
        if self.max_positions and active_positions > self.max_positions:
            return False
        if self.min_positions and active_positions < self.min_positions:
            return False
        
        return True


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    objective: OptimizationObjective
    optimization_time: datetime
    success: bool
    
    # Optimal allocation
    optimal_weights: Dict[Symbol, Decimal]
    expected_return: Decimal
    expected_volatility: Decimal
    expected_sharpe: Decimal
    
    # Performance metrics
    optimization_score: Decimal
    diversification_ratio: Decimal
    concentration_index: Decimal
    
    # Risk metrics
    portfolio_var: Decimal
    portfolio_cvar: Decimal
    maximum_drawdown_estimate: Decimal
    
    # Trade requirements
    required_trades: List[Dict[str, Any]]
    total_turnover: Decimal
    transaction_costs: Money
    
    # Optimization details
    iterations: int
    convergence_achieved: bool
    error_message: Optional[str] = None
    
    # Diagnostics
    eigenvalue_concentration: Optional[Decimal] = None
    condition_number: Optional[Decimal] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'objective': self.objective.value,
            'optimization_time': self.optimization_time.isoformat(),
            'success': self.success,
            'optimal_weights': {str(symbol): float(weight) for symbol, weight in self.optimal_weights.items()},
            'expected_return': float(self.expected_return),
            'expected_volatility': float(self.expected_volatility),
            'expected_sharpe': float(self.expected_sharpe),
            'optimization_score': float(self.optimization_score),
            'diversification_ratio': float(self.diversification_ratio),
            'concentration_index': float(self.concentration_index),
            'portfolio_var': float(self.portfolio_var),
            'portfolio_cvar': float(self.portfolio_cvar),
            'maximum_drawdown_estimate': float(self.maximum_drawdown_estimate),
            'total_turnover': float(self.total_turnover),
            'transaction_costs': float(self.transaction_costs.amount),
            'iterations': self.iterations,
            'convergence_achieved': self.convergence_achieved,
            'error_message': self.error_message,
            'eigenvalue_concentration': float(self.eigenvalue_concentration) if self.eigenvalue_concentration else None,
            'condition_number': float(self.condition_number) if self.condition_number else None
        }


@dataclass
class MarketData:
    """Market data for optimization."""
    symbols: List[Symbol]
    returns: Dict[Symbol, List[Decimal]]
    prices: Dict[Symbol, Price]
    volumes: Dict[Symbol, Decimal]
    market_caps: Dict[Symbol, Money]
    
    # Risk data
    correlations: Dict[Tuple[Symbol, Symbol], Decimal] = field(default_factory=dict)
    volatilities: Dict[Symbol, Decimal] = field(default_factory=dict)
    betas: Dict[Symbol, Decimal] = field(default_factory=dict)
    
    # Fundamental data
    pe_ratios: Dict[Symbol, Decimal] = field(default_factory=dict)
    dividend_yields: Dict[Symbol, Decimal] = field(default_factory=dict)
    sectors: Dict[Symbol, str] = field(default_factory=dict)
    
    # Factor exposures
    factor_exposures: Dict[Symbol, Dict[str, Decimal]] = field(default_factory=dict)
    
    def get_correlation_matrix(self) -> Dict[Symbol, Dict[Symbol, Decimal]]:
        """Get correlation matrix."""
        matrix = {}
        for symbol1 in self.symbols:
            matrix[symbol1] = {}
            for symbol2 in self.symbols:
                if symbol1 == symbol2:
                    matrix[symbol1][symbol2] = Decimal('1.0')
                else:
                    key = (symbol1, symbol2) if str(symbol1) < str(symbol2) else (symbol2, symbol1)
                    matrix[symbol1][symbol2] = self.correlations.get(key, Decimal('0.5'))
        return matrix
    
    def get_covariance_matrix(self) -> Dict[Symbol, Dict[Symbol, Decimal]]:
        """Get covariance matrix."""
        corr_matrix = self.get_correlation_matrix()
        cov_matrix = {}
        
        for symbol1 in self.symbols:
            cov_matrix[symbol1] = {}
            vol1 = self.volatilities.get(symbol1, Decimal('0.2'))
            
            for symbol2 in self.symbols:
                vol2 = self.volatilities.get(symbol2, Decimal('0.2'))
                correlation = corr_matrix[symbol1][symbol2]
                covariance = vol1 * vol2 * correlation
                cov_matrix[symbol1][symbol2] = covariance
        
        return cov_matrix


class PortfolioOptimizer(ABC):
    """Abstract base class for portfolio optimizers."""
    
    def __init__(self, constraints: OptimizationConstraints = None):
        self.constraints = constraints or OptimizationConstraints()
        self._lock = threading.Lock()
    
    @abstractmethod
    def optimize(
        self,
        market_data: MarketData,
        current_allocation: Dict[Symbol, Decimal] = None,
        objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_SHARPE
    ) -> OptimizationResult:
        """Perform portfolio optimization."""
        pass
    
    def _calculate_portfolio_metrics(
        self,
        weights: Dict[Symbol, Decimal],
        market_data: MarketData
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Calculate portfolio return, volatility, and Sharpe ratio."""
        # Expected return
        expected_return = Decimal('0')
        for symbol, weight in weights.items():
            if symbol in market_data.returns and market_data.returns[symbol]:
                avg_return = sum(market_data.returns[symbol]) / len(market_data.returns[symbol])
                expected_return += weight * avg_return
        
        # Portfolio volatility
        cov_matrix = market_data.get_covariance_matrix()
        portfolio_variance = Decimal('0')
        
        for symbol1, weight1 in weights.items():
            for symbol2, weight2 in weights.items():
                if symbol1 in cov_matrix and symbol2 in cov_matrix[symbol1]:
                    covariance = cov_matrix[symbol1][symbol2]
                    portfolio_variance += weight1 * weight2 * covariance
        
        portfolio_volatility = portfolio_variance ** Decimal('0.5') if portfolio_variance > 0 else Decimal('0')
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = Decimal('0.02') / Decimal('252')  # Daily rate
        sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else Decimal('0')
        
        return expected_return, portfolio_volatility, sharpe_ratio
    
    def _calculate_diversification_ratio(
        self,
        weights: Dict[Symbol, Decimal],
        market_data: MarketData
    ) -> Decimal:
        """Calculate diversification ratio."""
        # Weighted average of individual volatilities
        weighted_vol_sum = Decimal('0')
        for symbol, weight in weights.items():
            individual_vol = market_data.volatilities.get(symbol, Decimal('0.2'))
            weighted_vol_sum += weight * individual_vol
        
        # Portfolio volatility
        _, portfolio_vol, _ = self._calculate_portfolio_metrics(weights, market_data)
        
        # Diversification ratio
        if portfolio_vol > 0:
            return weighted_vol_sum / portfolio_vol
        else:
            return Decimal('1')
    
    def _calculate_concentration_index(self, weights: Dict[Symbol, Decimal]) -> Decimal:
        """Calculate Herfindahl concentration index."""
        return sum(weight ** 2 for weight in weights.values())
    
    def _calculate_turnover(
        self,
        current_weights: Dict[Symbol, Decimal],
        target_weights: Dict[Symbol, Decimal]
    ) -> Decimal:
        """Calculate portfolio turnover."""
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        turnover = Decimal('0')
        for symbol in all_symbols:
            current_weight = current_weights.get(symbol, Decimal('0'))
            target_weight = target_weights.get(symbol, Decimal('0'))
            turnover += abs(target_weight - current_weight)
        
        return turnover / 2  # Divide by 2 as turnover is one-way
    
    def _estimate_transaction_costs(
        self,
        trades: List[Dict[str, Any]],
        market_data: MarketData
    ) -> Money:
        """Estimate transaction costs."""
        total_cost = Decimal('0')
        
        for trade in trades:
            symbol = trade['symbol']
            trade_value = abs(trade['trade_value'])
            
            # Simple cost model: 0.1% of trade value
            cost_rate = Decimal('0.001')
            trade_cost = trade_value * cost_rate
            total_cost += trade_cost
        
        return Money(total_cost)
    
    def _generate_required_trades(
        self,
        current_allocation: Dict[Symbol, Decimal],
        target_allocation: Dict[Symbol, Decimal],
        portfolio_value: Money
    ) -> List[Dict[str, Any]]:
        """Generate required trades to achieve target allocation."""
        trades = []
        all_symbols = set(current_allocation.keys()) | set(target_allocation.keys())
        
        for symbol in all_symbols:
            current_weight = current_allocation.get(symbol, Decimal('0'))
            target_weight = target_allocation.get(symbol, Decimal('0'))
            
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > Decimal('0.005'):  # 0.5% threshold
                trade_value = weight_diff * portfolio_value.amount
                
                trade = {
                    'symbol': symbol,
                    'action': 'buy' if weight_diff > 0 else 'sell',
                    'current_weight': float(current_weight),
                    'target_weight': float(target_weight),
                    'weight_change': float(weight_diff),
                    'trade_value': float(abs(trade_value)),
                    'trade_direction': 'long' if target_weight > 0 else 'flat'
                }
                
                trades.append(trade)
        
        return trades


class MeanVarianceOptimizer(PortfolioOptimizer):
    """Mean-variance optimization (Markowitz)."""
    
    def __init__(self, constraints: OptimizationConstraints = None):
        super().__init__(constraints)
        self.risk_aversion = Decimal('5.0')  # Default risk aversion parameter
    
    def optimize(
        self,
        market_data: MarketData,
        current_allocation: Dict[Symbol, Decimal] = None,
        objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_SHARPE
    ) -> OptimizationResult:
        """Perform mean-variance optimization."""
        with self._lock:
            start_time = datetime.now()
            
            try:
                # Initialize current allocation if not provided
                if current_allocation is None:
                    current_allocation = {symbol: Decimal('0') for symbol in market_data.symbols}
                
                # Calculate expected returns and covariance matrix
                expected_returns = self._calculate_expected_returns(market_data)
                cov_matrix = market_data.get_covariance_matrix()
                
                # Perform optimization based on objective
                if objective == OptimizationObjective.MAXIMIZE_SHARPE:
                    optimal_weights = self._maximize_sharpe_ratio(expected_returns, cov_matrix, market_data)
                elif objective == OptimizationObjective.MINIMIZE_RISK:
                    optimal_weights = self._minimize_variance(cov_matrix, market_data)
                elif objective == OptimizationObjective.MAXIMIZE_RETURN:
                    optimal_weights = self._maximize_return(expected_returns, cov_matrix, market_data)
                else:
                    optimal_weights = self._maximize_sharpe_ratio(expected_returns, cov_matrix, market_data)
                
                # Calculate portfolio metrics
                exp_return, exp_vol, exp_sharpe = self._calculate_portfolio_metrics(optimal_weights, market_data)
                
                # Calculate additional metrics
                diversification_ratio = self._calculate_diversification_ratio(optimal_weights, market_data)
                concentration_index = self._calculate_concentration_index(optimal_weights)
                
                # Calculate VaR (simplified)
                portfolio_var = exp_vol * Decimal('1.65')  # 95% VaR approximation
                portfolio_cvar = exp_vol * Decimal('2.33')  # 99% CVaR approximation
                
                # Estimate maximum drawdown
                max_dd_estimate = exp_vol * Decimal('2.5')  # Rough estimate
                
                # Calculate turnover and trades
                turnover = self._calculate_turnover(current_allocation, optimal_weights)
                portfolio_value = Money(Decimal('100000'))  # Assume $100k portfolio for trade calculation
                required_trades = self._generate_required_trades(current_allocation, optimal_weights, portfolio_value)
                transaction_costs = self._estimate_transaction_costs(required_trades, market_data)
                
                return OptimizationResult(
                    objective=objective,
                    optimization_time=start_time,
                    success=True,
                    optimal_weights=optimal_weights,
                    expected_return=exp_return,
                    expected_volatility=exp_vol,
                    expected_sharpe=exp_sharpe,
                    optimization_score=exp_sharpe,  # Use Sharpe as score
                    diversification_ratio=diversification_ratio,
                    concentration_index=concentration_index,
                    portfolio_var=portfolio_var,
                    portfolio_cvar=portfolio_cvar,
                    maximum_drawdown_estimate=max_dd_estimate,
                    required_trades=required_trades,
                    total_turnover=turnover,
                    transaction_costs=transaction_costs,
                    iterations=100,  # Simulated iterations
                    convergence_achieved=True
                )
                
            except Exception as e:
                logger.error(f"Mean-variance optimization failed: {e}")
                return self._create_failed_result(objective, start_time, str(e))
    
    def _calculate_expected_returns(self, market_data: MarketData) -> Dict[Symbol, Decimal]:
        """Calculate expected returns for each symbol."""
        expected_returns = {}
        
        for symbol in market_data.symbols:
            if symbol in market_data.returns and market_data.returns[symbol]:
                returns = market_data.returns[symbol]
                # Simple average
                expected_returns[symbol] = sum(returns) / len(returns)
            else:
                # Default expected return
                expected_returns[symbol] = Decimal('0.08') / Decimal('252')  # 8% annualized
        
        return expected_returns
    
    def _maximize_sharpe_ratio(
        self,
        expected_returns: Dict[Symbol, Decimal],
        cov_matrix: Dict[Symbol, Dict[Symbol, Decimal]],
        market_data: MarketData
    ) -> Dict[Symbol, Decimal]:
        """Maximize Sharpe ratio (simplified implementation)."""
        symbols = list(market_data.symbols)
        n = len(symbols)
        
        if n == 0:
            return {}
        
        # Equal weight as starting point
        equal_weight = Decimal('1') / Decimal(str(n))
        weights = {symbol: equal_weight for symbol in symbols}
        
        # Simple iterative improvement (simplified optimization)
        best_sharpe = Decimal('0')
        best_weights = weights.copy()
        
        for _ in range(50):  # 50 iterations
            # Try small perturbations
            for i, symbol in enumerate(symbols):
                for delta in [Decimal('0.01'), Decimal('-0.01')]:
                    test_weights = weights.copy()
                    
                    # Adjust weight
                    test_weights[symbol] += delta
                    
                    # Renormalize
                    total_weight = sum(test_weights.values())
                    if total_weight > 0:
                        test_weights = {s: w / total_weight for s, w in test_weights.items()}
                    
                    # Check constraints
                    if self.constraints.is_valid_allocation(test_weights):
                        _, _, sharpe = self._calculate_portfolio_metrics(test_weights, market_data)
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_weights = test_weights.copy()
        
        return best_weights
    
    def _minimize_variance(
        self,
        cov_matrix: Dict[Symbol, Dict[Symbol, Decimal]],
        market_data: MarketData
    ) -> Dict[Symbol, Decimal]:
        """Minimize portfolio variance."""
        symbols = list(market_data.symbols)
        n = len(symbols)
        
        if n == 0:
            return {}
        
        # Start with equal weights
        equal_weight = Decimal('1') / Decimal(str(n))
        weights = {symbol: equal_weight for symbol in symbols}
        
        # Simple optimization: favor low volatility assets
        volatilities = {}
        for symbol in symbols:
            vol = market_data.volatilities.get(symbol, Decimal('0.2'))
            volatilities[symbol] = vol
        
        # Inverse volatility weighting
        inv_vol_sum = sum(Decimal('1') / vol for vol in volatilities.values())
        
        optimal_weights = {}
        for symbol in symbols:
            vol = volatilities[symbol]
            weight = (Decimal('1') / vol) / inv_vol_sum
            optimal_weights[symbol] = weight
        
        return optimal_weights
    
    def _maximize_return(
        self,
        expected_returns: Dict[Symbol, Decimal],
        cov_matrix: Dict[Symbol, Dict[Symbol, Decimal]],
        market_data: MarketData
    ) -> Dict[Symbol, Decimal]:
        """Maximize expected return subject to constraints."""
        symbols = list(market_data.symbols)
        
        if not symbols:
            return {}
        
        # Sort by expected return
        sorted_symbols = sorted(symbols, key=lambda s: expected_returns.get(s, Decimal('0')), reverse=True)
        
        # Allocate to highest return assets subject to constraints
        weights = {symbol: Decimal('0') for symbol in symbols}
        remaining_weight = Decimal('1')
        
        for symbol in sorted_symbols:
            max_allowed = min(self.constraints.max_concentration, remaining_weight)
            weights[symbol] = max_allowed
            remaining_weight -= max_allowed
            
            if remaining_weight <= Decimal('0.001'):
                break
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
        
        return weights
    
    def _create_failed_result(
        self,
        objective: OptimizationObjective,
        start_time: datetime,
        error_message: str
    ) -> OptimizationResult:
        """Create failed optimization result."""
        return OptimizationResult(
            objective=objective,
            optimization_time=start_time,
            success=False,
            optimal_weights={},
            expected_return=Decimal('0'),
            expected_volatility=Decimal('0'),
            expected_sharpe=Decimal('0'),
            optimization_score=Decimal('0'),
            diversification_ratio=Decimal('0'),
            concentration_index=Decimal('0'),
            portfolio_var=Decimal('0'),
            portfolio_cvar=Decimal('0'),
            maximum_drawdown_estimate=Decimal('0'),
            required_trades=[],
            total_turnover=Decimal('0'),
            transaction_costs=Money(Decimal('0')),
            iterations=0,
            convergence_achieved=False,
            error_message=error_message
        )


class RiskParityOptimizer(PortfolioOptimizer):
    """Risk parity optimization."""
    
    def optimize(
        self,
        market_data: MarketData,
        current_allocation: Dict[Symbol, Decimal] = None,
        objective: OptimizationObjective = OptimizationObjective.RISK_PARITY
    ) -> OptimizationResult:
        """Perform risk parity optimization."""
        with self._lock:
            start_time = datetime.now()
            
            try:
                symbols = list(market_data.symbols)
                n = len(symbols)
                
                if n == 0:
                    return self._create_failed_result(objective, start_time, "No symbols provided")
                
                # Initialize current allocation
                if current_allocation is None:
                    current_allocation = {symbol: Decimal('0') for symbol in symbols}
                
                # Risk parity: equal risk contribution from each asset
                optimal_weights = self._calculate_risk_parity_weights(market_data)
                
                # Calculate portfolio metrics
                exp_return, exp_vol, exp_sharpe = self._calculate_portfolio_metrics(optimal_weights, market_data)
                
                # Calculate additional metrics
                diversification_ratio = self._calculate_diversification_ratio(optimal_weights, market_data)
                concentration_index = self._calculate_concentration_index(optimal_weights)
                
                # Risk metrics
                portfolio_var = exp_vol * Decimal('1.65')
                portfolio_cvar = exp_vol * Decimal('2.33')
                max_dd_estimate = exp_vol * Decimal('2.5')
                
                # Trade calculation
                turnover = self._calculate_turnover(current_allocation, optimal_weights)
                portfolio_value = Money(Decimal('100000'))
                required_trades = self._generate_required_trades(current_allocation, optimal_weights, portfolio_value)
                transaction_costs = self._estimate_transaction_costs(required_trades, market_data)
                
                return OptimizationResult(
                    objective=objective,
                    optimization_time=start_time,
                    success=True,
                    optimal_weights=optimal_weights,
                    expected_return=exp_return,
                    expected_volatility=exp_vol,
                    expected_sharpe=exp_sharpe,
                    optimization_score=diversification_ratio,  # Use diversification as score
                    diversification_ratio=diversification_ratio,
                    concentration_index=concentration_index,
                    portfolio_var=portfolio_var,
                    portfolio_cvar=portfolio_cvar,
                    maximum_drawdown_estimate=max_dd_estimate,
                    required_trades=required_trades,
                    total_turnover=turnover,
                    transaction_costs=transaction_costs,
                    iterations=100,
                    convergence_achieved=True
                )
                
            except Exception as e:
                logger.error(f"Risk parity optimization failed: {e}")
                return self._create_failed_result(objective, start_time, str(e))
    
    def _calculate_risk_parity_weights(self, market_data: MarketData) -> Dict[Symbol, Decimal]:
        """Calculate risk parity weights."""
        symbols = list(market_data.symbols)
        
        # Get volatilities
        volatilities = {}
        for symbol in symbols:
            vol = market_data.volatilities.get(symbol, Decimal('0.2'))
            # Inverse volatility for risk parity
            volatilities[symbol] = Decimal('1') / vol if vol > 0 else Decimal('1')
        
        # Normalize to sum to 1
        total_inv_vol = sum(volatilities.values())
        
        weights = {}
        for symbol in symbols:
            weights[symbol] = volatilities[symbol] / total_inv_vol
        
        return weights
    
    def _create_failed_result(
        self,
        objective: OptimizationObjective,
        start_time: datetime,
        error_message: str
    ) -> OptimizationResult:
        """Create failed optimization result."""
        return OptimizationResult(
            objective=objective,
            optimization_time=start_time,
            success=False,
            optimal_weights={},
            expected_return=Decimal('0'),
            expected_volatility=Decimal('0'),
            expected_sharpe=Decimal('0'),
            optimization_score=Decimal('0'),
            diversification_ratio=Decimal('0'),
            concentration_index=Decimal('0'),
            portfolio_var=Decimal('0'),
            portfolio_cvar=Decimal('0'),
            maximum_drawdown_estimate=Decimal('0'),
            required_trades=[],
            total_turnover=Decimal('0'),
            transaction_costs=Money(Decimal('0')),
            iterations=0,
            convergence_achieved=False,
            error_message=error_message
        )


class PortfolioOptimizerFactory:
    """Factory for creating portfolio optimizers."""
    
    @staticmethod
    def create_optimizer(
        objective: OptimizationObjective,
        constraints: OptimizationConstraints = None
    ) -> PortfolioOptimizer:
        """Create optimizer based on objective."""
        if objective in [
            OptimizationObjective.MAXIMIZE_SHARPE,
            OptimizationObjective.MINIMIZE_RISK,
            OptimizationObjective.MAXIMIZE_RETURN,
            OptimizationObjective.MINIMIZE_VARIANCE
        ]:
            return MeanVarianceOptimizer(constraints)
        
        elif objective == OptimizationObjective.RISK_PARITY:
            return RiskParityOptimizer(constraints)
        
        else:
            # Default to mean-variance
            return MeanVarianceOptimizer(constraints)


class PortfolioOptimizationEngine:
    """Main portfolio optimization engine."""
    
    def __init__(
        self,
        portfolio_manager: PortfolioManager = None,
        performance_analyzer: PerformanceAnalyzer = None
    ):
        self.portfolio_manager = portfolio_manager
        self.performance_analyzer = performance_analyzer
        
        # Default constraints
        self.default_constraints = OptimizationConstraints()
        
        # Optimization history
        self.optimization_history: List[OptimizationResult] = []
        
        self._lock = threading.Lock()
    
    def optimize_portfolio(
        self,
        market_data: MarketData,
        objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_SHARPE,
        constraints: OptimizationConstraints = None,
        current_allocation: Dict[Symbol, Decimal] = None
    ) -> OptimizationResult:
        """Perform portfolio optimization."""
        with self._lock:
            # Use default constraints if none provided
            if constraints is None:
                constraints = self.default_constraints
            
            # Get current allocation if not provided
            if current_allocation is None and self.portfolio_manager:
                current_allocation = self._get_current_allocation()
            
            # Create optimizer
            optimizer = PortfolioOptimizerFactory.create_optimizer(objective, constraints)
            
            # Perform optimization
            result = optimizer.optimize(market_data, current_allocation, objective)
            
            # Store in history
            self.optimization_history.append(result)
            
            # Log result
            if result.success:
                logger.info(
                    f"Portfolio optimization successful: {objective.value}, "
                    f"Expected return: {result.expected_return:.4f}, "
                    f"Expected volatility: {result.expected_volatility:.4f}, "
                    f"Sharpe ratio: {result.expected_sharpe:.4f}"
                )
            else:
                logger.warning(f"Portfolio optimization failed: {result.error_message}")
            
            return result
    
    def run_multi_objective_optimization(
        self,
        market_data: MarketData,
        objectives: List[OptimizationObjective],
        constraints: OptimizationConstraints = None
    ) -> Dict[OptimizationObjective, OptimizationResult]:
        """Run optimization for multiple objectives."""
        results = {}
        
        for objective in objectives:
            result = self.optimize_portfolio(market_data, objective, constraints)
            results[objective] = result
        
        return results
    
    def generate_efficient_frontier(
        self,
        market_data: MarketData,
        num_points: int = 20,
        constraints: OptimizationConstraints = None
    ) -> List[OptimizationResult]:
        """Generate efficient frontier points."""
        efficient_frontier = []
        
        # Generate range of risk aversion parameters
        risk_aversions = [Decimal(str(i)) for i in range(1, num_points + 1)]
        
        for risk_aversion in risk_aversions:
            # Create mean-variance optimizer with specific risk aversion
            optimizer = MeanVarianceOptimizer(constraints)
            optimizer.risk_aversion = risk_aversion
            
            result = optimizer.optimize(
                market_data,
                objective=OptimizationObjective.MAXIMIZE_SHARPE
            )
            
            if result.success:
                efficient_frontier.append(result)
        
        return efficient_frontier
    
    def backtesting_optimization_strategy(
        self,
        historical_data: Dict[date, MarketData],
        rebalance_frequency_days: int = 30,
        objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_SHARPE
    ) -> Dict[str, Any]:
        """Backtest optimization strategy."""
        backtest_results = {
            'rebalance_dates': [],
            'portfolio_values': [],
            'optimization_results': [],
            'cumulative_return': Decimal('0'),
            'volatility': Decimal('0'),
            'sharpe_ratio': Decimal('0'),
            'max_drawdown': Decimal('0')
        }
        
        sorted_dates = sorted(historical_data.keys())
        portfolio_value = Money(Decimal('100000'))  # Start with $100k
        current_allocation = {}
        
        for i, rebalance_date in enumerate(sorted_dates):
            if i % rebalance_frequency_days == 0:  # Rebalance
                market_data = historical_data[rebalance_date]
                
                # Optimize portfolio
                result = self.optimize_portfolio(
                    market_data, objective, current_allocation=current_allocation
                )
                
                if result.success:
                    current_allocation = result.optimal_weights
                    backtest_results['rebalance_dates'].append(rebalance_date)
                    backtest_results['optimization_results'].append(result)
            
            # Calculate portfolio value (simplified)
            # In reality, would need to track actual price changes
            backtest_results['portfolio_values'].append((rebalance_date, portfolio_value))
        
        return backtest_results
    
    def _get_current_allocation(self) -> Dict[Symbol, Decimal]:
        """Get current portfolio allocation."""
        if not self.portfolio_manager:
            return {}
        
        positions = self.portfolio_manager.position_manager.get_all_positions()
        portfolio_value = self.portfolio_manager.portfolio.get_portfolio_value()
        
        if portfolio_value.amount == 0:
            return {}
        
        allocation = {}
        for symbol, position in positions.items():
            weight = position.market_value.amount / portfolio_value.amount
            allocation[symbol] = weight
        
        return allocation
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization engine statistics."""
        with self._lock:
            if not self.optimization_history:
                return {
                    'total_optimizations': 0,
                    'successful_optimizations': 0,
                    'success_rate': 0,
                    'average_expected_return': 0,
                    'average_expected_volatility': 0,
                    'average_sharpe_ratio': 0
                }
            
            successful_results = [r for r in self.optimization_history if r.success]
            success_rate = len(successful_results) / len(self.optimization_history)
            
            if successful_results:
                avg_return = sum(r.expected_return for r in successful_results) / len(successful_results)
                avg_volatility = sum(r.expected_volatility for r in successful_results) / len(successful_results)
                avg_sharpe = sum(r.expected_sharpe for r in successful_results) / len(successful_results)
            else:
                avg_return = avg_volatility = avg_sharpe = Decimal('0')
            
            return {
                'total_optimizations': len(self.optimization_history),
                'successful_optimizations': len(successful_results),
                'success_rate': float(success_rate),
                'average_expected_return': float(avg_return),
                'average_expected_volatility': float(avg_volatility),
                'average_sharpe_ratio': float(avg_sharpe),
                'latest_optimization': self.optimization_history[-1].to_dict() if self.optimization_history else None
            }


# Global optimizer instance
_portfolio_optimization_engine = None


def get_portfolio_optimization_engine(
    portfolio_manager: PortfolioManager = None,
    performance_analyzer: PerformanceAnalyzer = None
) -> PortfolioOptimizationEngine:
    """Get global portfolio optimization engine."""
    global _portfolio_optimization_engine
    if _portfolio_optimization_engine is None:
        _portfolio_optimization_engine = PortfolioOptimizationEngine(
            portfolio_manager, performance_analyzer
        )
    return _portfolio_optimization_engine