"""
Performance analytics and metrics calculation system.
"""

from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from enum import Enum
import math
import statistics
from collections import defaultdict
import threading

from domain.value_objects import Symbol, Price, Quantity, Money
from .portfolio_manager import PortfolioManager, PortfolioSnapshot, PortfolioHistory
from .pnl_calculator import PnLCalculator, PnLSnapshot, RealizedPnL
from .position_manager import PositionManager, Position
from utils.logging import get_logger

logger = get_logger(__name__)


class PerformancePeriod(Enum):
    """Performance calculation periods."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    INCEPTION = "inception"


class RiskMetricType(Enum):
    """Types of risk metrics."""
    VOLATILITY = "volatility"
    VAR = "var"
    CVAR = "cvar"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    BETA = "beta"
    ALPHA = "alpha"
    TRACKING_ERROR = "tracking_error"
    INFORMATION_RATIO = "information_ratio"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    period: PerformancePeriod
    start_date: date
    end_date: date
    calculation_time: datetime
    
    # Return metrics
    total_return: Decimal
    annualized_return: Decimal
    cumulative_return: Decimal
    average_daily_return: Decimal
    geometric_mean_return: Decimal
    
    # Risk metrics
    volatility: Decimal
    annualized_volatility: Decimal
    downside_volatility: Decimal
    max_drawdown: Decimal
    max_drawdown_duration: int  # days
    
    # Risk-adjusted returns
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    
    # Win/Loss metrics
    win_rate: Decimal
    profit_factor: Decimal
    largest_win: Money
    largest_loss: Money
    average_win: Money
    average_loss: Money
    
    # Portfolio metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    portfolio_value: Money
    cash_percentage: Decimal
    
    # Value at Risk
    var_95: Money
    var_99: Money
    
    # Additional metrics
    skewness: Optional[Decimal] = None
    kurtosis: Optional[Decimal] = None
    beta: Optional[Decimal] = None
    alpha: Optional[Decimal] = None
    tracking_error: Optional[Decimal] = None
    information_ratio: Optional[Decimal] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'period': self.period.value,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'calculation_time': self.calculation_time.isoformat(),
            'total_return': float(self.total_return),
            'annualized_return': float(self.annualized_return),
            'cumulative_return': float(self.cumulative_return),
            'average_daily_return': float(self.average_daily_return),
            'geometric_mean_return': float(self.geometric_mean_return),
            'volatility': float(self.volatility),
            'annualized_volatility': float(self.annualized_volatility),
            'downside_volatility': float(self.downside_volatility),
            'max_drawdown': float(self.max_drawdown),
            'max_drawdown_duration': self.max_drawdown_duration,
            'sharpe_ratio': float(self.sharpe_ratio),
            'sortino_ratio': float(self.sortino_ratio),
            'calmar_ratio': float(self.calmar_ratio),
            'win_rate': float(self.win_rate),
            'profit_factor': float(self.profit_factor),
            'largest_win': float(self.largest_win.amount),
            'largest_loss': float(self.largest_loss.amount),
            'average_win': float(self.average_win.amount),
            'average_loss': float(self.average_loss.amount),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'portfolio_value': float(self.portfolio_value.amount),
            'cash_percentage': float(self.cash_percentage),
            'var_95': float(self.var_95.amount),
            'var_99': float(self.var_99.amount),
            'skewness': float(self.skewness) if self.skewness else None,
            'kurtosis': float(self.kurtosis) if self.kurtosis else None,
            'beta': float(self.beta) if self.beta else None,
            'alpha': float(self.alpha) if self.alpha else None,
            'tracking_error': float(self.tracking_error) if self.tracking_error else None,
            'information_ratio': float(self.information_ratio) if self.information_ratio else None
        }


@dataclass
class BenchmarkComparison:
    """Comparison against a benchmark."""
    benchmark_symbol: Symbol
    period: PerformancePeriod
    start_date: date
    end_date: date
    
    # Portfolio metrics
    portfolio_return: Decimal
    portfolio_volatility: Decimal
    portfolio_sharpe: Decimal
    
    # Benchmark metrics
    benchmark_return: Decimal
    benchmark_volatility: Decimal
    benchmark_sharpe: Decimal
    
    # Relative metrics
    excess_return: Decimal
    tracking_error: Decimal
    information_ratio: Decimal
    beta: Decimal
    alpha: Decimal
    
    # Performance attribution
    up_capture_ratio: Decimal
    down_capture_ratio: Decimal
    correlation: Decimal
    
    def get_relative_performance(self) -> str:
        """Get relative performance description."""
        if self.excess_return > Decimal('0.05'):
            return "Significantly outperforming"
        elif self.excess_return > Decimal('0.01'):
            return "Outperforming"
        elif self.excess_return > Decimal('-0.01'):
            return "Matching"
        elif self.excess_return > Decimal('-0.05'):
            return "Underperforming"
        else:
            return "Significantly underperforming"


@dataclass
class RiskMetrics:
    """Risk-specific metrics."""
    calculation_date: date
    
    # Volatility measures
    daily_volatility: Decimal
    weekly_volatility: Decimal
    monthly_volatility: Decimal
    annualized_volatility: Decimal
    
    # Downside risk
    downside_deviation: Decimal
    upside_deviation: Decimal
    downside_volatility: Decimal
    
    # Value at Risk
    var_1d_95: Money
    var_1d_99: Money
    var_1w_95: Money
    var_1w_99: Money
    
    # Conditional Value at Risk
    cvar_1d_95: Money
    cvar_1d_99: Money
    
    # Drawdown metrics
    current_drawdown: Decimal
    max_drawdown: Decimal
    max_drawdown_duration: int
    average_drawdown: Decimal
    
    # Risk-adjusted metrics
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    sterling_ratio: Decimal
    
    # Distribution measures
    skewness: Decimal
    kurtosis: Decimal
    jarque_bera_stat: Optional[Decimal] = None
    jarque_bera_pvalue: Optional[Decimal] = None


@dataclass
class ReturnsAnalysis:
    """Detailed returns analysis."""
    period: PerformancePeriod
    
    # Returns series
    daily_returns: List[Tuple[date, Decimal]]
    weekly_returns: List[Tuple[date, Decimal]]
    monthly_returns: List[Tuple[date, Decimal]]
    
    # Statistics
    mean_return: Decimal
    median_return: Decimal
    std_return: Decimal
    min_return: Decimal
    max_return: Decimal
    
    # Percentiles
    percentile_5: Decimal
    percentile_25: Decimal
    percentile_75: Decimal
    percentile_95: Decimal
    
    # Streaks
    longest_winning_streak: int
    longest_losing_streak: int
    current_streak: int
    current_streak_type: str  # "winning" or "losing"
    
    # Frequency analysis
    positive_days: int
    negative_days: int
    flat_days: int
    
    def get_return_distribution(self) -> Dict[str, int]:
        """Get return distribution buckets."""
        buckets = {
            "< -5%": 0, "-5% to -2%": 0, "-2% to -1%": 0, "-1% to 0%": 0,
            "0% to 1%": 0, "1% to 2%": 0, "2% to 5%": 0, "> 5%": 0
        }
        
        for _, return_pct in self.daily_returns:
            if return_pct < Decimal('-0.05'):
                buckets["< -5%"] += 1
            elif return_pct < Decimal('-0.02'):
                buckets["-5% to -2%"] += 1
            elif return_pct < Decimal('-0.01'):
                buckets["-2% to -1%"] += 1
            elif return_pct < Decimal('0'):
                buckets["-1% to 0%"] += 1
            elif return_pct < Decimal('0.01'):
                buckets["0% to 1%"] += 1
            elif return_pct < Decimal('0.02'):
                buckets["1% to 2%"] += 1
            elif return_pct < Decimal('0.05'):
                buckets["2% to 5%"] += 1
            else:
                buckets["> 5%"] += 1
        
        return buckets


class SharpeCalculator:
    """Advanced Sharpe ratio calculator."""
    
    def __init__(self, risk_free_rate: Decimal = Decimal('0.02')):
        self.risk_free_rate = risk_free_rate
    
    def calculate_sharpe_ratio(
        self,
        returns: List[Decimal],
        period: PerformancePeriod = PerformancePeriod.DAILY
    ) -> Decimal:
        """Calculate Sharpe ratio for given returns."""
        if not returns or len(returns) < 2:
            return Decimal('0')
        
        # Adjust risk-free rate for period
        if period == PerformancePeriod.DAILY:
            risk_free_daily = self.risk_free_rate / Decimal('252')
        elif period == PerformancePeriod.WEEKLY:
            risk_free_daily = self.risk_free_rate / Decimal('52')
        elif period == PerformancePeriod.MONTHLY:
            risk_free_daily = self.risk_free_rate / Decimal('12')
        else:
            risk_free_daily = self.risk_free_rate
        
        # Calculate excess returns
        excess_returns = [r - risk_free_daily for r in returns]
        
        if len(excess_returns) < 2:
            return Decimal('0')
        
        # Calculate mean and standard deviation
        mean_excess = sum(excess_returns) / len(excess_returns)
        variance = sum((r - mean_excess) ** 2 for r in excess_returns) / (len(excess_returns) - 1)
        
        if variance <= 0:
            return Decimal('0')
        
        std_dev = variance ** Decimal('0.5')
        
        # Annualize if needed
        if period == PerformancePeriod.DAILY:
            sharpe = mean_excess / std_dev * Decimal('252') ** Decimal('0.5')
        elif period == PerformancePeriod.WEEKLY:
            sharpe = mean_excess / std_dev * Decimal('52') ** Decimal('0.5')
        elif period == PerformancePeriod.MONTHLY:
            sharpe = mean_excess / std_dev * Decimal('12') ** Decimal('0.5')
        else:
            sharpe = mean_excess / std_dev
        
        return sharpe
    
    def calculate_sortino_ratio(
        self,
        returns: List[Decimal],
        target_return: Decimal = Decimal('0'),
        period: PerformancePeriod = PerformancePeriod.DAILY
    ) -> Decimal:
        """Calculate Sortino ratio (downside deviation)."""
        if not returns or len(returns) < 2:
            return Decimal('0')
        
        # Calculate excess returns over target
        excess_returns = [r - target_return for r in returns]
        mean_excess = sum(excess_returns) / len(excess_returns)
        
        # Calculate downside deviation
        downside_returns = [r for r in excess_returns if r < 0]
        
        if len(downside_returns) < 2:
            return Decimal('0')
        
        downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
        downside_deviation = downside_variance ** Decimal('0.5')
        
        if downside_deviation == 0:
            return Decimal('0')
        
        # Annualize if needed
        if period == PerformancePeriod.DAILY:
            sortino = mean_excess / downside_deviation * Decimal('252') ** Decimal('0.5')
        elif period == PerformancePeriod.WEEKLY:
            sortino = mean_excess / downside_deviation * Decimal('52') ** Decimal('0.5')
        elif period == PerformancePeriod.MONTHLY:
            sortino = mean_excess / downside_deviation * Decimal('12') ** Decimal('0.5')
        else:
            sortino = mean_excess / downside_deviation
        
        return sortino


class DrawdownAnalyzer:
    """Drawdown analysis and calculation."""
    
    def __init__(self):
        self._lock = threading.Lock()
    
    def calculate_drawdowns(
        self,
        value_series: List[Tuple[datetime, Decimal]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive drawdown metrics."""
        with self._lock:
            if len(value_series) < 2:
                return self._empty_drawdown_metrics()
            
            # Sort by timestamp
            sorted_series = sorted(value_series, key=lambda x: x[0])
            
            # Calculate running maximum and drawdowns
            running_max = sorted_series[0][1]
            max_drawdown = Decimal('0')
            current_drawdown = Decimal('0')
            
            drawdown_periods = []
            current_drawdown_start = None
            current_drawdown_peak_value = running_max
            
            drawdown_values = []
            
            for timestamp, value in sorted_series:
                # Update running maximum
                if value > running_max:
                    running_max = value
                    # End current drawdown period if any
                    if current_drawdown_start:
                        drawdown_periods.append({
                            'start': current_drawdown_start,
                            'end': timestamp,
                            'duration_days': (timestamp - current_drawdown_start).days,
                            'peak_value': current_drawdown_peak_value,
                            'trough_value': min(v for _, v in sorted_series 
                                             if current_drawdown_start <= _ <= timestamp),
                            'recovery_value': value,
                            'max_drawdown_pct': current_drawdown
                        })
                        current_drawdown_start = None
                
                # Calculate current drawdown
                current_drawdown = (value - running_max) / running_max if running_max > 0 else Decimal('0')
                drawdown_values.append((timestamp, current_drawdown))
                
                # Track maximum drawdown
                if current_drawdown < max_drawdown:
                    max_drawdown = current_drawdown
                
                # Start new drawdown period
                if current_drawdown < 0 and current_drawdown_start is None:
                    current_drawdown_start = timestamp
                    current_drawdown_peak_value = running_max
            
            # Handle ongoing drawdown
            if current_drawdown_start and current_drawdown < 0:
                drawdown_periods.append({
                    'start': current_drawdown_start,
                    'end': sorted_series[-1][0],
                    'duration_days': (sorted_series[-1][0] - current_drawdown_start).days,
                    'peak_value': current_drawdown_peak_value,
                    'trough_value': sorted_series[-1][1],
                    'recovery_value': None,  # Ongoing
                    'max_drawdown_pct': current_drawdown
                })
            
            # Calculate statistics
            if drawdown_periods:
                avg_drawdown = sum(abs(period['max_drawdown_pct']) for period in drawdown_periods) / len(drawdown_periods)
                max_duration = max(period['duration_days'] for period in drawdown_periods)
                avg_duration = sum(period['duration_days'] for period in drawdown_periods) / len(drawdown_periods)
            else:
                avg_drawdown = Decimal('0')
                max_duration = 0
                avg_duration = Decimal('0')
            
            return {
                'max_drawdown_pct': abs(max_drawdown),
                'current_drawdown_pct': abs(current_drawdown),
                'max_drawdown_duration_days': max_duration,
                'average_drawdown_pct': avg_drawdown,
                'average_drawdown_duration_days': avg_duration,
                'total_drawdown_periods': len(drawdown_periods),
                'drawdown_periods': drawdown_periods,
                'drawdown_series': drawdown_values,
                'time_in_drawdown_pct': self._calculate_time_in_drawdown(drawdown_values) if drawdown_values else Decimal('0')
            }
    
    def _empty_drawdown_metrics(self) -> Dict[str, Any]:
        """Return empty drawdown metrics."""
        return {
            'max_drawdown_pct': Decimal('0'),
            'current_drawdown_pct': Decimal('0'),
            'max_drawdown_duration_days': 0,
            'average_drawdown_pct': Decimal('0'),
            'average_drawdown_duration_days': Decimal('0'),
            'total_drawdown_periods': 0,
            'drawdown_periods': [],
            'drawdown_series': [],
            'time_in_drawdown_pct': Decimal('0')
        }
    
    def _calculate_time_in_drawdown(self, drawdown_series: List[Tuple[datetime, Decimal]]) -> Decimal:
        """Calculate percentage of time spent in drawdown."""
        if len(drawdown_series) < 2:
            return Decimal('0')
        
        total_time = (drawdown_series[-1][0] - drawdown_series[0][0]).total_seconds()
        if total_time <= 0:
            return Decimal('0')
        
        drawdown_time = 0
        for i in range(len(drawdown_series) - 1):
            if drawdown_series[i][1] < 0:
                drawdown_time += (drawdown_series[i+1][0] - drawdown_series[i][0]).total_seconds()
        
        return Decimal(str(drawdown_time / total_time))


class PerformanceAnalyzer:
    """Main performance analysis engine."""
    
    def __init__(
        self,
        portfolio_manager: PortfolioManager = None,
        pnl_calculator: PnLCalculator = None,
        position_manager: PositionManager = None
    ):
        self.portfolio_manager = portfolio_manager
        self.pnl_calculator = pnl_calculator
        self.position_manager = position_manager
        
        self.sharpe_calculator = SharpeCalculator()
        self.drawdown_analyzer = DrawdownAnalyzer()
        
        # Cache for performance
        self._metrics_cache: Dict[str, PerformanceMetrics] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_duration = timedelta(minutes=5)
        
        self._lock = threading.Lock()
    
    def calculate_performance_metrics(
        self,
        period: PerformancePeriod = PerformancePeriod.INCEPTION,
        start_date: date = None,
        end_date: date = None
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        with self._lock:
            # Check cache
            cache_key = f"{period.value}_{start_date}_{end_date}"
            if (cache_key in self._metrics_cache and 
                cache_key in self._cache_expiry and
                datetime.now() < self._cache_expiry[cache_key]):
                return self._metrics_cache[cache_key]
            
            # Get portfolio history
            if not self.portfolio_manager:
                raise ValueError("Portfolio manager required for performance analysis")
            
            portfolio_history = self.portfolio_manager.get_portfolio_history()
            
            # Determine date range
            if not start_date or not end_date:
                start_date, end_date = self._determine_date_range(portfolio_history, period)
            
            # Filter snapshots for date range
            relevant_snapshots = [
                snap for snap in portfolio_history.snapshots
                if start_date <= snap.timestamp.date() <= end_date
            ]
            
            if len(relevant_snapshots) < 2:
                return self._create_empty_metrics(period, start_date, end_date)
            
            # Calculate return metrics
            return_metrics = self._calculate_return_metrics(relevant_snapshots, start_date, end_date)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(relevant_snapshots)
            
            # Calculate trade metrics
            trade_metrics = self._calculate_trade_metrics(start_date, end_date)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(relevant_snapshots[-1])
            
            # Combine all metrics
            metrics = PerformanceMetrics(
                period=period,
                start_date=start_date,
                end_date=end_date,
                calculation_time=datetime.now(),
                **return_metrics,
                **risk_metrics,
                **trade_metrics,
                **portfolio_metrics
            )
            
            # Cache result
            self._metrics_cache[cache_key] = metrics
            self._cache_expiry[cache_key] = datetime.now() + self._cache_duration
            
            return metrics
    
    def calculate_benchmark_comparison(
        self,
        benchmark_symbol: Symbol,
        benchmark_data: List[Tuple[date, Decimal]],
        period: PerformancePeriod = PerformancePeriod.INCEPTION
    ) -> BenchmarkComparison:
        """Calculate performance comparison against benchmark."""
        # Get portfolio performance
        portfolio_metrics = self.calculate_performance_metrics(period)
        
        # Calculate benchmark metrics (simplified - would need actual benchmark data)
        benchmark_returns = [price_change for _, price_change in benchmark_data]
        
        if not benchmark_returns:
            raise ValueError("No benchmark data provided")
        
        # Calculate benchmark statistics
        benchmark_return = sum(benchmark_returns)
        benchmark_volatility = Decimal(str(statistics.stdev(benchmark_returns))) if len(benchmark_returns) > 1 else Decimal('0')
        benchmark_sharpe = self.sharpe_calculator.calculate_sharpe_ratio(benchmark_returns)
        
        # Calculate relative metrics
        excess_return = portfolio_metrics.total_return - benchmark_return
        
        # Calculate beta and alpha (simplified)
        portfolio_returns = self._get_portfolio_returns(period)
        beta, alpha = self._calculate_beta_alpha(portfolio_returns, benchmark_returns)
        
        # Calculate tracking error
        excess_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns[:len(portfolio_returns)])]
        tracking_error = Decimal(str(statistics.stdev(excess_returns))) if len(excess_returns) > 1 else Decimal('0')
        
        # Information ratio
        information_ratio = excess_return / tracking_error if tracking_error > 0 else Decimal('0')
        
        # Calculate capture ratios
        up_capture, down_capture = self._calculate_capture_ratios(portfolio_returns, benchmark_returns)
        
        # Calculate correlation
        correlation = self._calculate_correlation(portfolio_returns, benchmark_returns)
        
        return BenchmarkComparison(
            benchmark_symbol=benchmark_symbol,
            period=period,
            start_date=portfolio_metrics.start_date,
            end_date=portfolio_metrics.end_date,
            portfolio_return=portfolio_metrics.total_return,
            portfolio_volatility=portfolio_metrics.volatility,
            portfolio_sharpe=portfolio_metrics.sharpe_ratio,
            benchmark_return=benchmark_return,
            benchmark_volatility=benchmark_volatility,
            benchmark_sharpe=benchmark_sharpe,
            excess_return=excess_return,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            beta=beta,
            alpha=alpha,
            up_capture_ratio=up_capture,
            down_capture_ratio=down_capture,
            correlation=correlation
        )
    
    def calculate_risk_metrics(
        self,
        period: PerformancePeriod = PerformancePeriod.INCEPTION
    ) -> RiskMetrics:
        """Calculate detailed risk metrics."""
        # Get portfolio data
        portfolio_history = self.portfolio_manager.get_portfolio_history()
        snapshots = portfolio_history.snapshots
        
        if len(snapshots) < 2:
            return self._create_empty_risk_metrics()
        
        # Get returns
        returns = self._extract_returns_from_snapshots(snapshots)
        values = [(snap.timestamp, snap.portfolio.get_portfolio_value().amount) for snap in snapshots]
        
        # Calculate volatility measures
        daily_vol = Decimal(str(statistics.stdev(returns))) if len(returns) > 1 else Decimal('0')
        weekly_vol = daily_vol * Decimal('7') ** Decimal('0.5')
        monthly_vol = daily_vol * Decimal('30') ** Decimal('0.5')
        annual_vol = daily_vol * Decimal('252') ** Decimal('0.5')
        
        # Calculate downside metrics
        negative_returns = [r for r in returns if r < 0]
        downside_dev = Decimal(str(statistics.stdev(negative_returns))) if len(negative_returns) > 1 else Decimal('0')
        positive_returns = [r for r in returns if r > 0]
        upside_dev = Decimal(str(statistics.stdev(positive_returns))) if len(positive_returns) > 1 else Decimal('0')
        
        # Calculate VaR
        current_value = snapshots[-1].portfolio.get_portfolio_value()
        var_metrics = self._calculate_var(returns, current_value)
        
        # Calculate drawdown metrics
        drawdown_metrics = self.drawdown_analyzer.calculate_drawdowns(values)
        
        # Calculate risk-adjusted ratios
        sharpe = self.sharpe_calculator.calculate_sharpe_ratio(returns)
        sortino = self.sharpe_calculator.calculate_sortino_ratio(returns)
        calmar = (sum(returns) / len(returns) * Decimal('252')) / drawdown_metrics['max_drawdown_pct'] if drawdown_metrics['max_drawdown_pct'] > 0 else Decimal('0')
        sterling = calmar  # Simplified
        
        # Calculate distribution measures
        skewness, kurtosis = self._calculate_distribution_measures(returns)
        
        return RiskMetrics(
            calculation_date=date.today(),
            daily_volatility=daily_vol,
            weekly_volatility=weekly_vol,
            monthly_volatility=monthly_vol,
            annualized_volatility=annual_vol,
            downside_deviation=downside_dev,
            upside_deviation=upside_dev,
            downside_volatility=downside_dev * Decimal('252') ** Decimal('0.5'),
            var_1d_95=var_metrics['var_1d_95'],
            var_1d_99=var_metrics['var_1d_99'],
            var_1w_95=var_metrics['var_1w_95'],
            var_1w_99=var_metrics['var_1w_99'],
            cvar_1d_95=var_metrics['cvar_1d_95'],
            cvar_1d_99=var_metrics['cvar_1d_99'],
            current_drawdown=drawdown_metrics['current_drawdown_pct'],
            max_drawdown=drawdown_metrics['max_drawdown_pct'],
            max_drawdown_duration=drawdown_metrics['max_drawdown_duration_days'],
            average_drawdown=drawdown_metrics['average_drawdown_pct'],
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            sterling_ratio=sterling,
            skewness=skewness,
            kurtosis=kurtosis
        )
    
    def calculate_returns_analysis(
        self,
        period: PerformancePeriod = PerformancePeriod.INCEPTION
    ) -> ReturnsAnalysis:
        """Calculate detailed returns analysis."""
        # Get portfolio history
        portfolio_history = self.portfolio_manager.get_portfolio_history()
        snapshots = portfolio_history.snapshots
        
        if len(snapshots) < 2:
            return self._create_empty_returns_analysis(period)
        
        # Extract returns
        daily_returns = []
        for i in range(1, len(snapshots)):
            prev_value = snapshots[i-1].portfolio.get_portfolio_value().amount
            curr_value = snapshots[i].portfolio.get_portfolio_value().amount
            
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                daily_returns.append((snapshots[i].timestamp.date(), daily_return))
        
        # Aggregate to weekly and monthly
        weekly_returns = self._aggregate_returns(daily_returns, 'weekly')
        monthly_returns = self._aggregate_returns(daily_returns, 'monthly')
        
        # Calculate statistics
        returns_values = [r for _, r in daily_returns]
        
        if not returns_values:
            return self._create_empty_returns_analysis(period)
        
        mean_return = sum(returns_values) / len(returns_values)
        median_return = Decimal(str(statistics.median(returns_values)))
        std_return = Decimal(str(statistics.stdev(returns_values))) if len(returns_values) > 1 else Decimal('0')
        min_return = min(returns_values)
        max_return = max(returns_values)
        
        # Calculate percentiles
        sorted_returns = sorted(returns_values)
        n = len(sorted_returns)
        percentile_5 = sorted_returns[int(0.05 * n)] if n > 0 else Decimal('0')
        percentile_25 = sorted_returns[int(0.25 * n)] if n > 0 else Decimal('0')
        percentile_75 = sorted_returns[int(0.75 * n)] if n > 0 else Decimal('0')
        percentile_95 = sorted_returns[int(0.95 * n)] if n > 0 else Decimal('0')
        
        # Calculate streaks
        streak_metrics = self._calculate_return_streaks(daily_returns)
        
        # Calculate frequency
        positive_days = sum(1 for r in returns_values if r > 0)
        negative_days = sum(1 for r in returns_values if r < 0)
        flat_days = sum(1 for r in returns_values if r == 0)
        
        return ReturnsAnalysis(
            period=period,
            daily_returns=daily_returns,
            weekly_returns=weekly_returns,
            monthly_returns=monthly_returns,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            min_return=min_return,
            max_return=max_return,
            percentile_5=percentile_5,
            percentile_25=percentile_25,
            percentile_75=percentile_75,
            percentile_95=percentile_95,
            longest_winning_streak=streak_metrics['longest_winning_streak'],
            longest_losing_streak=streak_metrics['longest_losing_streak'],
            current_streak=streak_metrics['current_streak'],
            current_streak_type=streak_metrics['current_streak_type'],
            positive_days=positive_days,
            negative_days=negative_days,
            flat_days=flat_days
        )
    
    def generate_performance_report(
        self,
        period: PerformancePeriod = PerformancePeriod.INCEPTION,
        include_benchmark: bool = False,
        benchmark_symbol: Symbol = None,
        benchmark_data: List[Tuple[date, Decimal]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'performance_metrics': self.calculate_performance_metrics(period).to_dict(),
            'risk_metrics': self.calculate_risk_metrics(period).__dict__,
            'returns_analysis': self.calculate_returns_analysis(period).__dict__,
            'generation_time': datetime.now().isoformat(),
            'period': period.value
        }
        
        # Add benchmark comparison if requested
        if include_benchmark and benchmark_symbol and benchmark_data:
            try:
                benchmark_comparison = self.calculate_benchmark_comparison(
                    benchmark_symbol, benchmark_data, period
                )
                report['benchmark_comparison'] = benchmark_comparison.__dict__
            except Exception as e:
                logger.warning(f"Failed to calculate benchmark comparison: {e}")
                report['benchmark_comparison'] = None
        
        return report
    
    # Helper methods
    
    def _determine_date_range(
        self,
        history: PortfolioHistory,
        period: PerformancePeriod
    ) -> Tuple[date, date]:
        """Determine date range for performance calculation."""
        if not history.snapshots:
            today = date.today()
            return today, today
        
        end_date = date.today()
        
        if period == PerformancePeriod.DAILY:
            start_date = end_date - timedelta(days=1)
        elif period == PerformancePeriod.WEEKLY:
            start_date = end_date - timedelta(weeks=1)
        elif period == PerformancePeriod.MONTHLY:
            start_date = end_date - timedelta(days=30)
        elif period == PerformancePeriod.QUARTERLY:
            start_date = end_date - timedelta(days=90)
        elif period == PerformancePeriod.YEARLY:
            start_date = end_date - timedelta(days=365)
        else:  # INCEPTION
            start_date = history.snapshots[0].timestamp.date()
        
        return start_date, end_date
    
    def _calculate_return_metrics(
        self,
        snapshots: List[PortfolioSnapshot],
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """Calculate return-related metrics."""
        if len(snapshots) < 2:
            return {
                'total_return': Decimal('0'),
                'annualized_return': Decimal('0'),
                'cumulative_return': Decimal('0'),
                'average_daily_return': Decimal('0'),
                'geometric_mean_return': Decimal('0')
            }
        
        # Calculate total return
        initial_value = snapshots[0].portfolio.get_portfolio_value().amount
        final_value = snapshots[-1].portfolio.get_portfolio_value().amount
        
        if initial_value > 0:
            total_return = (final_value - initial_value) / initial_value
        else:
            total_return = Decimal('0')
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(snapshots)):
            prev_value = snapshots[i-1].portfolio.get_portfolio_value().amount
            curr_value = snapshots[i].portfolio.get_portfolio_value().amount
            
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                daily_returns.append(daily_return)
        
        # Calculate metrics
        if daily_returns:
            average_daily_return = sum(daily_returns) / len(daily_returns)
            
            # Annualized return
            days = (end_date - start_date).days
            if days > 0:
                annualized_return = (1 + total_return) ** (Decimal('365') / Decimal(str(days))) - 1
            else:
                annualized_return = total_return
            
            # Geometric mean return
            if all(1 + r > 0 for r in daily_returns):
                product = Decimal('1')
                for r in daily_returns:
                    product *= (1 + r)
                geometric_mean_return = product ** (Decimal('1') / Decimal(str(len(daily_returns)))) - 1
            else:
                geometric_mean_return = average_daily_return
        else:
            average_daily_return = Decimal('0')
            annualized_return = Decimal('0')
            geometric_mean_return = Decimal('0')
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cumulative_return': total_return,
            'average_daily_return': average_daily_return,
            'geometric_mean_return': geometric_mean_return
        }
    
    def _calculate_risk_metrics(self, snapshots: List[PortfolioSnapshot]) -> Dict[str, Any]:
        """Calculate risk-related metrics."""
        returns = self._extract_returns_from_snapshots(snapshots)
        
        if len(returns) < 2:
            return {
                'volatility': Decimal('0'),
                'annualized_volatility': Decimal('0'),
                'downside_volatility': Decimal('0'),
                'max_drawdown': Decimal('0'),
                'max_drawdown_duration': 0,
                'sharpe_ratio': Decimal('0'),
                'sortino_ratio': Decimal('0'),
                'calmar_ratio': Decimal('0'),
                'var_95': Money(Decimal('0')),
                'var_99': Money(Decimal('0'))
            }
        
        # Calculate volatility
        volatility = Decimal(str(statistics.stdev(returns)))
        annualized_volatility = volatility * Decimal('252') ** Decimal('0.5')
        
        # Calculate downside volatility
        negative_returns = [r for r in returns if r < 0]
        downside_volatility = (Decimal(str(statistics.stdev(negative_returns))) * Decimal('252') ** Decimal('0.5') 
                             if len(negative_returns) > 1 else Decimal('0'))
        
        # Calculate drawdown
        values = [(snap.timestamp, snap.portfolio.get_portfolio_value().amount) for snap in snapshots]
        drawdown_metrics = self.drawdown_analyzer.calculate_drawdowns(values)
        
        # Calculate ratios
        sharpe_ratio = self.sharpe_calculator.calculate_sharpe_ratio(returns)
        sortino_ratio = self.sharpe_calculator.calculate_sortino_ratio(returns)
        
        max_dd = drawdown_metrics['max_drawdown_pct']
        calmar_ratio = (sum(returns) / len(returns) * Decimal('252')) / max_dd if max_dd > 0 else Decimal('0')
        
        # Calculate VaR
        current_value = snapshots[-1].portfolio.get_portfolio_value()
        var_metrics = self._calculate_var(returns, current_value)
        
        return {
            'volatility': volatility,
            'annualized_volatility': annualized_volatility,
            'downside_volatility': downside_volatility,
            'max_drawdown': max_dd,
            'max_drawdown_duration': drawdown_metrics['max_drawdown_duration_days'],
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'var_95': var_metrics['var_1d_95'],
            'var_99': var_metrics['var_1d_99']
        }
    
    def _calculate_trade_metrics(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Calculate trade-related metrics."""
        if not self.pnl_calculator:
            return self._empty_trade_metrics()
        
        # Get realized trades in period
        realized_trades = self.pnl_calculator.calculate_realized_pnl(start_date, end_date)
        
        if not realized_trades:
            return self._empty_trade_metrics()
        
        # Calculate metrics
        winning_trades = [trade for trade in realized_trades if trade.realized_amount.amount > 0]
        losing_trades = [trade for trade in realized_trades if trade.realized_amount.amount < 0]
        
        total_trades = len(realized_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else Decimal('0')
        
        # Calculate profit factor
        gross_profit = sum(trade.realized_amount.amount for trade in winning_trades)
        gross_loss = abs(sum(trade.realized_amount.amount for trade in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal('0')
        
        # Find largest win/loss
        largest_win = Money(max((trade.realized_amount.amount for trade in winning_trades), default=Decimal('0')))
        largest_loss = Money(min((trade.realized_amount.amount for trade in losing_trades), default=Decimal('0')))
        
        # Calculate averages
        average_win = Money(gross_profit / len(winning_trades) if winning_trades else Decimal('0'))
        average_loss = Money(gross_loss / len(losing_trades) if losing_trades else Decimal('0'))
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'average_win': average_win,
            'average_loss': average_loss
        }
    
    def _calculate_portfolio_metrics(self, latest_snapshot: PortfolioSnapshot) -> Dict[str, Any]:
        """Calculate current portfolio metrics."""
        return {
            'portfolio_value': latest_snapshot.portfolio.get_portfolio_value(),
            'cash_percentage': latest_snapshot.portfolio.get_cash_percentage()
        }
    
    def _extract_returns_from_snapshots(self, snapshots: List[PortfolioSnapshot]) -> List[Decimal]:
        """Extract returns from portfolio snapshots."""
        returns = []
        for i in range(1, len(snapshots)):
            prev_value = snapshots[i-1].portfolio.get_portfolio_value().amount
            curr_value = snapshots[i].portfolio.get_portfolio_value().amount
            
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)
        
        return returns
    
    def _calculate_var(self, returns: List[Decimal], current_value: Money) -> Dict[str, Money]:
        """Calculate Value at Risk metrics."""
        if len(returns) < 10:
            return {
                'var_1d_95': Money(Decimal('0')),
                'var_1d_99': Money(Decimal('0')),
                'var_1w_95': Money(Decimal('0')),
                'var_1w_99': Money(Decimal('0')),
                'cvar_1d_95': Money(Decimal('0')),
                'cvar_1d_99': Money(Decimal('0'))
            }
        
        sorted_returns = sorted(returns)
        n = len(sorted_returns)
        
        # Calculate percentiles
        var_95_return = sorted_returns[int(0.05 * n)]
        var_99_return = sorted_returns[int(0.01 * n)]
        
        # Convert to dollar amounts
        var_1d_95 = Money(abs(var_95_return * current_value.amount))
        var_1d_99 = Money(abs(var_99_return * current_value.amount))
        
        # Weekly VaR (approximation)
        var_1w_95 = Money(var_1d_95.amount * Decimal('7') ** Decimal('0.5'))
        var_1w_99 = Money(var_1d_99.amount * Decimal('7') ** Decimal('0.5'))
        
        # Conditional VaR (average of tail)
        tail_95 = [r for r in sorted_returns if r <= var_95_return]
        tail_99 = [r for r in sorted_returns if r <= var_99_return]
        
        cvar_95_return = sum(tail_95) / len(tail_95) if tail_95 else var_95_return
        cvar_99_return = sum(tail_99) / len(tail_99) if tail_99 else var_99_return
        
        cvar_1d_95 = Money(abs(cvar_95_return * current_value.amount))
        cvar_1d_99 = Money(abs(cvar_99_return * current_value.amount))
        
        return {
            'var_1d_95': var_1d_95,
            'var_1d_99': var_1d_99,
            'var_1w_95': var_1w_95,
            'var_1w_99': var_1w_99,
            'cvar_1d_95': cvar_1d_95,
            'cvar_1d_99': cvar_1d_99
        }
    
    def _get_portfolio_returns(self, period: PerformancePeriod) -> List[Decimal]:
        """Get portfolio returns for specified period."""
        portfolio_history = self.portfolio_manager.get_portfolio_history()
        return self._extract_returns_from_snapshots(portfolio_history.snapshots)
    
    def _calculate_beta_alpha(
        self,
        portfolio_returns: List[Decimal],
        benchmark_returns: List[Decimal]
    ) -> Tuple[Decimal, Decimal]:
        """Calculate beta and alpha."""
        if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
            return Decimal('1'), Decimal('0')
        
        # Align lengths
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]
        
        try:
            # Calculate beta using covariance
            portfolio_mean = sum(portfolio_returns) / len(portfolio_returns)
            benchmark_mean = sum(benchmark_returns) / len(benchmark_returns)
            
            covariance = sum(
                (p - portfolio_mean) * (b - benchmark_mean)
                for p, b in zip(portfolio_returns, benchmark_returns)
            ) / (len(portfolio_returns) - 1)
            
            benchmark_variance = sum(
                (b - benchmark_mean) ** 2 for b in benchmark_returns
            ) / (len(benchmark_returns) - 1)
            
            beta = covariance / benchmark_variance if benchmark_variance > 0 else Decimal('1')
            
            # Calculate alpha
            alpha = portfolio_mean - beta * benchmark_mean
            
            return beta, alpha
            
        except Exception:
            return Decimal('1'), Decimal('0')
    
    def _calculate_capture_ratios(
        self,
        portfolio_returns: List[Decimal],
        benchmark_returns: List[Decimal]
    ) -> Tuple[Decimal, Decimal]:
        """Calculate up and down capture ratios."""
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return Decimal('1'), Decimal('1')
        
        up_portfolio = []
        up_benchmark = []
        down_portfolio = []
        down_benchmark = []
        
        for p, b in zip(portfolio_returns, benchmark_returns):
            if b > 0:
                up_portfolio.append(p)
                up_benchmark.append(b)
            elif b < 0:
                down_portfolio.append(p)
                down_benchmark.append(b)
        
        # Calculate capture ratios
        up_capture = Decimal('1')
        if up_benchmark:
            avg_up_portfolio = sum(up_portfolio) / len(up_portfolio)
            avg_up_benchmark = sum(up_benchmark) / len(up_benchmark)
            up_capture = avg_up_portfolio / avg_up_benchmark if avg_up_benchmark > 0 else Decimal('1')
        
        down_capture = Decimal('1')
        if down_benchmark:
            avg_down_portfolio = sum(down_portfolio) / len(down_portfolio)
            avg_down_benchmark = sum(down_benchmark) / len(down_benchmark)
            down_capture = avg_down_portfolio / avg_down_benchmark if avg_down_benchmark < 0 else Decimal('1')
        
        return up_capture, down_capture
    
    def _calculate_correlation(
        self,
        portfolio_returns: List[Decimal],
        benchmark_returns: List[Decimal]
    ) -> Decimal:
        """Calculate correlation coefficient."""
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return Decimal('0')
        
        try:
            # Convert to float for calculation
            port_float = [float(r) for r in portfolio_returns]
            bench_float = [float(r) for r in benchmark_returns]
            
            correlation = statistics.correlation(port_float, bench_float)
            return Decimal(str(correlation))
            
        except Exception:
            return Decimal('0')
    
    def _aggregate_returns(
        self,
        daily_returns: List[Tuple[date, Decimal]],
        frequency: str
    ) -> List[Tuple[date, Decimal]]:
        """Aggregate daily returns to weekly or monthly."""
        if not daily_returns:
            return []
        
        aggregated = {}
        
        for return_date, return_value in daily_returns:
            if frequency == 'weekly':
                # Get Monday of the week
                key = return_date - timedelta(days=return_date.weekday())
            else:  # monthly
                # Get first day of month
                key = return_date.replace(day=1)
            
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(return_value)
        
        # Calculate period returns
        result = []
        for period_start, period_returns in aggregated.items():
            # Compound returns
            period_return = Decimal('1')
            for r in period_returns:
                period_return *= (1 + r)
            period_return -= 1
            
            result.append((period_start, period_return))
        
        return sorted(result)
    
    def _calculate_return_streaks(
        self,
        daily_returns: List[Tuple[date, Decimal]]
    ) -> Dict[str, Any]:
        """Calculate winning and losing streaks."""
        if not daily_returns:
            return {
                'longest_winning_streak': 0,
                'longest_losing_streak': 0,
                'current_streak': 0,
                'current_streak_type': 'none'
            }
        
        longest_winning = 0
        longest_losing = 0
        current_winning = 0
        current_losing = 0
        
        for _, return_value in daily_returns:
            if return_value > 0:
                current_winning += 1
                current_losing = 0
                longest_winning = max(longest_winning, current_winning)
            elif return_value < 0:
                current_losing += 1
                current_winning = 0
                longest_losing = max(longest_losing, current_losing)
            else:
                current_winning = 0
                current_losing = 0
        
        # Determine current streak
        if current_winning > 0:
            current_streak = current_winning
            current_streak_type = 'winning'
        elif current_losing > 0:
            current_streak = current_losing
            current_streak_type = 'losing'
        else:
            current_streak = 0
            current_streak_type = 'none'
        
        return {
            'longest_winning_streak': longest_winning,
            'longest_losing_streak': longest_losing,
            'current_streak': current_streak,
            'current_streak_type': current_streak_type
        }
    
    def _calculate_distribution_measures(
        self,
        returns: List[Decimal]
    ) -> Tuple[Decimal, Decimal]:
        """Calculate skewness and kurtosis."""
        if len(returns) < 3:
            return Decimal('0'), Decimal('0')
        
        try:
            mean = sum(returns) / len(returns)
            variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
            std_dev = variance ** Decimal('0.5')
            
            if std_dev == 0:
                return Decimal('0'), Decimal('0')
            
            # Skewness
            skewness = sum((r - mean) ** 3 for r in returns) / (len(returns) * std_dev ** 3)
            
            # Kurtosis
            kurtosis = sum((r - mean) ** 4 for r in returns) / (len(returns) * std_dev ** 4) - 3
            
            return skewness, kurtosis
            
        except Exception:
            return Decimal('0'), Decimal('0')
    
    def _create_empty_metrics(
        self,
        period: PerformancePeriod,
        start_date: date,
        end_date: date
    ) -> PerformanceMetrics:
        """Create empty performance metrics."""
        return PerformanceMetrics(
            period=period,
            start_date=start_date,
            end_date=end_date,
            calculation_time=datetime.now(),
            total_return=Decimal('0'),
            annualized_return=Decimal('0'),
            cumulative_return=Decimal('0'),
            average_daily_return=Decimal('0'),
            geometric_mean_return=Decimal('0'),
            volatility=Decimal('0'),
            annualized_volatility=Decimal('0'),
            downside_volatility=Decimal('0'),
            max_drawdown=Decimal('0'),
            max_drawdown_duration=0,
            sharpe_ratio=Decimal('0'),
            sortino_ratio=Decimal('0'),
            calmar_ratio=Decimal('0'),
            win_rate=Decimal('0'),
            profit_factor=Decimal('0'),
            largest_win=Money(Decimal('0')),
            largest_loss=Money(Decimal('0')),
            average_win=Money(Decimal('0')),
            average_loss=Money(Decimal('0')),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            portfolio_value=Money(Decimal('0')),
            cash_percentage=Decimal('1'),
            var_95=Money(Decimal('0')),
            var_99=Money(Decimal('0'))
        )
    
    def _create_empty_risk_metrics(self) -> RiskMetrics:
        """Create empty risk metrics."""
        return RiskMetrics(
            calculation_date=date.today(),
            daily_volatility=Decimal('0'),
            weekly_volatility=Decimal('0'),
            monthly_volatility=Decimal('0'),
            annualized_volatility=Decimal('0'),
            downside_deviation=Decimal('0'),
            upside_deviation=Decimal('0'),
            downside_volatility=Decimal('0'),
            var_1d_95=Money(Decimal('0')),
            var_1d_99=Money(Decimal('0')),
            var_1w_95=Money(Decimal('0')),
            var_1w_99=Money(Decimal('0')),
            cvar_1d_95=Money(Decimal('0')),
            cvar_1d_99=Money(Decimal('0')),
            current_drawdown=Decimal('0'),
            max_drawdown=Decimal('0'),
            max_drawdown_duration=0,
            average_drawdown=Decimal('0'),
            sharpe_ratio=Decimal('0'),
            sortino_ratio=Decimal('0'),
            calmar_ratio=Decimal('0'),
            sterling_ratio=Decimal('0'),
            skewness=Decimal('0'),
            kurtosis=Decimal('0')
        )
    
    def _create_empty_returns_analysis(self, period: PerformancePeriod) -> ReturnsAnalysis:
        """Create empty returns analysis."""
        return ReturnsAnalysis(
            period=period,
            daily_returns=[],
            weekly_returns=[],
            monthly_returns=[],
            mean_return=Decimal('0'),
            median_return=Decimal('0'),
            std_return=Decimal('0'),
            min_return=Decimal('0'),
            max_return=Decimal('0'),
            percentile_5=Decimal('0'),
            percentile_25=Decimal('0'),
            percentile_75=Decimal('0'),
            percentile_95=Decimal('0'),
            longest_winning_streak=0,
            longest_losing_streak=0,
            current_streak=0,
            current_streak_type='none',
            positive_days=0,
            negative_days=0,
            flat_days=0
        )
    
    def _empty_trade_metrics(self) -> Dict[str, Any]:
        """Return empty trade metrics."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': Decimal('0'),
            'profit_factor': Decimal('0'),
            'largest_win': Money(Decimal('0')),
            'largest_loss': Money(Decimal('0')),
            'average_win': Money(Decimal('0')),
            'average_loss': Money(Decimal('0'))
        }


# Global performance analyzer instance
_performance_analyzer = None


def get_performance_analyzer(
    portfolio_manager: PortfolioManager = None,
    pnl_calculator: PnLCalculator = None,
    position_manager: PositionManager = None
) -> PerformanceAnalyzer:
    """Get global performance analyzer."""
    global _performance_analyzer
    if _performance_analyzer is None:
        _performance_analyzer = PerformanceAnalyzer(
            portfolio_manager, pnl_calculator, position_manager
        )
    return _performance_analyzer