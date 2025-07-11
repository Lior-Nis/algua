"""
Portfolio reporting and visualization system.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from enum import Enum
import json
import csv
from pathlib import Path
from abc import ABC, abstractmethod
import threading

from domain.value_objects import Symbol, Price, Quantity, Money
from .portfolio_manager import PortfolioManager, PortfolioSnapshot
from .performance_analytics import PerformanceAnalyzer, PerformanceMetrics, RiskMetrics
from .portfolio_optimizer import PortfolioOptimizationEngine, OptimizationResult
from .pnl_calculator import PnLCalculator, PnLSnapshot
from .position_manager import PositionManager, Position
from utils.logging import get_logger

logger = get_logger(__name__)


class ReportType(Enum):
    """Types of portfolio reports."""
    PERFORMANCE_SUMMARY = "performance_summary"
    RISK_ANALYSIS = "risk_analysis"
    POSITION_DETAIL = "position_detail"
    ALLOCATION_BREAKDOWN = "allocation_breakdown"
    TRADE_ANALYSIS = "trade_analysis"
    OPTIMIZATION_REPORT = "optimization_report"
    BENCHMARK_COMPARISON = "benchmark_comparison"
    ATTRIBUTION_ANALYSIS = "attribution_analysis"
    COMPLIANCE_REPORT = "compliance_report"
    EXECUTIVE_SUMMARY = "executive_summary"


class ReportFormat(Enum):
    """Report output formats."""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    EXCEL = "excel"
    MARKDOWN = "markdown"


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    report_type: ReportType
    format: ReportFormat
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    include_charts: bool = True
    include_tables: bool = True
    include_summary: bool = True
    currency: str = "USD"
    precision: int = 4
    
    # Filtering options
    symbols_filter: Optional[List[Symbol]] = None
    strategies_filter: Optional[List[str]] = None
    min_position_size: Optional[Money] = None
    
    # Customization
    custom_metrics: List[str] = field(default_factory=list)
    branding: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'report_type': self.report_type.value,
            'format': self.format.value,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'include_charts': self.include_charts,
            'include_tables': self.include_tables,
            'include_summary': self.include_summary,
            'currency': self.currency,
            'precision': self.precision,
            'symbols_filter': [str(s) for s in self.symbols_filter] if self.symbols_filter else None,
            'strategies_filter': self.strategies_filter,
            'min_position_size': float(self.min_position_size.amount) if self.min_position_size else None,
            'custom_metrics': self.custom_metrics,
            'branding': self.branding
        }


@dataclass
class PerformanceReport:
    """Performance report data structure."""
    config: ReportConfig
    generation_time: datetime
    
    # Portfolio overview
    portfolio_summary: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    
    # Position details
    current_positions: List[Dict[str, Any]]
    closed_positions: List[Dict[str, Any]]
    
    # Time series data
    portfolio_value_series: List[Tuple[datetime, Decimal]]
    returns_series: List[Tuple[datetime, Decimal]]
    drawdown_series: List[Tuple[datetime, Decimal]]
    
    # Analytics
    sector_allocation: Dict[str, Decimal]
    strategy_attribution: Dict[str, Dict[str, Any]]
    top_performers: List[Dict[str, Any]]
    worst_performers: List[Dict[str, Any]]
    
    # Risk analysis
    var_analysis: Dict[str, Any]
    correlation_matrix: Dict[Symbol, Dict[Symbol, Decimal]]
    
    # Compliance
    constraint_violations: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config': self.config.to_dict(),
            'generation_time': self.generation_time.isoformat(),
            'portfolio_summary': self.portfolio_summary,
            'performance_metrics': self.performance_metrics,
            'risk_metrics': self.risk_metrics,
            'current_positions': self.current_positions,
            'closed_positions': self.closed_positions,
            'portfolio_value_series': [(dt.isoformat(), float(val)) for dt, val in self.portfolio_value_series],
            'returns_series': [(dt.isoformat(), float(val)) for dt, val in self.returns_series],
            'drawdown_series': [(dt.isoformat(), float(val)) for dt, val in self.drawdown_series],
            'sector_allocation': {sector: float(allocation) for sector, allocation in self.sector_allocation.items()},
            'strategy_attribution': self.strategy_attribution,
            'top_performers': self.top_performers,
            'worst_performers': self.worst_performers,
            'var_analysis': self.var_analysis,
            'correlation_matrix': {
                str(symbol1): {str(symbol2): float(corr) for symbol2, corr in corr_row.items()}
                for symbol1, corr_row in self.correlation_matrix.items()
            },
            'constraint_violations': self.constraint_violations
        }


@dataclass
class RiskReport:
    """Risk-focused report."""
    config: ReportConfig
    generation_time: datetime
    
    # Risk overview
    risk_summary: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    
    # Risk decomposition
    portfolio_risk_contribution: Dict[Symbol, Decimal]
    sector_risk_contribution: Dict[str, Decimal]
    factor_risk_exposure: Dict[str, Decimal]
    
    # Stress testing
    stress_test_results: Dict[str, Dict[str, Any]]
    scenario_analysis: Dict[str, Dict[str, Any]]
    
    # Risk limits
    risk_limit_utilization: Dict[str, Dict[str, Any]]
    
    # Value at Risk
    var_breakdown: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config': self.config.to_dict(),
            'generation_time': self.generation_time.isoformat(),
            'risk_summary': self.risk_summary,
            'risk_metrics': self.risk_metrics,
            'portfolio_risk_contribution': {str(symbol): float(contrib) for symbol, contrib in self.portfolio_risk_contribution.items()},
            'sector_risk_contribution': {sector: float(contrib) for sector, contrib in self.sector_risk_contribution.items()},
            'factor_risk_exposure': {factor: float(exposure) for factor, exposure in self.factor_risk_exposure.items()},
            'stress_test_results': self.stress_test_results,
            'scenario_analysis': self.scenario_analysis,
            'risk_limit_utilization': self.risk_limit_utilization,
            'var_breakdown': self.var_breakdown
        }


@dataclass
class AllocationReport:
    """Portfolio allocation report."""
    config: ReportConfig
    generation_time: datetime
    
    # Current allocation
    current_allocation: Dict[Symbol, Decimal]
    target_allocation: Dict[Symbol, Decimal]
    allocation_drift: Dict[Symbol, Decimal]
    
    # Sector/style breakdown
    sector_allocation: Dict[str, Decimal]
    market_cap_allocation: Dict[str, Decimal]
    geography_allocation: Dict[str, Decimal]
    
    # Optimization insights
    optimization_recommendations: Optional[OptimizationResult]
    rebalancing_suggestions: List[Dict[str, Any]]
    
    # Historical allocation
    allocation_history: List[Tuple[date, Dict[Symbol, Decimal]]]
    
    # Diversification metrics
    concentration_metrics: Dict[str, Any]
    diversification_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config': self.config.to_dict(),
            'generation_time': self.generation_time.isoformat(),
            'current_allocation': {str(symbol): float(weight) for symbol, weight in self.current_allocation.items()},
            'target_allocation': {str(symbol): float(weight) for symbol, weight in self.target_allocation.items()},
            'allocation_drift': {str(symbol): float(drift) for symbol, drift in self.allocation_drift.items()},
            'sector_allocation': {sector: float(weight) for sector, weight in self.sector_allocation.items()},
            'market_cap_allocation': {cap: float(weight) for cap, weight in self.market_cap_allocation.items()},
            'geography_allocation': {geo: float(weight) for geo, weight in self.geography_allocation.items()},
            'optimization_recommendations': self.optimization_recommendations.to_dict() if self.optimization_recommendations else None,
            'rebalancing_suggestions': self.rebalancing_suggestions,
            'allocation_history': [
                (dt.isoformat(), {str(symbol): float(weight) for symbol, weight in allocation.items()})
                for dt, allocation in self.allocation_history
            ],
            'concentration_metrics': self.concentration_metrics,
            'diversification_metrics': self.diversification_metrics
        }


class ReportGenerator(ABC):
    """Abstract base class for report generators."""
    
    @abstractmethod
    def generate(self, report_data: Union[PerformanceReport, RiskReport, AllocationReport]) -> str:
        """Generate report in specific format."""
        pass
    
    @abstractmethod
    def save_to_file(self, report_data: Union[PerformanceReport, RiskReport, AllocationReport], file_path: Path) -> None:
        """Save report to file."""
        pass


class JSONReportGenerator(ReportGenerator):
    """JSON report generator."""
    
    def generate(self, report_data: Union[PerformanceReport, RiskReport, AllocationReport]) -> str:
        """Generate JSON report."""
        return json.dumps(report_data.to_dict(), indent=2, default=str)
    
    def save_to_file(self, report_data: Union[PerformanceReport, RiskReport, AllocationReport], file_path: Path) -> None:
        """Save JSON report to file."""
        with open(file_path, 'w') as f:
            json.dump(report_data.to_dict(), f, indent=2, default=str)


class CSVReportGenerator(ReportGenerator):
    """CSV report generator."""
    
    def generate(self, report_data: Union[PerformanceReport, RiskReport, AllocationReport]) -> str:
        """Generate CSV report."""
        # For CSV, we'll focus on tabular data
        if isinstance(report_data, PerformanceReport):
            return self._generate_performance_csv(report_data)
        elif isinstance(report_data, RiskReport):
            return self._generate_risk_csv(report_data)
        elif isinstance(report_data, AllocationReport):
            return self._generate_allocation_csv(report_data)
        else:
            return ""
    
    def save_to_file(self, report_data: Union[PerformanceReport, RiskReport, AllocationReport], file_path: Path) -> None:
        """Save CSV report to file."""
        csv_content = self.generate(report_data)
        with open(file_path, 'w', newline='') as f:
            f.write(csv_content)
    
    def _generate_performance_csv(self, report: PerformanceReport) -> str:
        """Generate performance CSV."""
        output = []
        
        # Header
        output.append("Portfolio Performance Report")
        output.append(f"Generated: {report.generation_time}")
        output.append("")
        
        # Portfolio summary
        output.append("Portfolio Summary")
        for key, value in report.portfolio_summary.items():
            output.append(f"{key},{value}")
        output.append("")
        
        # Current positions
        output.append("Current Positions")
        if report.current_positions:
            headers = ["Symbol", "Quantity", "Market Value", "Unrealized P&L", "Position %"]
            output.append(",".join(headers))
            
            for position in report.current_positions:
                row = [
                    position.get('symbol', ''),
                    str(position.get('quantity', '')),
                    str(position.get('market_value', '')),
                    str(position.get('unrealized_pnl', '')),
                    str(position.get('position_percentage', ''))
                ]
                output.append(",".join(row))
        
        return "\n".join(output)
    
    def _generate_risk_csv(self, report: RiskReport) -> str:
        """Generate risk CSV."""
        output = []
        
        # Header
        output.append("Risk Analysis Report")
        output.append(f"Generated: {report.generation_time}")
        output.append("")
        
        # Risk metrics
        output.append("Risk Metrics")
        for key, value in report.risk_metrics.items():
            output.append(f"{key},{value}")
        
        return "\n".join(output)
    
    def _generate_allocation_csv(self, report: AllocationReport) -> str:
        """Generate allocation CSV."""
        output = []
        
        # Header
        output.append("Portfolio Allocation Report")
        output.append(f"Generated: {report.generation_time}")
        output.append("")
        
        # Current allocation
        output.append("Current Allocation")
        output.append("Symbol,Weight")
        for symbol, weight in report.current_allocation.items():
            output.append(f"{symbol},{weight}")
        
        return "\n".join(output)


class MarkdownReportGenerator(ReportGenerator):
    """Markdown report generator."""
    
    def generate(self, report_data: Union[PerformanceReport, RiskReport, AllocationReport]) -> str:
        """Generate Markdown report."""
        if isinstance(report_data, PerformanceReport):
            return self._generate_performance_markdown(report_data)
        elif isinstance(report_data, RiskReport):
            return self._generate_risk_markdown(report_data)
        elif isinstance(report_data, AllocationReport):
            return self._generate_allocation_markdown(report_data)
        else:
            return ""
    
    def save_to_file(self, report_data: Union[PerformanceReport, RiskReport, AllocationReport], file_path: Path) -> None:
        """Save Markdown report to file."""
        markdown_content = self.generate(report_data)
        with open(file_path, 'w') as f:
            f.write(markdown_content)
    
    def _generate_performance_markdown(self, report: PerformanceReport) -> str:
        """Generate performance Markdown."""
        md = []
        
        # Header
        md.append("# Portfolio Performance Report")
        md.append(f"**Generated:** {report.generation_time}")
        md.append("")
        
        # Executive Summary
        md.append("## Executive Summary")
        summary = report.portfolio_summary
        md.append(f"- **Portfolio Value:** ${summary.get('portfolio_value', 0):,.2f}")
        md.append(f"- **Total Return:** {summary.get('total_return', 0):.2%}")
        md.append(f"- **Cash Position:** {summary.get('cash_percentage', 0):.1%}")
        md.append("")
        
        # Performance Metrics
        md.append("## Performance Metrics")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        
        metrics = report.performance_metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float, Decimal)):
                if 'return' in key.lower() or 'ratio' in key.lower():
                    md.append(f"| {key.replace('_', ' ').title()} | {float(value):.4f} |")
                else:
                    md.append(f"| {key.replace('_', ' ').title()} | {value} |")
        md.append("")
        
        # Current Positions
        md.append("## Current Positions")
        if report.current_positions:
            md.append("| Symbol | Quantity | Market Value | P&L | Weight |")
            md.append("|--------|----------|--------------|-----|--------|")
            
            for position in report.current_positions:
                symbol = position.get('symbol', '')
                quantity = position.get('quantity', 0)
                market_value = position.get('market_value', 0)
                pnl = position.get('unrealized_pnl', 0)
                weight = position.get('position_percentage', 0)
                
                md.append(f"| {symbol} | {quantity} | ${market_value:,.2f} | ${pnl:,.2f} | {weight:.1%} |")
        md.append("")
        
        # Top Performers
        if report.top_performers:
            md.append("## Top Performers")
            md.append("| Symbol | Return | P&L |")
            md.append("|--------|--------|-----|")
            
            for performer in report.top_performers[:5]:
                symbol = performer.get('symbol', '')
                return_pct = performer.get('return_pct', 0)
                pnl = performer.get('unrealized_pnl', 0)
                md.append(f"| {symbol} | {return_pct:.2%} | ${pnl:,.2f} |")
        md.append("")
        
        # Risk Metrics
        md.append("## Risk Analysis")
        risk = report.risk_metrics
        md.append(f"- **Volatility:** {risk.get('volatility', 0):.2%}")
        md.append(f"- **Sharpe Ratio:** {risk.get('sharpe_ratio', 0):.3f}")
        md.append(f"- **Max Drawdown:** {risk.get('max_drawdown', 0):.2%}")
        md.append(f"- **VaR (95%):** ${risk.get('var_95', 0):,.2f}")
        md.append("")
        
        return "\n".join(md)
    
    def _generate_risk_markdown(self, report: RiskReport) -> str:
        """Generate risk Markdown."""
        md = []
        
        md.append("# Risk Analysis Report")
        md.append(f"**Generated:** {report.generation_time}")
        md.append("")
        
        # Risk Overview
        md.append("## Risk Overview")
        summary = report.risk_summary
        for key, value in summary.items():
            md.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        md.append("")
        
        # Risk Contributions
        md.append("## Risk Contributions by Position")
        md.append("| Symbol | Risk Contribution |")
        md.append("|--------|-------------------|")
        
        for symbol, contribution in report.portfolio_risk_contribution.items():
            md.append(f"| {symbol} | {contribution:.2%} |")
        md.append("")
        
        return "\n".join(md)
    
    def _generate_allocation_markdown(self, report: AllocationReport) -> str:
        """Generate allocation Markdown."""
        md = []
        
        md.append("# Portfolio Allocation Report")
        md.append(f"**Generated:** {report.generation_time}")
        md.append("")
        
        # Current Allocation
        md.append("## Current Allocation")
        md.append("| Symbol | Current Weight | Target Weight | Drift |")
        md.append("|--------|----------------|---------------|-------|")
        
        for symbol in report.current_allocation:
            current = report.current_allocation.get(symbol, Decimal('0'))
            target = report.target_allocation.get(symbol, Decimal('0'))
            drift = report.allocation_drift.get(symbol, Decimal('0'))
            
            md.append(f"| {symbol} | {current:.2%} | {target:.2%} | {drift:+.2%} |")
        md.append("")
        
        # Sector Allocation
        md.append("## Sector Allocation")
        md.append("| Sector | Weight |")
        md.append("|--------|--------|")
        
        for sector, weight in report.sector_allocation.items():
            md.append(f"| {sector} | {weight:.1%} |")
        md.append("")
        
        return "\n".join(md)


class PortfolioReporter:
    """Main portfolio reporting system."""
    
    def __init__(
        self,
        portfolio_manager: PortfolioManager = None,
        performance_analyzer: PerformanceAnalyzer = None,
        optimization_engine: PortfolioOptimizationEngine = None,
        pnl_calculator: PnLCalculator = None,
        position_manager: PositionManager = None
    ):
        self.portfolio_manager = portfolio_manager
        self.performance_analyzer = performance_analyzer
        self.optimization_engine = optimization_engine
        self.pnl_calculator = pnl_calculator
        self.position_manager = position_manager
        
        # Report generators
        self.generators = {
            ReportFormat.JSON: JSONReportGenerator(),
            ReportFormat.CSV: CSVReportGenerator(),
            ReportFormat.MARKDOWN: MarkdownReportGenerator()
        }
        
        # Report cache
        self.report_cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_duration = timedelta(minutes=10)
        
        self._lock = threading.Lock()
    
    def generate_performance_report(self, config: ReportConfig) -> PerformanceReport:
        """Generate comprehensive performance report."""
        with self._lock:
            generation_time = datetime.now()
            
            # Portfolio summary
            portfolio_summary = self._generate_portfolio_summary(config)
            
            # Performance metrics
            performance_metrics = self._generate_performance_metrics(config)
            
            # Risk metrics
            risk_metrics = self._generate_risk_metrics(config)
            
            # Position details
            current_positions, closed_positions = self._generate_position_details(config)
            
            # Time series data
            portfolio_value_series = self._generate_portfolio_value_series(config)
            returns_series = self._generate_returns_series(config)
            drawdown_series = self._generate_drawdown_series(config)
            
            # Analytics
            sector_allocation = self._generate_sector_allocation(config)
            strategy_attribution = self._generate_strategy_attribution(config)
            top_performers = self._generate_top_performers(config)
            worst_performers = self._generate_worst_performers(config)
            
            # Risk analysis
            var_analysis = self._generate_var_analysis(config)
            correlation_matrix = self._generate_correlation_matrix(config)
            
            # Compliance
            constraint_violations = self._generate_constraint_violations(config)
            
            return PerformanceReport(
                config=config,
                generation_time=generation_time,
                portfolio_summary=portfolio_summary,
                performance_metrics=performance_metrics,
                risk_metrics=risk_metrics,
                current_positions=current_positions,
                closed_positions=closed_positions,
                portfolio_value_series=portfolio_value_series,
                returns_series=returns_series,
                drawdown_series=drawdown_series,
                sector_allocation=sector_allocation,
                strategy_attribution=strategy_attribution,
                top_performers=top_performers,
                worst_performers=worst_performers,
                var_analysis=var_analysis,
                correlation_matrix=correlation_matrix,
                constraint_violations=constraint_violations
            )
    
    def generate_risk_report(self, config: ReportConfig) -> RiskReport:
        """Generate risk-focused report."""
        with self._lock:
            generation_time = datetime.now()
            
            # Risk overview
            risk_summary = self._generate_risk_summary(config)
            risk_metrics = self._generate_risk_metrics(config)
            
            # Risk decomposition
            portfolio_risk_contribution = self._generate_portfolio_risk_contribution(config)
            sector_risk_contribution = self._generate_sector_risk_contribution(config)
            factor_risk_exposure = self._generate_factor_risk_exposure(config)
            
            # Stress testing
            stress_test_results = self._generate_stress_test_results(config)
            scenario_analysis = self._generate_scenario_analysis(config)
            
            # Risk limits
            risk_limit_utilization = self._generate_risk_limit_utilization(config)
            
            # VaR breakdown
            var_breakdown = self._generate_var_breakdown(config)
            
            return RiskReport(
                config=config,
                generation_time=generation_time,
                risk_summary=risk_summary,
                risk_metrics=risk_metrics,
                portfolio_risk_contribution=portfolio_risk_contribution,
                sector_risk_contribution=sector_risk_contribution,
                factor_risk_exposure=factor_risk_exposure,
                stress_test_results=stress_test_results,
                scenario_analysis=scenario_analysis,
                risk_limit_utilization=risk_limit_utilization,
                var_breakdown=var_breakdown
            )
    
    def generate_allocation_report(self, config: ReportConfig) -> AllocationReport:
        """Generate allocation-focused report."""
        with self._lock:
            generation_time = datetime.now()
            
            # Current vs target allocation
            current_allocation = self._generate_current_allocation(config)
            target_allocation = self._generate_target_allocation(config)
            allocation_drift = self._calculate_allocation_drift(current_allocation, target_allocation)
            
            # Breakdowns
            sector_allocation = self._generate_sector_allocation(config)
            market_cap_allocation = self._generate_market_cap_allocation(config)
            geography_allocation = self._generate_geography_allocation(config)
            
            # Optimization
            optimization_recommendations = self._generate_optimization_recommendations(config)
            rebalancing_suggestions = self._generate_rebalancing_suggestions(config)
            
            # Historical data
            allocation_history = self._generate_allocation_history(config)
            
            # Diversification metrics
            concentration_metrics = self._generate_concentration_metrics(config)
            diversification_metrics = self._generate_diversification_metrics(config)
            
            return AllocationReport(
                config=config,
                generation_time=generation_time,
                current_allocation=current_allocation,
                target_allocation=target_allocation,
                allocation_drift=allocation_drift,
                sector_allocation=sector_allocation,
                market_cap_allocation=market_cap_allocation,
                geography_allocation=geography_allocation,
                optimization_recommendations=optimization_recommendations,
                rebalancing_suggestions=rebalancing_suggestions,
                allocation_history=allocation_history,
                concentration_metrics=concentration_metrics,
                diversification_metrics=diversification_metrics
            )
    
    def generate_report(self, config: ReportConfig) -> Union[PerformanceReport, RiskReport, AllocationReport]:
        """Generate report based on type."""
        if config.report_type == ReportType.PERFORMANCE_SUMMARY:
            return self.generate_performance_report(config)
        elif config.report_type == ReportType.RISK_ANALYSIS:
            return self.generate_risk_report(config)
        elif config.report_type == ReportType.ALLOCATION_BREAKDOWN:
            return self.generate_allocation_report(config)
        else:
            # Default to performance report
            return self.generate_performance_report(config)
    
    def export_report(
        self,
        report: Union[PerformanceReport, RiskReport, AllocationReport],
        output_format: ReportFormat,
        file_path: Path
    ) -> None:
        """Export report to file."""
        if output_format not in self.generators:
            raise ValueError(f"Unsupported report format: {output_format}")
        
        generator = self.generators[output_format]
        generator.save_to_file(report, file_path)
        
        logger.info(f"Report exported to {file_path} in {output_format.value} format")
    
    def get_report_as_string(
        self,
        report: Union[PerformanceReport, RiskReport, AllocationReport],
        output_format: ReportFormat
    ) -> str:
        """Get report as string."""
        if output_format not in self.generators:
            raise ValueError(f"Unsupported report format: {output_format}")
        
        generator = self.generators[output_format]
        return generator.generate(report)
    
    # Helper methods for data generation
    
    def _generate_portfolio_summary(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate portfolio summary."""
        if not self.portfolio_manager:
            return {}
        
        summary = self.portfolio_manager.get_portfolio_summary()
        return summary
    
    def _generate_performance_metrics(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate performance metrics."""
        if not self.performance_analyzer:
            return {}
        
        try:
            from .performance_analytics import PerformancePeriod
            metrics = self.performance_analyzer.calculate_performance_metrics(PerformancePeriod.INCEPTION)
            return metrics.to_dict()
        except Exception as e:
            logger.warning(f"Failed to generate performance metrics: {e}")
            return {}
    
    def _generate_risk_metrics(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate risk metrics."""
        if not self.performance_analyzer:
            return {}
        
        try:
            risk_metrics = self.performance_analyzer.calculate_risk_metrics()
            return risk_metrics.__dict__
        except Exception as e:
            logger.warning(f"Failed to generate risk metrics: {e}")
            return {}
    
    def _generate_position_details(self, config: ReportConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generate position details."""
        current_positions = []
        closed_positions = []
        
        if self.position_manager:
            # Current positions
            positions = self.position_manager.get_all_positions()
            for symbol, position in positions.items():
                current_positions.append(position.to_dict())
            
            # Closed positions (recent ones)
            for position in self.position_manager.closed_positions[-20:]:  # Last 20 closed positions
                closed_positions.append(position.to_dict())
        
        return current_positions, closed_positions
    
    def _generate_portfolio_value_series(self, config: ReportConfig) -> List[Tuple[datetime, Decimal]]:
        """Generate portfolio value time series."""
        if not self.portfolio_manager:
            return []
        
        history = self.portfolio_manager.get_portfolio_history()
        return [(snap.timestamp, snap.portfolio.get_portfolio_value().amount) for snap in history.snapshots]
    
    def _generate_returns_series(self, config: ReportConfig) -> List[Tuple[datetime, Decimal]]:
        """Generate returns time series."""
        if not self.portfolio_manager:
            return []
        
        history = self.portfolio_manager.get_portfolio_history()
        return [(snap.timestamp, snap.portfolio.daily_return) for snap in history.snapshots]
    
    def _generate_drawdown_series(self, config: ReportConfig) -> List[Tuple[datetime, Decimal]]:
        """Generate drawdown time series."""
        # Simplified drawdown calculation
        value_series = self._generate_portfolio_value_series(config)
        
        if len(value_series) < 2:
            return []
        
        drawdown_series = []
        running_max = value_series[0][1]
        
        for timestamp, value in value_series:
            if value > running_max:
                running_max = value
            
            drawdown = (value - running_max) / running_max if running_max > 0 else Decimal('0')
            drawdown_series.append((timestamp, drawdown))
        
        return drawdown_series
    
    def _generate_sector_allocation(self, config: ReportConfig) -> Dict[str, Decimal]:
        """Generate sector allocation."""
        # Simplified sector allocation
        return {
            'Technology': Decimal('0.30'),
            'Healthcare': Decimal('0.20'),
            'Financial': Decimal('0.15'),
            'Consumer': Decimal('0.15'),
            'Industrial': Decimal('0.10'),
            'Energy': Decimal('0.05'),
            'Utilities': Decimal('0.05')
        }
    
    def _generate_strategy_attribution(self, config: ReportConfig) -> Dict[str, Dict[str, Any]]:
        """Generate strategy attribution."""
        return {
            'momentum_strategy': {
                'return_contribution': 0.045,
                'risk_contribution': 0.12,
                'allocation': 0.40
            },
            'mean_reversion_strategy': {
                'return_contribution': 0.023,
                'risk_contribution': 0.08,
                'allocation': 0.30
            },
            'carry_strategy': {
                'return_contribution': 0.018,
                'risk_contribution': 0.06,
                'allocation': 0.30
            }
        }
    
    def _generate_top_performers(self, config: ReportConfig) -> List[Dict[str, Any]]:
        """Generate top performers."""
        if not self.position_manager:
            return []
        
        profitable_positions = self.position_manager.get_profitable_positions()
        sorted_positions = sorted(profitable_positions, key=lambda p: p.unrealized_pnl.amount, reverse=True)
        
        return [
            {
                'symbol': str(pos.symbol),
                'return_pct': float(pos.get_pnl_percentage()),
                'unrealized_pnl': float(pos.unrealized_pnl.amount),
                'market_value': float(pos.market_value.amount)
            }
            for pos in sorted_positions[:10]
        ]
    
    def _generate_worst_performers(self, config: ReportConfig) -> List[Dict[str, Any]]:
        """Generate worst performers."""
        if not self.position_manager:
            return []
        
        losing_positions = self.position_manager.get_losing_positions()
        sorted_positions = sorted(losing_positions, key=lambda p: p.unrealized_pnl.amount)
        
        return [
            {
                'symbol': str(pos.symbol),
                'return_pct': float(pos.get_pnl_percentage()),
                'unrealized_pnl': float(pos.unrealized_pnl.amount),
                'market_value': float(pos.market_value.amount)
            }
            for pos in sorted_positions[:10]
        ]
    
    def _generate_var_analysis(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate VaR analysis."""
        return {
            'var_1d_95': 25000,
            'var_1d_99': 45000,
            'var_1w_95': 65000,
            'var_1w_99': 120000,
            'expected_shortfall_95': 35000,
            'expected_shortfall_99': 65000
        }
    
    def _generate_correlation_matrix(self, config: ReportConfig) -> Dict[Symbol, Dict[Symbol, Decimal]]:
        """Generate correlation matrix."""
        # Simplified correlation matrix
        symbols = [Symbol('AAPL'), Symbol('MSFT'), Symbol('GOOGL')]
        matrix = {}
        
        for i, symbol1 in enumerate(symbols):
            matrix[symbol1] = {}
            for j, symbol2 in enumerate(symbols):
                if i == j:
                    matrix[symbol1][symbol2] = Decimal('1.0')
                else:
                    # Random correlation for demo
                    matrix[symbol1][symbol2] = Decimal('0.6')
        
        return matrix
    
    def _generate_constraint_violations(self, config: ReportConfig) -> List[Dict[str, Any]]:
        """Generate constraint violations."""
        return []  # No violations for demo
    
    # Additional helper methods for other report types...
    
    def _generate_risk_summary(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate risk summary."""
        return {
            'portfolio_volatility': 0.15,
            'portfolio_var_95': 25000,
            'max_drawdown': 0.12,
            'sharpe_ratio': 1.45,
            'beta': 0.95
        }
    
    def _generate_portfolio_risk_contribution(self, config: ReportConfig) -> Dict[Symbol, Decimal]:
        """Generate portfolio risk contribution."""
        return {
            Symbol('AAPL'): Decimal('0.25'),
            Symbol('MSFT'): Decimal('0.20'),
            Symbol('GOOGL'): Decimal('0.18'),
            Symbol('TSLA'): Decimal('0.15'),
            Symbol('AMZN'): Decimal('0.22')
        }
    
    def _generate_sector_risk_contribution(self, config: ReportConfig) -> Dict[str, Decimal]:
        """Generate sector risk contribution."""
        return {
            'Technology': Decimal('0.45'),
            'Healthcare': Decimal('0.20'),
            'Financial': Decimal('0.15'),
            'Consumer': Decimal('0.12'),
            'Energy': Decimal('0.08')
        }
    
    def _generate_factor_risk_exposure(self, config: ReportConfig) -> Dict[str, Decimal]:
        """Generate factor risk exposure."""
        return {
            'Market': Decimal('0.95'),
            'Size': Decimal('0.15'),
            'Value': Decimal('-0.10'),
            'Momentum': Decimal('0.25'),
            'Quality': Decimal('0.20')
        }
    
    def _generate_stress_test_results(self, config: ReportConfig) -> Dict[str, Dict[str, Any]]:
        """Generate stress test results."""
        return {
            'market_crash_2008': {
                'portfolio_impact': -0.35,
                'max_loss': 450000,
                'recovery_time_days': 180
            },
            'covid_crash_2020': {
                'portfolio_impact': -0.28,
                'max_loss': 360000,
                'recovery_time_days': 120
            }
        }
    
    def _generate_scenario_analysis(self, config: ReportConfig) -> Dict[str, Dict[str, Any]]:
        """Generate scenario analysis."""
        return {
            'bull_market': {
                'probability': 0.30,
                'expected_return': 0.25,
                'portfolio_impact': 320000
            },
            'bear_market': {
                'probability': 0.20,
                'expected_return': -0.20,
                'portfolio_impact': -250000
            },
            'neutral_market': {
                'probability': 0.50,
                'expected_return': 0.08,
                'portfolio_impact': 100000
            }
        }
    
    def _generate_risk_limit_utilization(self, config: ReportConfig) -> Dict[str, Dict[str, Any]]:
        """Generate risk limit utilization."""
        return {
            'var_limit': {
                'limit': 50000,
                'current': 25000,
                'utilization': 0.50,
                'status': 'Green'
            },
            'concentration_limit': {
                'limit': 0.20,
                'current': 0.15,
                'utilization': 0.75,
                'status': 'Yellow'
            }
        }
    
    def _generate_var_breakdown(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate VaR breakdown."""
        return {
            'total_var': 25000,
            'diversified_var': 22000,
            'undiversified_var': 35000,
            'diversification_benefit': 13000,
            'component_var': {
                'equities': 18000,
                'bonds': 3000,
                'commodities': 2000,
                'currencies': 2000
            }
        }
    
    def _generate_current_allocation(self, config: ReportConfig) -> Dict[Symbol, Decimal]:
        """Generate current allocation."""
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
    
    def _generate_target_allocation(self, config: ReportConfig) -> Dict[Symbol, Decimal]:
        """Generate target allocation."""
        # For demo purposes, return equal weights
        current = self._generate_current_allocation(config)
        if not current:
            return {}
        
        equal_weight = Decimal('1') / Decimal(str(len(current)))
        return {symbol: equal_weight for symbol in current.keys()}
    
    def _calculate_allocation_drift(
        self,
        current: Dict[Symbol, Decimal],
        target: Dict[Symbol, Decimal]
    ) -> Dict[Symbol, Decimal]:
        """Calculate allocation drift."""
        drift = {}
        all_symbols = set(current.keys()) | set(target.keys())
        
        for symbol in all_symbols:
            current_weight = current.get(symbol, Decimal('0'))
            target_weight = target.get(symbol, Decimal('0'))
            drift[symbol] = current_weight - target_weight
        
        return drift
    
    def _generate_market_cap_allocation(self, config: ReportConfig) -> Dict[str, Decimal]:
        """Generate market cap allocation."""
        return {
            'Large Cap': Decimal('0.60'),
            'Mid Cap': Decimal('0.25'),
            'Small Cap': Decimal('0.15')
        }
    
    def _generate_geography_allocation(self, config: ReportConfig) -> Dict[str, Decimal]:
        """Generate geography allocation."""
        return {
            'US': Decimal('0.70'),
            'Europe': Decimal('0.15'),
            'Asia': Decimal('0.10'),
            'Emerging Markets': Decimal('0.05')
        }
    
    def _generate_optimization_recommendations(self, config: ReportConfig) -> Optional[OptimizationResult]:
        """Generate optimization recommendations."""
        # Return None for demo - would use optimization engine in practice
        return None
    
    def _generate_rebalancing_suggestions(self, config: ReportConfig) -> List[Dict[str, Any]]:
        """Generate rebalancing suggestions."""
        return [
            {
                'symbol': 'AAPL',
                'action': 'reduce',
                'current_weight': 0.25,
                'target_weight': 0.20,
                'trade_amount': -50000,
                'reason': 'Overweight vs target'
            },
            {
                'symbol': 'MSFT',
                'action': 'increase',
                'current_weight': 0.15,
                'target_weight': 0.20,
                'trade_amount': 50000,
                'reason': 'Underweight vs target'
            }
        ]
    
    def _generate_allocation_history(self, config: ReportConfig) -> List[Tuple[date, Dict[Symbol, Decimal]]]:
        """Generate allocation history."""
        # Simplified allocation history
        today = date.today()
        history = []
        
        for i in range(30, 0, -1):  # Last 30 days
            hist_date = today - timedelta(days=i)
            # Generate sample allocation for each date
            allocation = {
                Symbol('AAPL'): Decimal('0.25'),
                Symbol('MSFT'): Decimal('0.20'),
                Symbol('GOOGL'): Decimal('0.15')
            }
            history.append((hist_date, allocation))
        
        return history
    
    def _generate_concentration_metrics(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate concentration metrics."""
        return {
            'herfindahl_index': 0.18,
            'top_5_concentration': 0.65,
            'top_10_concentration': 0.85,
            'effective_number_of_positions': 12.5
        }
    
    def _generate_diversification_metrics(self, config: ReportConfig) -> Dict[str, Any]:
        """Generate diversification metrics."""
        return {
            'diversification_ratio': 1.25,
            'correlation_dispersion': 0.15,
            'risk_dispersion': 0.22,
            'sector_diversification': 0.78
        }


# Global reporter instance
_portfolio_reporter = None


def get_portfolio_reporter(
    portfolio_manager: PortfolioManager = None,
    performance_analyzer: PerformanceAnalyzer = None,
    optimization_engine: PortfolioOptimizationEngine = None,
    pnl_calculator: PnLCalculator = None,
    position_manager: PositionManager = None
) -> PortfolioReporter:
    """Get global portfolio reporter."""
    global _portfolio_reporter
    if _portfolio_reporter is None:
        _portfolio_reporter = PortfolioReporter(
            portfolio_manager, performance_analyzer, optimization_engine,
            pnl_calculator, position_manager
        )
    return _portfolio_reporter