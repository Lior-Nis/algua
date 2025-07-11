"""
Portfolio tracking module for Algua trading platform.

This module provides comprehensive portfolio tracking functionality including:
- Real-time position management and tracking
- P&L calculation (realized and unrealized)
- Portfolio performance analytics and metrics
- Risk attribution and exposure analysis
- Position lifecycle management
- Portfolio rebalancing and optimization
- Performance benchmarking and attribution
"""

# Core portfolio components
from .position_manager import (
    Position, PositionType, PositionStatus,
    PositionManager, PositionSnapshot, PositionHistory,
    get_position_manager
)

# Portfolio management
from .portfolio_manager import (
    Portfolio, PortfolioSnapshot, PortfolioHistory,
    PortfolioManager, PortfolioConfiguration,
    get_portfolio_manager
)

# P&L calculation
from .pnl_calculator import (
    PnLCalculator, PnLSnapshot, PnLHistory,
    RealizedPnL, UnrealizedPnL, TotalPnL,
    PnLAttribution, get_pnl_calculator
)

# Performance analytics
from .performance_analytics import (
    PerformanceMetrics, PerformanceAnalyzer,
    BenchmarkComparison, RiskMetrics, ReturnsAnalysis,
    SharpeCalculator, DrawdownAnalyzer, PerformancePeriod,
    RiskMetricType, get_performance_analyzer
)

# Portfolio optimization
from .portfolio_optimizer import (
    PortfolioOptimizer, OptimizationObjective,
    OptimizationConstraints, OptimizationResult,
    MeanVarianceOptimizer, RiskParityOptimizer,
    PortfolioOptimizationEngine, MarketData,
    get_portfolio_optimization_engine
)

# Reporting and visualization
from .portfolio_reporter import (
    PortfolioReporter, ReportType, ReportFormat,
    PerformanceReport, RiskReport, AllocationReport,
    ReportConfig, get_portfolio_reporter
)

__all__ = [
    # Position management
    "Position", "PositionType", "PositionStatus",
    "PositionManager", "PositionSnapshot", "PositionHistory",
    "get_position_manager",
    
    # Portfolio management
    "Portfolio", "PortfolioSnapshot", "PortfolioHistory",
    "PortfolioManager", "PortfolioConfiguration",
    "get_portfolio_manager",
    
    # P&L calculation
    "PnLCalculator", "PnLSnapshot", "PnLHistory",
    "RealizedPnL", "UnrealizedPnL", "TotalPnL",
    "PnLAttribution", "get_pnl_calculator",
    
    # Performance analytics
    "PerformanceMetrics", "PerformanceAnalyzer",
    "BenchmarkComparison", "RiskMetrics", "ReturnsAnalysis",
    "SharpeCalculator", "DrawdownAnalyzer", "PerformancePeriod",
    "RiskMetricType", "get_performance_analyzer",
    
    # Portfolio optimization
    "PortfolioOptimizer", "OptimizationObjective",
    "OptimizationConstraints", "OptimizationResult",
    "MeanVarianceOptimizer", "RiskParityOptimizer",
    "PortfolioOptimizationEngine", "MarketData",
    "get_portfolio_optimization_engine",
    
    # Reporting
    "PortfolioReporter", "ReportType", "ReportFormat",
    "PerformanceReport", "RiskReport", "AllocationReport",
    "ReportConfig", "get_portfolio_reporter"
]