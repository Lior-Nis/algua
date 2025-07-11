"""
Risk management module for Algua trading platform.

This module provides comprehensive risk management functionality including:
- Position sizing with multiple methods (fixed risk, Kelly criterion, volatility targeting, ATR-based)
- Stop loss mechanisms (fixed, trailing, ATR-based, time-based)
- Portfolio exposure limits and concentration risk monitoring
- Drawdown controls with recovery mode management
- Risk event system with configurable handlers
- Comprehensive risk configuration management
"""

# Core risk management components
from .position_sizer import (
    PositionSizer, PositionSizeParams, PositionSizingMethod,
    BasePositionSizer, FixedRiskPositionSizer, KellyPositionSizer,
    VolatilityTargetPositionSizer, ATRBasedPositionSizer
)
from .risk_calculator import RiskCalculator
from .risk_manager import RiskManager

# Stop loss system
from .stop_loss import (
    StopLossManager, StopLossType, StopLossOrder,
    BaseStopLoss, FixedPercentageStopLoss, TrailingPercentageStopLoss,
    ATRBasedStopLoss, TimeBasedStopLoss, StopLossFactory,
    get_stop_loss_manager
)

# Portfolio limits and exposure management
from .portfolio_limits import (
    PortfolioRiskLimiter, ExposureType, ExposureLimit,
    PortfolioExposure, get_portfolio_limiter
)

# Drawdown controls
from .drawdown_controls import (
    DrawdownController, DrawdownMetrics, DrawdownSeverity,
    RecoveryMode, DrawdownPeriod, DrawdownAnalyzer,
    get_drawdown_controller
)

# Risk interfaces and abstractions
from .interfaces import (
    RiskLevel, RiskEventType, RiskEvent, RiskMetrics, PositionRisk,
    RiskManagerInterface, RiskCalculatorProtocol, PositionSizerProtocol,
    StopLossProtocol, RiskMonitorProtocol, RiskEventHandlerProtocol,
    RiskConfigurationProtocol
)

# Configuration system
from .configuration import (
    RiskConfiguration, RiskConfigurationManager,
    get_risk_config, update_risk_config
)

# Event system
from .event_system import (
    RiskEventBus, RiskEventHandler, LoggingRiskEventHandler,
    EmailRiskEventHandler, PositionCloseRiskEventHandler,
    RiskEventFactory, RiskEventStats,
    get_risk_event_bus, publish_risk_event
)

__all__ = [
    # Core components
    "PositionSizer", "RiskCalculator", "RiskManager",
    
    # Position sizing
    "PositionSizeParams", "PositionSizingMethod", "BasePositionSizer",
    "FixedRiskPositionSizer", "KellyPositionSizer", "VolatilityTargetPositionSizer",
    "ATRBasedPositionSizer",
    
    # Stop loss system
    "StopLossManager", "StopLossType", "StopLossOrder", "BaseStopLoss",
    "FixedPercentageStopLoss", "TrailingPercentageStopLoss", "ATRBasedStopLoss",
    "TimeBasedStopLoss", "StopLossFactory", "get_stop_loss_manager",
    
    # Portfolio limits
    "PortfolioRiskLimiter", "ExposureType", "ExposureLimit",
    "PortfolioExposure", "get_portfolio_limiter",
    
    # Drawdown controls
    "DrawdownController", "DrawdownMetrics", "DrawdownSeverity",
    "RecoveryMode", "DrawdownPeriod", "DrawdownAnalyzer",
    "get_drawdown_controller",
    
    # Interfaces
    "RiskLevel", "RiskEventType", "RiskEvent", "RiskMetrics", "PositionRisk",
    "RiskManagerInterface", "RiskCalculatorProtocol", "PositionSizerProtocol",
    "StopLossProtocol", "RiskMonitorProtocol", "RiskEventHandlerProtocol",
    "RiskConfigurationProtocol",
    
    # Configuration
    "RiskConfiguration", "RiskConfigurationManager",
    "get_risk_config", "update_risk_config",
    
    # Event system
    "RiskEventBus", "RiskEventHandler", "LoggingRiskEventHandler",
    "EmailRiskEventHandler", "PositionCloseRiskEventHandler",
    "RiskEventFactory", "RiskEventStats",
    "get_risk_event_bus", "publish_risk_event"
]