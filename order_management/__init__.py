"""
Order management module for Algua trading platform.

This module provides comprehensive order management functionality including:
- Order lifecycle management with state machine
- Multiple order types (market, limit, stop, stop-limit)
- Order execution engine with routing
- Fill handling and slippage simulation
- Order validation and risk checks
- Real-time order status tracking
"""

# Order types and core components
from .order_types import (
    OrderType, OrderSide, OrderStatus, TimeInForce,
    Order, MarketOrder, LimitOrder, StopOrder, StopLimitOrder,
    OrderFactory
)

# Order lifecycle and state management
from .order_lifecycle import (
    OrderState, OrderTransition, OrderStateMachine,
    OrderLifecycleManager
)

# Order execution system
from .execution_engine import (
    OrderExecutionEngine, ExecutionStrategy, FillReport,
    MarketExecutionStrategy, LimitExecutionStrategy,
    StopExecutionStrategy, ExecutionResult
)

# Order validation and routing
from .order_validator import (
    OrderValidator, ValidationRule, ValidationResult,
    RiskValidationRule, MarketHoursValidationRule,
    PositionSizeValidationRule
)

# Fill handling and slippage
from .fill_handler import (
    FillHandler, FillType, Fill, PartialFill,
    SlippageCalculator, SlippageModel, LinearSlippageModel,
    SquareRootSlippageModel, MarketImpactModel
)

# Order tracking and monitoring
from .order_tracker import (
    OrderTracker, OrderSnapshot, OrderHistory,
    OrderMetrics, OrderPerformanceAnalyzer
)

__all__ = [
    # Order types
    "OrderType", "OrderSide", "OrderStatus", "TimeInForce",
    "Order", "MarketOrder", "LimitOrder", "StopOrder", "StopLimitOrder",
    "OrderFactory",
    
    # Order lifecycle
    "OrderState", "OrderTransition", "OrderStateMachine",
    "OrderLifecycleManager",
    
    # Execution engine
    "OrderExecutionEngine", "ExecutionStrategy", "FillReport",
    "MarketExecutionStrategy", "LimitExecutionStrategy",
    "StopExecutionStrategy", "ExecutionResult",
    
    # Validation
    "OrderValidator", "ValidationRule", "ValidationResult",
    "RiskValidationRule", "MarketHoursValidationRule",
    "PositionSizeValidationRule",
    
    # Fill handling
    "FillHandler", "FillType", "Fill", "PartialFill",
    "SlippageCalculator", "SlippageModel", "LinearSlippageModel",
    "SquareRootSlippageModel", "MarketImpactModel",
    
    # Order tracking
    "OrderTracker", "OrderSnapshot", "OrderHistory",
    "OrderMetrics", "OrderPerformanceAnalyzer"
]