"""
Order execution engine with multiple execution strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import random

from domain.value_objects import Symbol, Price, Quantity
from .order_types import Order, OrderType, OrderSide, MarketOrder, LimitOrder, StopOrder, StopLimitOrder
from .fill_handler import FillHandler, Fill, PartialFill
from .order_lifecycle import get_lifecycle_manager, OrderTransition
from utils.logging import get_logger

logger = get_logger(__name__)


class ExecutionVenue(Enum):
    """Execution venues."""
    BROKER = "broker"
    EXCHANGE = "exchange"
    DARK_POOL = "dark_pool"
    SMART_ROUTING = "smart_routing"


class ExecutionPriority(Enum):
    """Execution priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class ExecutionResult:
    """Result of order execution attempt."""
    success: bool
    fills: List[Fill]
    partial_fill: Optional[PartialFill]
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    venue: Optional[ExecutionVenue] = None
    remaining_quantity: Optional[Quantity] = None


@dataclass
class FillReport:
    """Comprehensive fill report."""
    order_id: str
    symbol: Symbol
    total_fills: int
    total_quantity: Quantity
    average_price: Price
    total_commission: Decimal
    total_slippage: Decimal
    execution_time_ms: float
    venue: ExecutionVenue
    fill_details: List[Fill]


class ExecutionStrategy(ABC):
    """Abstract base class for execution strategies."""
    
    @abstractmethod
    def execute_order(
        self,
        order: Order,
        market_data: Dict[str, Any],
        venue: ExecutionVenue = ExecutionVenue.SMART_ROUTING
    ) -> ExecutionResult:
        """Execute an order using this strategy."""
        pass
    
    @abstractmethod
    def can_handle_order(self, order: Order) -> bool:
        """Check if strategy can handle this order type."""
        pass


class MarketExecutionStrategy(ExecutionStrategy):
    """Execution strategy for market orders."""
    
    def __init__(self, fill_handler: FillHandler):
        self.fill_handler = fill_handler
        self.max_slippage = Decimal('0.005')  # 0.5% max slippage
    
    def execute_order(
        self,
        order: Order,
        market_data: Dict[str, Any],
        venue: ExecutionVenue = ExecutionVenue.SMART_ROUTING
    ) -> ExecutionResult:
        """Execute market order immediately."""
        start_time = datetime.now()
        
        try:
            # Market orders execute immediately
            fills, partial_fill = self.fill_handler.execute_order(
                order,
                market_data,
                partial_fill_probability=Decimal('0.05')  # 5% chance of partial fill
            )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Check slippage limits
            for fill in fills:
                if fill.market_impact and fill.market_impact > self.max_slippage:
                    logger.warning(
                        f"High slippage detected: {fill.market_impact:.4f} for order {order.order_id}"
                    )
            
            # Apply fills to order
            for fill in fills:
                order.add_fill(fill)
            
            remaining_quantity = order.remaining_quantity if partial_fill else None
            
            logger.info(
                f"Market order {order.order_id} executed: {len(fills)} fills, "
                f"{execution_time:.2f}ms execution time"
            )
            
            return ExecutionResult(
                success=True,
                fills=fills,
                partial_fill=partial_fill,
                execution_time_ms=execution_time,
                venue=venue,
                remaining_quantity=remaining_quantity
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Market order execution failed for {order.order_id}: {e}")
            
            return ExecutionResult(
                success=False,
                fills=[],
                partial_fill=None,
                error_message=str(e),
                execution_time_ms=execution_time,
                venue=venue
            )
    
    def can_handle_order(self, order: Order) -> bool:
        """Market strategy handles market orders."""
        return isinstance(order, MarketOrder)


class LimitExecutionStrategy(ExecutionStrategy):
    """Execution strategy for limit orders."""
    
    def __init__(self, fill_handler: FillHandler):
        self.fill_handler = fill_handler
        self.price_improvement_threshold = Decimal('0.001')  # 0.1% improvement
    
    def execute_order(
        self,
        order: Order,
        market_data: Dict[str, Any],
        venue: ExecutionVenue = ExecutionVenue.SMART_ROUTING
    ) -> ExecutionResult:
        """Execute limit order if price conditions are met."""
        start_time = datetime.now()
        
        try:
            if not isinstance(order, LimitOrder):
                raise ValueError("LimitExecutionStrategy requires LimitOrder")
            
            current_price = Price(market_data.get('current_price', Decimal('100')))
            
            # Check if limit order can execute
            if not order.is_executable(current_price):
                return ExecutionResult(
                    success=False,
                    fills=[],
                    partial_fill=None,
                    error_message="Limit price not reached",
                    execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    venue=venue
                )
            
            # Execute with potential price improvement
            fills, partial_fill = self.fill_handler.execute_order(
                order,
                market_data,
                partial_fill_probability=Decimal('0.15')  # 15% chance for limit orders
            )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Apply fills to order
            for fill in fills:
                order.add_fill(fill)
            
            # Check for price improvement
            if fills:
                avg_fill_price = sum(f.price.value * f.quantity.value for f in fills) / sum(f.quantity.value for f in fills)
                if order.side == OrderSide.BUY and avg_fill_price < order.price.value:
                    improvement = order.price.value - avg_fill_price
                    logger.info(f"Price improvement: ${improvement:.4f} for order {order.order_id}")
                elif order.side == OrderSide.SELL and avg_fill_price > order.price.value:
                    improvement = avg_fill_price - order.price.value
                    logger.info(f"Price improvement: ${improvement:.4f} for order {order.order_id}")
            
            remaining_quantity = order.remaining_quantity if partial_fill else None
            
            return ExecutionResult(
                success=True,
                fills=fills,
                partial_fill=partial_fill,
                execution_time_ms=execution_time,
                venue=venue,
                remaining_quantity=remaining_quantity
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Limit order execution failed for {order.order_id}: {e}")
            
            return ExecutionResult(
                success=False,
                fills=[],
                partial_fill=None,
                error_message=str(e),
                execution_time_ms=execution_time,
                venue=venue
            )
    
    def can_handle_order(self, order: Order) -> bool:
        """Limit strategy handles limit orders."""
        return isinstance(order, LimitOrder)


class StopExecutionStrategy(ExecutionStrategy):
    """Execution strategy for stop orders."""
    
    def __init__(self, fill_handler: FillHandler):
        self.fill_handler = fill_handler
        self.triggered_orders: Dict[str, bool] = {}
    
    def execute_order(
        self,
        order: Order,
        market_data: Dict[str, Any],
        venue: ExecutionVenue = ExecutionVenue.SMART_ROUTING
    ) -> ExecutionResult:
        """Execute stop order when triggered."""
        start_time = datetime.now()
        
        try:
            if not isinstance(order, (StopOrder, StopLimitOrder)):
                raise ValueError("StopExecutionStrategy requires StopOrder or StopLimitOrder")
            
            current_price = Price(market_data.get('current_price', Decimal('100')))
            
            # Check if stop order can execute (triggers check)
            if not order.is_executable(current_price):
                return ExecutionResult(
                    success=False,
                    fills=[],
                    partial_fill=None,
                    error_message="Stop condition not met",
                    execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                    venue=venue
                )
            
            # Mark as triggered if not already
            if order.order_id not in self.triggered_orders:
                self.triggered_orders[order.order_id] = True
                logger.info(f"Stop order {order.order_id} triggered at price {current_price.value}")
            
            # Execute the triggered order
            fills, partial_fill = self.fill_handler.execute_order(
                order,
                market_data,
                partial_fill_probability=Decimal('0.10')  # 10% chance for stop orders
            )
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Apply fills to order
            for fill in fills:
                order.add_fill(fill)
            
            remaining_quantity = order.remaining_quantity if partial_fill else None
            
            return ExecutionResult(
                success=True,
                fills=fills,
                partial_fill=partial_fill,
                execution_time_ms=execution_time,
                venue=venue,
                remaining_quantity=remaining_quantity
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Stop order execution failed for {order.order_id}: {e}")
            
            return ExecutionResult(
                success=False,
                fills=[],
                partial_fill=None,
                error_message=str(e),
                execution_time_ms=execution_time,
                venue=venue
            )
    
    def can_handle_order(self, order: Order) -> bool:
        """Stop strategy handles stop and stop-limit orders."""
        return isinstance(order, (StopOrder, StopLimitOrder))


class OrderExecutionEngine:
    """Main order execution engine."""
    
    def __init__(self, max_concurrent_executions: int = 10):
        self.fill_handler = FillHandler()
        self.lifecycle_manager = get_lifecycle_manager()
        self.max_concurrent_executions = max_concurrent_executions
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_executions)
        
        # Initialize execution strategies
        self.strategies: List[ExecutionStrategy] = [
            MarketExecutionStrategy(self.fill_handler),
            LimitExecutionStrategy(self.fill_handler),
            StopExecutionStrategy(self.fill_handler)
        ]
        
        # Execution queue
        self.pending_executions: List[Tuple[Order, Dict[str, Any], ExecutionVenue]] = []
        self.active_executions: Dict[str, asyncio.Task] = {}
        
        # Execution statistics
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time_ms': 0,
            'total_fills': 0,
            'total_slippage': Decimal('0'),
            'venue_distribution': {}
        }
    
    def execute_order(
        self,
        order: Order,
        market_data: Dict[str, Any],
        venue: ExecutionVenue = ExecutionVenue.SMART_ROUTING,
        priority: ExecutionPriority = ExecutionPriority.NORMAL
    ) -> ExecutionResult:
        """Execute a single order."""
        logger.info(f"Executing order {order.order_id} ({order.order_type.value})")
        
        # Update lifecycle
        self.lifecycle_manager.transition_order(order, OrderTransition.SUBMIT)
        
        # Find appropriate strategy
        strategy = self._get_execution_strategy(order)
        if not strategy:
            error_msg = f"No execution strategy available for order type {order.order_type.value}"
            logger.error(error_msg)
            
            self.lifecycle_manager.transition_order(order, OrderTransition.REJECT)
            order.reject(error_msg)
            
            return ExecutionResult(
                success=False,
                fills=[],
                partial_fill=None,
                error_message=error_msg,
                venue=venue
            )
        
        # Execute order
        self.lifecycle_manager.transition_order(order, OrderTransition.ACKNOWLEDGE)
        result = strategy.execute_order(order, market_data, venue)
        
        # Update lifecycle based on result
        if result.success:
            if result.partial_fill:
                self.lifecycle_manager.transition_order(order, OrderTransition.PARTIAL_FILL)
            else:
                self.lifecycle_manager.transition_order(order, OrderTransition.COMPLETE_FILL)
        else:
            self.lifecycle_manager.transition_order(order, OrderTransition.REJECT)
            order.reject(result.error_message or "Execution failed")
        
        # Update statistics
        self._update_execution_stats(result, venue)
        
        return result
    
    def execute_orders_batch(
        self,
        orders: List[Tuple[Order, Dict[str, Any]]],
        venue: ExecutionVenue = ExecutionVenue.SMART_ROUTING
    ) -> List[ExecutionResult]:
        """Execute multiple orders in parallel."""
        logger.info(f"Executing batch of {len(orders)} orders")
        
        futures = []
        for order, market_data in orders:
            future = self.executor.submit(self.execute_order, order, market_data, venue)
            futures.append(future)
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)  # 30 second timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Batch execution error: {e}")
                results.append(ExecutionResult(
                    success=False,
                    fills=[],
                    partial_fill=None,
                    error_message=str(e),
                    venue=venue
                ))
        
        return results
    
    def cancel_order(self, order: Order, reason: str = "User request") -> bool:
        """Cancel an order."""
        try:
            # Check if order can be canceled
            if not order.is_active():
                logger.warning(f"Cannot cancel order {order.order_id} in status {order.status.value}")
                return False
            
            # Cancel the order
            order.cancel(reason)
            self.lifecycle_manager.transition_order(order, OrderTransition.CANCEL, reason)
            
            logger.info(f"Order {order.order_id} canceled: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order.order_id}: {e}")
            return False
    
    def get_execution_report(self, order: Order) -> Optional[FillReport]:
        """Get execution report for an order."""
        if not order.fills:
            return None
        
        total_quantity = sum(fill.quantity.value for fill in order.fills)
        total_value = sum(fill.quantity.value * fill.price.value for fill in order.fills)
        average_price = Price(total_value / total_quantity) if total_quantity > 0 else Price(Decimal('0'))
        
        total_commission = sum(fill.commission.amount for fill in order.fills)
        total_slippage = sum(fill.slippage.amount for fill in order.fills if fill.slippage)
        
        # Calculate execution time
        if order.fills:
            first_fill_time = min(fill.timestamp for fill in order.fills)
            last_fill_time = max(fill.timestamp for fill in order.fills)
            execution_time_ms = (last_fill_time - first_fill_time).total_seconds() * 1000
        else:
            execution_time_ms = 0
        
        return FillReport(
            order_id=order.order_id,
            symbol=order.symbol,
            total_fills=len(order.fills),
            total_quantity=Quantity(total_quantity),
            average_price=average_price,
            total_commission=total_commission,
            total_slippage=total_slippage,
            execution_time_ms=execution_time_ms,
            venue=ExecutionVenue.SMART_ROUTING,  # Default
            fill_details=order.fills.copy()
        )
    
    def _get_execution_strategy(self, order: Order) -> Optional[ExecutionStrategy]:
        """Get appropriate execution strategy for order."""
        for strategy in self.strategies:
            if strategy.can_handle_order(order):
                return strategy
        return None
    
    def _update_execution_stats(self, result: ExecutionResult, venue: ExecutionVenue) -> None:
        """Update execution statistics."""
        self.execution_stats['total_executions'] += 1
        
        if result.success:
            self.execution_stats['successful_executions'] += 1
            self.execution_stats['total_fills'] += len(result.fills)
            
            if result.execution_time_ms:
                self.execution_stats['total_execution_time_ms'] += result.execution_time_ms
            
            # Update slippage
            for fill in result.fills:
                if fill.slippage:
                    self.execution_stats['total_slippage'] += fill.slippage.amount
        else:
            self.execution_stats['failed_executions'] += 1
        
        # Update venue distribution
        venue_name = venue.value
        if venue_name not in self.execution_stats['venue_distribution']:
            self.execution_stats['venue_distribution'][venue_name] = 0
        self.execution_stats['venue_distribution'][venue_name] += 1
    
    def add_execution_strategy(self, strategy: ExecutionStrategy) -> None:
        """Add custom execution strategy."""
        self.strategies.append(strategy)
        logger.info(f"Added execution strategy: {strategy.__class__.__name__}")
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        stats = self.execution_stats.copy()
        
        if stats['total_executions'] > 0:
            stats['success_rate'] = stats['successful_executions'] / stats['total_executions']
            stats['average_execution_time_ms'] = stats['total_execution_time_ms'] / stats['successful_executions'] if stats['successful_executions'] > 0 else 0
            stats['average_slippage'] = float(stats['total_slippage']) / stats['total_fills'] if stats['total_fills'] > 0 else 0
        else:
            stats['success_rate'] = 0
            stats['average_execution_time_ms'] = 0
            stats['average_slippage'] = 0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_execution_time_ms': 0,
            'total_fills': 0,
            'total_slippage': Decimal('0'),
            'venue_distribution': {}
        }
        logger.info("Execution statistics reset")
    
    def shutdown(self) -> None:
        """Shutdown execution engine."""
        self.executor.shutdown(wait=True)
        logger.info("Order execution engine shutdown")


# Global execution engine instance
_execution_engine = None


def get_execution_engine() -> OrderExecutionEngine:
    """Get global order execution engine."""
    global _execution_engine
    if _execution_engine is None:
        _execution_engine = OrderExecutionEngine()
    return _execution_engine