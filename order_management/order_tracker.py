"""
Order tracking and monitoring system.
"""

from typing import Dict, List, Optional, Set, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import threading
from queue import Queue

from domain.value_objects import Symbol, Price, Quantity, Money
from .order_types import Order, OrderStatus, OrderSide, OrderType
from .fill_handler import Fill
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OrderSnapshot:
    """Point-in-time snapshot of an order."""
    timestamp: datetime
    order_id: str
    status: OrderStatus
    filled_quantity: Quantity
    remaining_quantity: Quantity
    average_fill_price: Optional[Price]
    commission_paid: Money
    current_market_price: Optional[Price] = None
    unrealized_pnl: Optional[Money] = None
    notes: Optional[str] = None


@dataclass
class OrderMetrics:
    """Performance metrics for an order."""
    order_id: str
    symbol: Symbol
    side: OrderSide
    order_type: OrderType
    
    # Timing metrics
    creation_time: datetime
    submission_time: Optional[datetime]
    first_fill_time: Optional[datetime]
    completion_time: Optional[datetime]
    total_duration_seconds: Optional[float]
    time_to_first_fill_seconds: Optional[float]
    
    # Execution metrics
    requested_quantity: Quantity
    filled_quantity: Quantity
    fill_rate: Decimal  # filled / requested
    number_of_fills: int
    average_fill_price: Optional[Price]
    
    # Cost metrics
    total_commission: Money
    estimated_slippage: Money
    total_transaction_cost: Money
    
    # Performance metrics
    execution_shortfall: Optional[Money]  # vs benchmark price
    implementation_shortfall: Optional[Money]  # total cost of execution
    
    def __post_init__(self):
        # Calculate derived metrics
        if self.requested_quantity.value > 0:
            self.fill_rate = self.filled_quantity.value / self.requested_quantity.value
        else:
            self.fill_rate = Decimal('0')
        
        if self.completion_time and self.creation_time:
            self.total_duration_seconds = (self.completion_time - self.creation_time).total_seconds()
        
        if self.first_fill_time and self.submission_time:
            self.time_to_first_fill_seconds = (self.first_fill_time - self.submission_time).total_seconds()


@dataclass
class OrderHistory:
    """Complete history of an order."""
    order: Order
    snapshots: List[OrderSnapshot] = field(default_factory=list)
    fills: List[Fill] = field(default_factory=list)
    state_changes: List[Dict[str, any]] = field(default_factory=list)
    metrics: Optional[OrderMetrics] = None
    
    def add_snapshot(self, snapshot: OrderSnapshot) -> None:
        """Add order snapshot."""
        self.snapshots.append(snapshot)
    
    def add_fill(self, fill: Fill) -> None:
        """Add fill to history."""
        self.fills.append(fill)
    
    def add_state_change(self, old_status: OrderStatus, new_status: OrderStatus, reason: str = "") -> None:
        """Add state change record."""
        self.state_changes.append({
            'timestamp': datetime.now().isoformat(),
            'old_status': old_status.value,
            'new_status': new_status.value,
            'reason': reason
        })
    
    def calculate_metrics(self, benchmark_price: Optional[Price] = None) -> OrderMetrics:
        """Calculate order metrics."""
        # Timing data
        creation_time = self.order.created_at
        submission_time = self.order.submitted_at
        first_fill_time = min(fill.timestamp for fill in self.fills) if self.fills else None
        completion_time = self.order.filled_at or self.order.canceled_at
        
        # Execution data
        filled_quantity = self.order.filled_quantity
        number_of_fills = len(self.fills)
        average_fill_price = self.order.average_fill_price
        
        # Cost calculations
        total_commission = Money(sum(fill.commission.amount for fill in self.fills))
        estimated_slippage = Money(sum(fill.slippage.amount for fill in self.fills if fill.slippage))
        total_transaction_cost = Money(total_commission.amount + estimated_slippage.amount)
        
        # Performance calculations
        execution_shortfall = None
        implementation_shortfall = None
        
        if benchmark_price and average_fill_price and filled_quantity.value > 0:
            # Calculate execution shortfall (price difference vs benchmark)
            price_diff = average_fill_price.value - benchmark_price.value
            if self.order.side == OrderSide.SELL:
                price_diff = -price_diff  # For sells, lower price is worse
            
            execution_shortfall = Money(price_diff * filled_quantity.value)
            
            # Implementation shortfall includes transaction costs
            implementation_shortfall = Money(execution_shortfall.amount + total_transaction_cost.amount)
        
        metrics = OrderMetrics(
            order_id=self.order.order_id,
            symbol=self.order.symbol,
            side=self.order.side,
            order_type=self.order.order_type,
            creation_time=creation_time,
            submission_time=submission_time,
            first_fill_time=first_fill_time,
            completion_time=completion_time,
            total_duration_seconds=None,
            time_to_first_fill_seconds=None,
            requested_quantity=self.order.quantity,
            filled_quantity=filled_quantity,
            fill_rate=Decimal('0'),
            number_of_fills=number_of_fills,
            average_fill_price=average_fill_price,
            total_commission=total_commission,
            estimated_slippage=estimated_slippage,
            total_transaction_cost=total_transaction_cost,
            execution_shortfall=execution_shortfall,
            implementation_shortfall=implementation_shortfall
        )
        
        self.metrics = metrics
        return metrics


class OrderTracker:
    """Order tracking and monitoring system."""
    
    def __init__(self, snapshot_interval_seconds: int = 30):
        self.orders: Dict[str, Order] = {}
        self.order_histories: Dict[str, OrderHistory] = {}
        self.active_orders: Set[str] = set()
        self.completed_orders: Set[str] = set()
        
        # Monitoring
        self.snapshot_interval = snapshot_interval_seconds
        self.last_snapshot_time = datetime.now()
        self.monitoring_enabled = True
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Order updates queue
        self.update_queue = Queue()
        
        # Statistics
        self.tracking_stats = {
            'total_orders_tracked': 0,
            'active_orders_count': 0,
            'completed_orders_count': 0,
            'total_snapshots_taken': 0,
            'orders_by_status': defaultdict(int),
            'orders_by_type': defaultdict(int),
            'average_order_duration_seconds': 0
        }
        
        self._lock = threading.Lock()
    
    def start_monitoring(self) -> None:
        """Start order monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self.monitoring_enabled = True
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Order tracking monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop order monitoring thread."""
        self.monitoring_enabled = False
        self._stop_monitoring.set()
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        
        logger.info("Order tracking monitoring stopped")
    
    def track_order(self, order: Order) -> None:
        """Start tracking an order."""
        with self._lock:
            self.orders[order.order_id] = order
            self.order_histories[order.order_id] = OrderHistory(order=order)
            self.active_orders.add(order.order_id)
            
            # Update statistics
            self.tracking_stats['total_orders_tracked'] += 1
            self.tracking_stats['active_orders_count'] += 1
            self.tracking_stats['orders_by_type'][order.order_type.value] += 1
            
            # Take initial snapshot
            self._take_snapshot(order.order_id)
            
            logger.info(f"Started tracking order {order.order_id} ({order.order_type.value})")
    
    def update_order_status(
        self,
        order_id: str,
        old_status: OrderStatus,
        new_status: OrderStatus,
        reason: str = ""
    ) -> None:
        """Update order status and record change."""
        with self._lock:
            if order_id not in self.order_histories:
                logger.warning(f"Attempted to update untracked order {order_id}")
                return
            
            history = self.order_histories[order_id]
            history.add_state_change(old_status, new_status, reason)
            
            # Update statistics
            self.tracking_stats['orders_by_status'][old_status.value] -= 1
            self.tracking_stats['orders_by_status'][new_status.value] += 1
            
            # Move to completed if order is done
            if new_status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                if order_id in self.active_orders:
                    self.active_orders.remove(order_id)
                    self.completed_orders.add(order_id)
                    self.tracking_stats['active_orders_count'] -= 1
                    self.tracking_stats['completed_orders_count'] += 1
            
            # Take snapshot after status change
            self._take_snapshot(order_id)
            
            logger.debug(f"Order {order_id} status updated: {old_status.value} -> {new_status.value}")
    
    def record_fill(self, order_id: str, fill: Fill) -> None:
        """Record a fill for an order."""
        with self._lock:
            if order_id not in self.order_histories:
                logger.warning(f"Attempted to record fill for untracked order {order_id}")
                return
            
            history = self.order_histories[order_id]
            history.add_fill(fill)
            
            # Take snapshot after fill
            self._take_snapshot(order_id)
            
            logger.debug(f"Recorded fill for order {order_id}: {fill.quantity.value} @ {fill.price.value}")
    
    def get_order_history(self, order_id: str) -> Optional[OrderHistory]:
        """Get complete history for an order."""
        with self._lock:
            return self.order_histories.get(order_id)
    
    def get_order_metrics(self, order_id: str, benchmark_price: Optional[Price] = None) -> Optional[OrderMetrics]:
        """Get performance metrics for an order."""
        with self._lock:
            history = self.order_histories.get(order_id)
            if history:
                return history.calculate_metrics(benchmark_price)
            return None
    
    def get_active_orders(self) -> List[Order]:
        """Get list of currently active orders."""
        with self._lock:
            return [self.orders[order_id] for order_id in self.active_orders if order_id in self.orders]
    
    def get_orders_by_symbol(self, symbol: Symbol) -> List[Order]:
        """Get orders for a specific symbol."""
        with self._lock:
            return [order for order in self.orders.values() if order.symbol == symbol]
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get orders with specific status."""
        with self._lock:
            return [order for order in self.orders.values() if order.status == status]
    
    def search_orders(
        self,
        symbol: Optional[Symbol] = None,
        status: Optional[OrderStatus] = None,
        order_type: Optional[OrderType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Order]:
        """Search orders with multiple criteria."""
        with self._lock:
            results = []
            
            for order in self.orders.values():
                # Apply filters
                if symbol and order.symbol != symbol:
                    continue
                if status and order.status != status:
                    continue
                if order_type and order.order_type != order_type:
                    continue
                if start_time and order.created_at < start_time:
                    continue
                if end_time and order.created_at > end_time:
                    continue
                
                results.append(order)
            
            return results
    
    def _take_snapshot(self, order_id: str, current_market_price: Optional[Price] = None) -> None:
        """Take snapshot of order state."""
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        
        # Calculate unrealized P&L if market price available
        unrealized_pnl = None
        if current_market_price and order.average_fill_price and order.filled_quantity.value > 0:
            price_diff = current_market_price.value - order.average_fill_price.value
            if order.side == OrderSide.SELL:
                price_diff = -price_diff
            unrealized_pnl = Money(price_diff * order.filled_quantity.value)
        
        snapshot = OrderSnapshot(
            timestamp=datetime.now(),
            order_id=order.order_id,
            status=order.status,
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.remaining_quantity,
            average_fill_price=order.average_fill_price,
            commission_paid=order.commission or Money(Decimal('0')),
            current_market_price=current_market_price,
            unrealized_pnl=unrealized_pnl
        )
        
        if order_id in self.order_histories:
            self.order_histories[order_id].add_snapshot(snapshot)
            self.tracking_stats['total_snapshots_taken'] += 1
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_enabled and not self._stop_monitoring.is_set():
            try:
                current_time = datetime.now()
                
                # Take periodic snapshots
                if (current_time - self.last_snapshot_time).total_seconds() >= self.snapshot_interval:
                    with self._lock:
                        for order_id in self.active_orders.copy():
                            self._take_snapshot(order_id)
                    
                    self.last_snapshot_time = current_time
                
                # Process any pending updates
                self._process_update_queue()
                
                # Sleep until next check
                self._stop_monitoring.wait(timeout=1.0)
                
            except Exception as e:
                logger.error(f"Error in order monitoring loop: {e}")
    
    def _process_update_queue(self) -> None:
        """Process pending order updates."""
        while not self.update_queue.empty():
            try:
                update = self.update_queue.get_nowait()
                # Process update based on type
                if update.get('type') == 'status_change':
                    self.update_order_status(
                        update['order_id'],
                        update['old_status'],
                        update['new_status'],
                        update.get('reason', '')
                    )
                elif update.get('type') == 'fill':
                    self.record_fill(update['order_id'], update['fill'])
            except Exception as e:
                logger.error(f"Error processing order update: {e}")
    
    def cleanup_old_orders(self, retention_days: int = 7) -> int:
        """Clean up old completed orders."""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        cleaned_count = 0
        
        with self._lock:
            orders_to_remove = []
            
            for order_id in self.completed_orders.copy():
                if order_id in self.orders:
                    order = self.orders[order_id]
                    completion_time = order.filled_at or order.canceled_at
                    
                    if completion_time and completion_time < cutoff_time:
                        orders_to_remove.append(order_id)
            
            for order_id in orders_to_remove:
                if order_id in self.orders:
                    del self.orders[order_id]
                if order_id in self.order_histories:
                    del self.order_histories[order_id]
                if order_id in self.completed_orders:
                    self.completed_orders.remove(order_id)
                
                cleaned_count += 1
                self.tracking_stats['completed_orders_count'] -= 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old orders (older than {retention_days} days)")
        
        return cleaned_count
    
    def get_tracking_statistics(self) -> Dict[str, any]:
        """Get order tracking statistics."""
        with self._lock:
            stats = self.tracking_stats.copy()
            
            # Calculate average order duration
            total_duration = 0
            completed_with_duration = 0
            
            for order_id in self.completed_orders:
                if order_id in self.order_histories:
                    metrics = self.order_histories[order_id].metrics
                    if metrics and metrics.total_duration_seconds:
                        total_duration += metrics.total_duration_seconds
                        completed_with_duration += 1
            
            if completed_with_duration > 0:
                stats['average_order_duration_seconds'] = total_duration / completed_with_duration
            
            # Add current status distribution
            current_status_dist = defaultdict(int)
            for order in self.orders.values():
                current_status_dist[order.status.value] += 1
            
            stats['current_status_distribution'] = dict(current_status_dist)
            stats['monitoring_enabled'] = self.monitoring_enabled
            stats['snapshot_interval_seconds'] = self.snapshot_interval
            
            return stats


class OrderPerformanceAnalyzer:
    """Analyzer for order execution performance."""
    
    def __init__(self, order_tracker: OrderTracker):
        self.tracker = order_tracker
    
    def analyze_execution_performance(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbol: Optional[Symbol] = None
    ) -> Dict[str, any]:
        """Analyze execution performance over a period."""
        # Get orders in date range
        orders = self.tracker.search_orders(
            symbol=symbol,
            start_time=start_date,
            end_time=end_date
        )
        
        if not orders:
            return {'message': 'No orders found for analysis'}
        
        # Calculate metrics
        completed_orders = [o for o in orders if o.is_complete()]
        total_orders = len(orders)
        completion_rate = len(completed_orders) / total_orders if total_orders > 0 else 0
        
        # Fill rate analysis
        fill_rates = []
        execution_times = []
        commission_costs = []
        
        for order in completed_orders:
            history = self.tracker.get_order_history(order.order_id)
            if history and history.metrics:
                metrics = history.metrics
                fill_rates.append(float(metrics.fill_rate))
                
                if metrics.total_duration_seconds:
                    execution_times.append(metrics.total_duration_seconds)
                
                commission_costs.append(float(metrics.total_commission.amount))
        
        # Calculate averages
        avg_fill_rate = sum(fill_rates) / len(fill_rates) if fill_rates else 0
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        avg_commission = sum(commission_costs) / len(commission_costs) if commission_costs else 0
        
        # Order type distribution
        type_distribution = defaultdict(int)
        for order in orders:
            type_distribution[order.order_type.value] += 1
        
        # Status distribution
        status_distribution = defaultdict(int)
        for order in orders:
            status_distribution[order.status.value] += 1
        
        return {
            'analysis_period': {
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None,
                'symbol': str(symbol) if symbol else 'All'
            },
            'order_summary': {
                'total_orders': total_orders,
                'completed_orders': len(completed_orders),
                'completion_rate': completion_rate
            },
            'execution_metrics': {
                'average_fill_rate': avg_fill_rate,
                'average_execution_time_seconds': avg_execution_time,
                'average_commission_cost': avg_commission
            },
            'distributions': {
                'order_types': dict(type_distribution),
                'order_status': dict(status_distribution)
            }
        }
    
    def get_worst_performing_orders(
        self,
        limit: int = 10,
        metric: str = 'fill_rate'
    ) -> List[Dict[str, any]]:
        """Get worst performing orders by specified metric."""
        order_performances = []
        
        for order_id, history in self.tracker.order_histories.items():
            if history.metrics:
                metrics = history.metrics
                
                performance_value = 0
                if metric == 'fill_rate':
                    performance_value = float(metrics.fill_rate)
                elif metric == 'execution_time':
                    performance_value = metrics.total_duration_seconds or 0
                elif metric == 'commission':
                    performance_value = float(metrics.total_commission.amount)
                elif metric == 'slippage':
                    performance_value = float(metrics.estimated_slippage.amount)
                
                order_performances.append({
                    'order_id': order_id,
                    'symbol': str(metrics.symbol),
                    'order_type': metrics.order_type.value,
                    'metric_value': performance_value,
                    'metrics': metrics
                })
        
        # Sort by metric (ascending for fill_rate, descending for costs/time)
        reverse_sort = metric in ['execution_time', 'commission', 'slippage']
        order_performances.sort(key=lambda x: x['metric_value'], reverse=reverse_sort)
        
        return order_performances[:limit]


# Global order tracker instance
_order_tracker = None


def get_order_tracker() -> OrderTracker:
    """Get global order tracker."""
    global _order_tracker
    if _order_tracker is None:
        _order_tracker = OrderTracker()
        _order_tracker.start_monitoring()
    return _order_tracker