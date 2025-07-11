"""
Performance monitoring and metrics collection system.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import statistics
import psutil
import os
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import functools

from utils.logging import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class MetricUnit(Enum):
    """Metric units."""
    MILLISECONDS = "ms"
    SECONDS = "s"
    MINUTES = "min"
    HOURS = "h"
    BYTES = "bytes"
    KILOBYTES = "kb"
    MEGABYTES = "mb"
    GIGABYTES = "gb"
    PERCENT = "%"
    COUNT = "count"
    REQUESTS_PER_SECOND = "req/s"
    TRANSACTIONS_PER_SECOND = "tps"


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    name: str
    metric_type: MetricType
    value: Union[int, float, Decimal]
    unit: MetricUnit
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Additional metadata
    labels: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None
    
    # Statistical data (for histograms)
    samples: List[float] = field(default_factory=list)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    avg_value: Optional[float] = None
    percentiles: Dict[str, float] = field(default_factory=dict)
    
    def add_sample(self, value: float) -> None:
        """Add a sample to the metric."""
        self.samples.append(value)
        self.value = value  # Update current value
        
        # Update statistics
        if self.samples:
            self.min_value = min(self.samples)
            self.max_value = max(self.samples)
            self.avg_value = sum(self.samples) / len(self.samples)
            
            # Calculate percentiles
            if len(self.samples) >= 2:
                sorted_samples = sorted(self.samples)
                self.percentiles = {
                    'p50': statistics.median(sorted_samples),
                    'p90': sorted_samples[int(len(sorted_samples) * 0.9)],
                    'p95': sorted_samples[int(len(sorted_samples) * 0.95)],
                    'p99': sorted_samples[int(len(sorted_samples) * 0.99)]
                }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.metric_type.value,
            'value': float(self.value),
            'unit': self.unit.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels,
            'description': self.description,
            'statistics': {
                'min': self.min_value,
                'max': self.max_value,
                'avg': self.avg_value,
                'samples_count': len(self.samples),
                'percentiles': self.percentiles
            }
        }


@dataclass
class LatencyMetrics:
    """Latency-specific metrics."""
    operation: str
    duration: float  # in milliseconds
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Context
    component: Optional[str] = None
    method: Optional[str] = None
    status: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation': self.operation,
            'duration': self.duration,
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'method': self.method,
            'status': self.status,
            'metadata': self.metadata
        }


class LatencyTracker:
    """Tracks latency metrics for operations."""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.latency_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def record_latency(
        self,
        operation: str,
        duration: float,
        component: Optional[str] = None,
        method: Optional[str] = None,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record latency for an operation."""
        with self._lock:
            latency = LatencyMetrics(
                operation=operation,
                duration=duration,
                component=component,
                method=method,
                status=status,
                metadata=metadata or {}
            )
            
            self.latency_data[operation].append(latency)
            self.operation_counts[operation] += 1
    
    def get_latency_statistics(self, operation: str) -> Dict[str, Any]:
        """Get latency statistics for an operation."""
        with self._lock:
            if operation not in self.latency_data:
                return {}
            
            latencies = [entry.duration for entry in self.latency_data[operation]]
            
            if not latencies:
                return {}
            
            return {
                'operation': operation,
                'count': len(latencies),
                'min': min(latencies),
                'max': max(latencies),
                'avg': sum(latencies) / len(latencies),
                'median': statistics.median(latencies),
                'p90': latencies[int(len(latencies) * 0.9)] if len(latencies) > 1 else latencies[0],
                'p95': latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
                'p99': latencies[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0],
                'total_operations': self.operation_counts[operation]
            }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all operations."""
        with self._lock:
            return {op: self.get_latency_statistics(op) for op in self.latency_data.keys()}
    
    def clear_data(self, operation: Optional[str] = None) -> None:
        """Clear latency data."""
        with self._lock:
            if operation:
                if operation in self.latency_data:
                    self.latency_data[operation].clear()
                    self.operation_counts[operation] = 0
            else:
                self.latency_data.clear()
                self.operation_counts.clear()


class ThroughputMonitor:
    """Monitors throughput metrics."""
    
    def __init__(self, window_size: int = 300):  # 5 minutes
        self.window_size = window_size
        self.events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = threading.Lock()
    
    def record_event(self, event_type: str, count: int = 1) -> None:
        """Record an event occurrence."""
        with self._lock:
            timestamp = time.time()
            self.events[event_type].append((timestamp, count))
    
    def get_throughput(self, event_type: str, window_seconds: int = 60) -> float:
        """Get throughput for an event type over a time window."""
        with self._lock:
            if event_type not in self.events:
                return 0.0
            
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            
            total_count = 0
            for timestamp, count in self.events[event_type]:
                if timestamp >= cutoff_time:
                    total_count += count
            
            return total_count / window_seconds  # events per second
    
    def get_all_throughput(self, window_seconds: int = 60) -> Dict[str, float]:
        """Get throughput for all event types."""
        with self._lock:
            return {event_type: self.get_throughput(event_type, window_seconds) 
                   for event_type in self.events.keys()}


class SystemMetricsCollector:
    """Collects system-level performance metrics."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.boot_time = psutil.boot_time()
    
    def collect_cpu_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Collect CPU metrics."""
        metrics = {}
        
        # System CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics['system_cpu_usage'] = PerformanceMetrics(
            name='system_cpu_usage',
            metric_type=MetricType.GAUGE,
            value=cpu_percent,
            unit=MetricUnit.PERCENT,
            description='System CPU usage percentage'
        )
        
        # Process CPU usage
        process_cpu = self.process.cpu_percent()
        metrics['process_cpu_usage'] = PerformanceMetrics(
            name='process_cpu_usage',
            metric_type=MetricType.GAUGE,
            value=process_cpu,
            unit=MetricUnit.PERCENT,
            description='Process CPU usage percentage'
        )
        
        # CPU count
        metrics['cpu_count'] = PerformanceMetrics(
            name='cpu_count',
            metric_type=MetricType.GAUGE,
            value=psutil.cpu_count(),
            unit=MetricUnit.COUNT,
            description='Number of CPU cores'
        )
        
        # Load average (if available)
        try:
            load_avg = os.getloadavg()
            metrics['load_average_1m'] = PerformanceMetrics(
                name='load_average_1m',
                metric_type=MetricType.GAUGE,
                value=load_avg[0],
                unit=MetricUnit.COUNT,
                description='1-minute load average'
            )
        except (OSError, AttributeError):
            pass
        
        return metrics
    
    def collect_memory_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Collect memory metrics."""
        metrics = {}
        
        # System memory
        memory = psutil.virtual_memory()
        metrics['system_memory_usage'] = PerformanceMetrics(
            name='system_memory_usage',
            metric_type=MetricType.GAUGE,
            value=memory.percent,
            unit=MetricUnit.PERCENT,
            description='System memory usage percentage'
        )
        
        metrics['system_memory_available'] = PerformanceMetrics(
            name='system_memory_available',
            metric_type=MetricType.GAUGE,
            value=memory.available / (1024**3),  # Convert to GB
            unit=MetricUnit.GIGABYTES,
            description='Available system memory'
        )
        
        # Process memory
        process_memory = self.process.memory_info()
        metrics['process_memory_rss'] = PerformanceMetrics(
            name='process_memory_rss',
            metric_type=MetricType.GAUGE,
            value=process_memory.rss / (1024**2),  # Convert to MB
            unit=MetricUnit.MEGABYTES,
            description='Process RSS memory'
        )
        
        metrics['process_memory_vms'] = PerformanceMetrics(
            name='process_memory_vms',
            metric_type=MetricType.GAUGE,
            value=process_memory.vms / (1024**2),  # Convert to MB
            unit=MetricUnit.MEGABYTES,
            description='Process VMS memory'
        )
        
        return metrics
    
    def collect_disk_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Collect disk metrics."""
        metrics = {}
        
        # Disk usage
        disk = psutil.disk_usage('/')
        metrics['disk_usage'] = PerformanceMetrics(
            name='disk_usage',
            metric_type=MetricType.GAUGE,
            value=(disk.used / disk.total) * 100,
            unit=MetricUnit.PERCENT,
            description='Disk usage percentage'
        )
        
        metrics['disk_free'] = PerformanceMetrics(
            name='disk_free',
            metric_type=MetricType.GAUGE,
            value=disk.free / (1024**3),  # Convert to GB
            unit=MetricUnit.GIGABYTES,
            description='Free disk space'
        )
        
        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics['disk_read_bytes'] = PerformanceMetrics(
                    name='disk_read_bytes',
                    metric_type=MetricType.COUNTER,
                    value=disk_io.read_bytes,
                    unit=MetricUnit.BYTES,
                    description='Total disk read bytes'
                )
                
                metrics['disk_write_bytes'] = PerformanceMetrics(
                    name='disk_write_bytes',
                    metric_type=MetricType.COUNTER,
                    value=disk_io.write_bytes,
                    unit=MetricUnit.BYTES,
                    description='Total disk write bytes'
                )
        except Exception:
            pass
        
        return metrics
    
    def collect_network_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Collect network metrics."""
        metrics = {}
        
        try:
            network_io = psutil.net_io_counters()
            if network_io:
                metrics['network_bytes_sent'] = PerformanceMetrics(
                    name='network_bytes_sent',
                    metric_type=MetricType.COUNTER,
                    value=network_io.bytes_sent,
                    unit=MetricUnit.BYTES,
                    description='Total network bytes sent'
                )
                
                metrics['network_bytes_recv'] = PerformanceMetrics(
                    name='network_bytes_recv',
                    metric_type=MetricType.COUNTER,
                    value=network_io.bytes_recv,
                    unit=MetricUnit.BYTES,
                    description='Total network bytes received'
                )
                
                metrics['network_packets_sent'] = PerformanceMetrics(
                    name='network_packets_sent',
                    metric_type=MetricType.COUNTER,
                    value=network_io.packets_sent,
                    unit=MetricUnit.COUNT,
                    description='Total network packets sent'
                )
                
                metrics['network_packets_recv'] = PerformanceMetrics(
                    name='network_packets_recv',
                    metric_type=MetricType.COUNTER,
                    value=network_io.packets_recv,
                    unit=MetricUnit.COUNT,
                    description='Total network packets received'
                )
        except Exception:
            pass
        
        return metrics
    
    def collect_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Collect all system metrics."""
        all_metrics = {}
        
        all_metrics.update(self.collect_cpu_metrics())
        all_metrics.update(self.collect_memory_metrics())
        all_metrics.update(self.collect_disk_metrics())
        all_metrics.update(self.collect_network_metrics())
        
        return all_metrics


class PerformanceMonitor:
    """Central performance monitoring system."""
    
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.metrics_history: List[Dict[str, PerformanceMetrics]] = []
        
        # Specialized monitors
        self.latency_tracker = LatencyTracker()
        self.throughput_monitor = ThroughputMonitor()
        self.system_collector = SystemMetricsCollector()
        
        # Configuration
        self.max_history_size = 1440  # 24 hours of minute-by-minute data
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time': 1000.0  # ms
        }
        
        # State
        self.monitoring_active = False
        self.last_collection_time = None
        
        # Statistics
        self.monitoring_stats = {
            'total_collections': 0,
            'total_metrics': 0,
            'average_collection_time': 0.0,
            'last_collection': None
        }
        
        # Threading
        self._lock = threading.Lock()
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        logger.info("Performance monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start continuous performance monitoring."""
        with self._lock:
            if self.monitoring_active:
                logger.warning("Performance monitoring is already active")
                return
            
            self.monitoring_active = True
            self._stop_event.clear()
            
            # Start monitoring thread
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="PerformanceMonitor",
                daemon=True
            )
            self._monitor_thread.start()
            
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous performance monitoring."""
        with self._lock:
            if not self.monitoring_active:
                logger.warning("Performance monitoring is not active")
                return
            
            self.monitoring_active = False
            self._stop_event.set()
            
            # Wait for thread to finish
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=10)
            
            logger.info("Performance monitoring stopped")
    
    def collect_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Collect all performance metrics."""
        with self._lock:
            start_time = time.time()
            
            # Collect system metrics
            system_metrics = self.system_collector.collect_all_metrics()
            
            # Update metrics storage
            self.metrics.update(system_metrics)
            
            # Store in history
            self.metrics_history.append(system_metrics.copy())
            self._cleanup_history()
            
            # Update statistics
            collection_time = time.time() - start_time
            self.monitoring_stats['total_collections'] += 1
            self.monitoring_stats['total_metrics'] = len(self.metrics)
            self.monitoring_stats['average_collection_time'] = (
                (self.monitoring_stats['average_collection_time'] * 
                 (self.monitoring_stats['total_collections'] - 1) + collection_time) /
                self.monitoring_stats['total_collections']
            )
            
            self.last_collection_time = datetime.now()
            self.monitoring_stats['last_collection'] = self.last_collection_time.isoformat()
            
            logger.debug(f"Metrics collection completed in {collection_time:.2f} seconds")
            
            return system_metrics
    
    def record_operation_latency(
        self,
        operation: str,
        duration: float,
        component: Optional[str] = None,
        method: Optional[str] = None,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record latency for an operation."""
        self.latency_tracker.record_latency(
            operation=operation,
            duration=duration,
            component=component,
            method=method,
            status=status,
            metadata=metadata
        )
    
    def record_throughput_event(self, event_type: str, count: int = 1) -> None:
        """Record a throughput event."""
        self.throughput_monitor.record_event(event_type, count)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self._lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'system_metrics': {name: metric.to_dict() for name, metric in self.metrics.items()},
                'latency_statistics': self.latency_tracker.get_all_statistics(),
                'throughput_statistics': self.throughput_monitor.get_all_throughput(),
                'monitoring_stats': self.monitoring_stats
            }
    
    def get_metric_trends(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get trends for a specific metric over time."""
        with self._lock:
            trends = []
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            for snapshot in self.metrics_history:
                if metric_name in snapshot:
                    metric = snapshot[metric_name]
                    if metric.timestamp >= cutoff_time:
                        trends.append({
                            'timestamp': metric.timestamp.isoformat(),
                            'value': float(metric.value),
                            'unit': metric.unit.value
                        })
            
            return sorted(trends, key=lambda x: x['timestamp'])
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and alerts."""
        with self._lock:
            summary = {
                'status': 'healthy',
                'alerts': [],
                'metrics_summary': {},
                'last_updated': self.last_collection_time.isoformat() if self.last_collection_time else None
            }
            
            # Check for performance issues
            for metric_name, metric in self.metrics.items():
                if metric_name in self.alert_thresholds:
                    threshold = self.alert_thresholds[metric_name]
                    value = float(metric.value)
                    
                    if value > threshold:
                        summary['alerts'].append({
                            'metric': metric_name,
                            'value': value,
                            'threshold': threshold,
                            'severity': 'warning' if value < threshold * 1.2 else 'critical'
                        })
                        
                        if summary['status'] == 'healthy':
                            summary['status'] = 'warning'
            
            # Add metrics summary
            if self.metrics:
                summary['metrics_summary'] = {
                    'total_metrics': len(self.metrics),
                    'cpu_usage': self.metrics.get('system_cpu_usage', {}).value if 'system_cpu_usage' in self.metrics else 0,
                    'memory_usage': self.metrics.get('system_memory_usage', {}).value if 'system_memory_usage' in self.metrics else 0,
                    'disk_usage': self.metrics.get('disk_usage', {}).value if 'disk_usage' in self.metrics else 0
                }
            
            return summary
    
    def export_metrics(self, file_path: str, format: str = 'json') -> None:
        """Export metrics to file."""
        with self._lock:
            export_data = {
                'export_time': datetime.now().isoformat(),
                'current_metrics': self.get_current_metrics(),
                'monitoring_stats': self.monitoring_stats
            }
            
            if format.lower() == 'json':
                import json
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Metrics exported to {file_path}")
    
    # Private methods
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Performance monitoring loop started")
        
        while not self._stop_event.is_set():
            try:
                # Collect metrics
                self.collect_metrics()
                
                # Wait for next collection interval
                if not self._stop_event.wait(self.collection_interval):
                    continue  # Continue if timeout (normal operation)
                else:
                    break  # Stop event was set
                    
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                # Continue monitoring even if there's an error
                time.sleep(60)  # Brief pause before retrying
        
        logger.info("Performance monitoring loop stopped")
    
    def _cleanup_history(self) -> None:
        """Clean up old metrics history."""
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]


def performance_monitor(operation: str, component: Optional[str] = None):
    """Decorator for monitoring operation performance."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                # Record latency
                duration = (time.time() - start_time) * 1000  # Convert to ms
                monitor = get_performance_monitor()
                monitor.record_operation_latency(
                    operation=operation,
                    duration=duration,
                    component=component or func.__module__,
                    method=func.__name__,
                    status=status
                )
        
        return wrapper
    return decorator


# Global performance monitor instance
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor