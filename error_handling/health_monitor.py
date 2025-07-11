"""
System health monitoring and diagnostics.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import psutil
import os
from abc import ABC, abstractmethod

from utils.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"


class ComponentType(Enum):
    """Types of system components."""
    DATABASE = "database"
    API_SERVICE = "api_service"
    DATA_FEED = "data_feed"
    TRADING_ENGINE = "trading_engine"
    RISK_MANAGER = "risk_manager"
    PORTFOLIO_MANAGER = "portfolio_manager"
    ORDER_MANAGER = "order_manager"
    STRATEGY_ENGINE = "strategy_engine"
    NOTIFICATION_SERVICE = "notification_service"
    FILE_SYSTEM = "file_system"
    NETWORK = "network"
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: Union[int, float, Decimal, str, bool]
    unit: Optional[str] = None
    threshold_warning: Optional[Union[int, float, Decimal]] = None
    threshold_critical: Optional[Union[int, float, Decimal]] = None
    higher_is_better: bool = True
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_status(self) -> HealthStatus:
        """Get status based on thresholds."""
        if isinstance(self.value, (bool, str)):
            return HealthStatus.HEALTHY if self.value else HealthStatus.CRITICAL
        
        if not isinstance(self.value, (int, float, Decimal)):
            return HealthStatus.UNKNOWN
        
        value = float(self.value)
        
        if self.threshold_critical is not None:
            critical_threshold = float(self.threshold_critical)
            if self.higher_is_better:
                if value <= critical_threshold:
                    return HealthStatus.CRITICAL
            else:
                if value >= critical_threshold:
                    return HealthStatus.CRITICAL
        
        if self.threshold_warning is not None:
            warning_threshold = float(self.threshold_warning)
            if self.higher_is_better:
                if value <= warning_threshold:
                    return HealthStatus.WARNING
            else:
                if value >= warning_threshold:
                    return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    component_id: str
    component_type: ComponentType
    status: HealthStatus
    last_check: datetime
    
    # Metrics
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    
    # Status details
    message: Optional[str] = None
    error_count: int = 0
    uptime: Optional[timedelta] = None
    response_time: Optional[timedelta] = None
    
    # Configuration
    check_interval: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    timeout: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    
    # Metadata
    version: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def add_metric(self, metric: HealthMetric) -> None:
        """Add a health metric."""
        self.metrics[metric.name] = metric
        
        # Update overall status based on worst metric
        metric_status = metric.get_status()
        if metric_status.value == "critical":
            self.status = HealthStatus.CRITICAL
        elif metric_status.value == "warning" and self.status != HealthStatus.CRITICAL:
            self.status = HealthStatus.WARNING
    
    def get_overall_status(self) -> HealthStatus:
        """Calculate overall status from all metrics."""
        if not self.metrics:
            return self.status
        
        has_critical = any(m.get_status() == HealthStatus.CRITICAL for m in self.metrics.values())
        has_warning = any(m.get_status() == HealthStatus.WARNING for m in self.metrics.values())
        
        if has_critical:
            return HealthStatus.CRITICAL
        elif has_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component_id': self.component_id,
            'component_type': self.component_type.value,
            'status': self.status.value,
            'last_check': self.last_check.isoformat(),
            'message': self.message,
            'error_count': self.error_count,
            'uptime': str(self.uptime) if self.uptime else None,
            'response_time': str(self.response_time) if self.response_time else None,
            'metrics': {name: {
                'value': metric.value,
                'unit': metric.unit,
                'status': metric.get_status().value,
                'timestamp': metric.timestamp.isoformat()
            } for name, metric in self.metrics.items()},
            'version': self.version,
            'dependencies': self.dependencies,
            'tags': self.tags
        }


class HealthCheck(ABC):
    """Base class for health checks."""
    
    def __init__(self, component_id: str, component_type: ComponentType):
        self.component_id = component_id
        self.component_type = component_type
        self.enabled = True
    
    @abstractmethod
    def check_health(self) -> ComponentHealth:
        """Perform health check and return status."""
        pass
    
    def is_enabled(self) -> bool:
        """Check if health check is enabled."""
        return self.enabled
    
    def enable(self) -> None:
        """Enable health check."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable health check."""
        self.enabled = False


class SystemResourceHealthCheck(HealthCheck):
    """Health check for system resources (CPU, Memory, Disk)."""
    
    def __init__(self):
        super().__init__("system_resources", ComponentType.CPU)
        
        # Thresholds
        self.cpu_warning_threshold = 80.0  # %
        self.cpu_critical_threshold = 95.0  # %
        self.memory_warning_threshold = 80.0  # %
        self.memory_critical_threshold = 90.0  # %
        self.disk_warning_threshold = 80.0  # %
        self.disk_critical_threshold = 90.0  # %
    
    def check_health(self) -> ComponentHealth:
        """Check system resource health."""
        health = ComponentHealth(
            component_id=self.component_id,
            component_type=self.component_type,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now()
        )
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_metric = HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="%",
                threshold_warning=self.cpu_warning_threshold,
                threshold_critical=self.cpu_critical_threshold,
                higher_is_better=False
            )
            health.add_metric(cpu_metric)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_metric = HealthMetric(
                name="memory_usage",
                value=memory.percent,
                unit="%",
                threshold_warning=self.memory_warning_threshold,
                threshold_critical=self.memory_critical_threshold,
                higher_is_better=False
            )
            health.add_metric(memory_metric)
            
            # Available memory
            available_gb = memory.available / (1024**3)
            memory_available_metric = HealthMetric(
                name="memory_available",
                value=round(available_gb, 2),
                unit="GB",
                threshold_warning=2.0,
                threshold_critical=1.0,
                higher_is_better=True
            )
            health.add_metric(memory_available_metric)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_metric = HealthMetric(
                name="disk_usage",
                value=round(disk_percent, 2),
                unit="%",
                threshold_warning=self.disk_warning_threshold,
                threshold_critical=self.disk_critical_threshold,
                higher_is_better=False
            )
            health.add_metric(disk_metric)
            
            # Process count
            process_count = len(psutil.pids())
            process_metric = HealthMetric(
                name="process_count",
                value=process_count,
                unit="processes",
                threshold_warning=500,
                threshold_critical=1000,
                higher_is_better=False
            )
            health.add_metric(process_metric)
            
            # Load average (if available)
            try:
                load_avg = os.getloadavg()[0]  # 1-minute load average
                load_metric = HealthMetric(
                    name="load_average",
                    value=round(load_avg, 2),
                    unit="load",
                    threshold_warning=psutil.cpu_count() * 0.8,
                    threshold_critical=psutil.cpu_count() * 1.2,
                    higher_is_better=False
                )
                health.add_metric(load_metric)
            except (OSError, AttributeError):
                # getloadavg not available on all platforms
                pass
            
            # Update overall status
            health.status = health.get_overall_status()
            
            if health.status == HealthStatus.CRITICAL:
                health.message = "System resources critically low"
            elif health.status == HealthStatus.WARNING:
                health.message = "System resources under pressure"
            else:
                health.message = "System resources healthy"
            
        except Exception as e:
            health.status = HealthStatus.UNKNOWN
            health.message = f"Failed to check system resources: {str(e)}"
            logger.error(f"System resource health check failed: {e}")
        
        return health


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity."""
    
    def __init__(self, connection_string: Optional[str] = None):
        super().__init__("database", ComponentType.DATABASE)
        self.connection_string = connection_string
    
    def check_health(self) -> ComponentHealth:
        """Check database health."""
        health = ComponentHealth(
            component_id=self.component_id,
            component_type=self.component_type,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now()
        )
        
        try:
            start_time = time.time()
            
            # Simulate database connection check
            # In real implementation, this would connect to actual database
            if self.connection_string:
                # Simulate connection attempt
                time.sleep(0.01)  # Simulate connection time
                
                response_time = time.time() - start_time
                health.response_time = timedelta(seconds=response_time)
                
                # Response time metric
                response_metric = HealthMetric(
                    name="response_time",
                    value=round(response_time * 1000, 2),  # Convert to ms
                    unit="ms",
                    threshold_warning=100,
                    threshold_critical=500,
                    higher_is_better=False
                )
                health.add_metric(response_metric)
                
                # Connection status
                connection_metric = HealthMetric(
                    name="connection_status",
                    value=True,
                    unit="boolean"
                )
                health.add_metric(connection_metric)
                
                health.status = HealthStatus.HEALTHY
                health.message = "Database connection successful"
            else:
                health.status = HealthStatus.WARNING
                health.message = "No database connection string configured"
            
        except Exception as e:
            health.status = HealthStatus.CRITICAL
            health.message = f"Database connection failed: {str(e)}"
            health.error_count += 1
            logger.error(f"Database health check failed: {e}")
        
        return health


class APIServiceHealthCheck(HealthCheck):
    """Health check for API services."""
    
    def __init__(self, service_name: str, endpoint_url: Optional[str] = None):
        super().__init__(f"api_{service_name}", ComponentType.API_SERVICE)
        self.service_name = service_name
        self.endpoint_url = endpoint_url
    
    def check_health(self) -> ComponentHealth:
        """Check API service health."""
        health = ComponentHealth(
            component_id=self.component_id,
            component_type=self.component_type,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now()
        )
        
        try:
            if self.endpoint_url:
                start_time = time.time()
                
                # Simulate API call
                # In real implementation, this would make actual HTTP request
                time.sleep(0.05)  # Simulate API response time
                
                response_time = time.time() - start_time
                health.response_time = timedelta(seconds=response_time)
                
                # Response time metric
                response_metric = HealthMetric(
                    name="response_time",
                    value=round(response_time * 1000, 2),
                    unit="ms",
                    threshold_warning=200,
                    threshold_critical=1000,
                    higher_is_better=False
                )
                health.add_metric(response_metric)
                
                # Service availability
                availability_metric = HealthMetric(
                    name="availability",
                    value=True,
                    unit="boolean"
                )
                health.add_metric(availability_metric)
                
                health.status = HealthStatus.HEALTHY
                health.message = f"{self.service_name} API service is healthy"
            else:
                health.status = HealthStatus.WARNING
                health.message = f"No endpoint configured for {self.service_name}"
            
        except Exception as e:
            health.status = HealthStatus.CRITICAL
            health.message = f"{self.service_name} API service failed: {str(e)}"
            health.error_count += 1
            logger.error(f"API service health check failed for {self.service_name}: {e}")
        
        return health


class HealthMonitor:
    """Central health monitoring system."""
    
    def __init__(self, check_interval: timedelta = timedelta(minutes=1)):
        self.check_interval = check_interval
        self.health_checks: Dict[str, HealthCheck] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.health_history: List[Dict[str, ComponentHealth]] = []
        
        # Configuration
        self.max_history_size = 1440  # 24 hours of minute-by-minute data
        self.alert_on_status_change = True
        self.recovery_detection = True
        
        # State
        self.monitoring_active = False
        self.last_check_time = None
        
        # Statistics
        self.monitoring_stats = {
            'total_checks': 0,
            'healthy_components': 0,
            'warning_components': 0,
            'critical_components': 0,
            'unknown_components': 0,
            'last_full_check': None,
            'average_check_duration': 0.0
        }
        
        # Threading
        self._lock = threading.Lock()
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        # Initialize default health checks
        self._initialize_default_checks()
        
        logger.info("Health monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        with self._lock:
            if self.monitoring_active:
                logger.warning("Health monitoring is already active")
                return
            
            self.monitoring_active = True
            self._stop_event.clear()
            
            # Start monitoring thread
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="HealthMonitor",
                daemon=True
            )
            self._monitor_thread.start()
            
            logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        with self._lock:
            if not self.monitoring_active:
                logger.warning("Health monitoring is not active")
                return
            
            self.monitoring_active = False
            self._stop_event.set()
            
            # Wait for thread to finish
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=10)
            
            logger.info("Health monitoring stopped")
    
    def check_all_components(self) -> Dict[str, ComponentHealth]:
        """Perform health check on all registered components."""
        with self._lock:
            start_time = time.time()
            results = {}
            
            for check_id, health_check in self.health_checks.items():
                if not health_check.is_enabled():
                    continue
                
                try:
                    component_health = health_check.check_health()
                    results[check_id] = component_health
                    self.component_health[check_id] = component_health
                    
                except Exception as e:
                    logger.error(f"Health check failed for {check_id}: {e}")
                    
                    # Create error health status
                    error_health = ComponentHealth(
                        component_id=check_id,
                        component_type=health_check.component_type,
                        status=HealthStatus.UNKNOWN,
                        last_check=datetime.now(),
                        message=f"Health check failed: {str(e)}",
                        error_count=1
                    )
                    results[check_id] = error_health
                    self.component_health[check_id] = error_health
            
            # Update statistics
            self._update_monitoring_statistics(results)
            
            # Store in history
            self.health_history.append(results.copy())
            self._cleanup_history()
            
            # Calculate check duration
            check_duration = time.time() - start_time
            self.monitoring_stats['average_check_duration'] = (
                (self.monitoring_stats['average_check_duration'] * 
                 (self.monitoring_stats['total_checks'] - 1) + check_duration) /
                self.monitoring_stats['total_checks']
            )
            
            self.last_check_time = datetime.now()
            self.monitoring_stats['last_full_check'] = self.last_check_time.isoformat()
            
            logger.debug(f"Health check completed in {check_duration:.2f} seconds")
            
            return results
    
    def register_health_check(self, health_check: HealthCheck) -> None:
        """Register a new health check."""
        with self._lock:
            self.health_checks[health_check.component_id] = health_check
            logger.info(f"Registered health check: {health_check.component_id}")
    
    def unregister_health_check(self, component_id: str) -> bool:
        """Unregister a health check."""
        with self._lock:
            if component_id in self.health_checks:
                del self.health_checks[component_id]
                if component_id in self.component_health:
                    del self.component_health[component_id]
                logger.info(f"Unregistered health check: {component_id}")
                return True
            return False
    
    def get_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Get health status of a specific component."""
        with self._lock:
            return self.component_health.get(component_id)
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        with self._lock:
            if not self.component_health:
                return {
                    'status': HealthStatus.UNKNOWN.value,
                    'message': 'No health data available',
                    'components': {},
                    'statistics': self.monitoring_stats
                }
            
            # Determine overall status
            component_statuses = [health.status for health in self.component_health.values()]
            
            if HealthStatus.CRITICAL in component_statuses:
                overall_status = HealthStatus.CRITICAL
                message = "One or more components are in critical state"
            elif HealthStatus.WARNING in component_statuses:
                overall_status = HealthStatus.WARNING
                message = "One or more components have warnings"
            elif HealthStatus.UNKNOWN in component_statuses:
                overall_status = HealthStatus.DEGRADED
                message = "Some components have unknown status"
            else:
                overall_status = HealthStatus.HEALTHY
                message = "All components are healthy"
            
            # Component summary
            component_summary = {}
            for comp_id, health in self.component_health.items():
                component_summary[comp_id] = {
                    'status': health.status.value,
                    'last_check': health.last_check.isoformat(),
                    'message': health.message
                }
            
            return {
                'status': overall_status.value,
                'message': message,
                'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
                'components': component_summary,
                'statistics': self.monitoring_stats
            }
    
    def get_health_trends(self, component_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health trends for a component over time."""
        with self._lock:
            trends = []
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            for snapshot in self.health_history:
                if component_id in snapshot:
                    health = snapshot[component_id]
                    if health.last_check >= cutoff_time:
                        trends.append({
                            'timestamp': health.last_check.isoformat(),
                            'status': health.status.value,
                            'metrics': {
                                name: {
                                    'value': metric.value,
                                    'status': metric.get_status().value
                                }
                                for name, metric in health.metrics.items()
                            }
                        })
            
            return sorted(trends, key=lambda x: x['timestamp'])
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        with self._lock:
            stats = self.monitoring_stats.copy()
            stats['registered_checks'] = len(self.health_checks)
            stats['active_checks'] = len([c for c in self.health_checks.values() if c.is_enabled()])
            stats['monitoring_active'] = self.monitoring_active
            return stats
    
    # Private methods
    
    def _initialize_default_checks(self) -> None:
        """Initialize default health checks."""
        default_checks = [
            SystemResourceHealthCheck(),
            DatabaseHealthCheck(),
            APIServiceHealthCheck("market_data"),
            APIServiceHealthCheck("trading_api")
        ]
        
        for check in default_checks:
            self.health_checks[check.component_id] = check
        
        logger.info(f"Initialized {len(default_checks)} default health checks")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Health monitoring loop started")
        
        while not self._stop_event.is_set():
            try:
                # Perform health checks
                self.check_all_components()
                
                # Wait for next check interval
                if not self._stop_event.wait(self.check_interval.total_seconds()):
                    continue  # Continue if timeout (normal operation)
                else:
                    break  # Stop event was set
                    
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                # Continue monitoring even if there's an error
                time.sleep(10)  # Brief pause before retrying
        
        logger.info("Health monitoring loop stopped")
    
    def _update_monitoring_statistics(self, results: Dict[str, ComponentHealth]) -> None:
        """Update monitoring statistics."""
        self.monitoring_stats['total_checks'] += 1
        
        # Count components by status
        status_counts = {
            'healthy_components': 0,
            'warning_components': 0,
            'critical_components': 0,
            'unknown_components': 0
        }
        
        for health in results.values():
            if health.status == HealthStatus.HEALTHY:
                status_counts['healthy_components'] += 1
            elif health.status == HealthStatus.WARNING:
                status_counts['warning_components'] += 1
            elif health.status == HealthStatus.CRITICAL:
                status_counts['critical_components'] += 1
            else:
                status_counts['unknown_components'] += 1
        
        self.monitoring_stats.update(status_counts)
    
    def _cleanup_history(self) -> None:
        """Clean up old health history data."""
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]


# Global health monitor instance
_health_monitor = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor