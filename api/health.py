"""
Health check endpoints and monitoring.
"""

from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import psutil
import asyncio

from configs.settings import get_settings
from infrastructure.container import get_container


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    uptime_seconds: float
    version: str
    environment: str
    checks: Dict[str, Dict[str, Any]]


class ComponentHealth(BaseModel):
    """Individual component health."""
    status: str
    message: str
    response_time_ms: float
    last_check: datetime


router = APIRouter(prefix="/health", tags=["health"])


class HealthChecker:
    """Health check service."""
    
    def __init__(self):
        self.settings = get_settings()
        self.start_time = datetime.now()
        self.container = get_container()
    
    async def check_database(self) -> ComponentHealth:
        """Check database connectivity."""
        start_time = datetime.now()
        try:
            # TODO: Implement actual database check
            # For now, simulate a check
            await asyncio.sleep(0.01)
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return ComponentHealth(
                status="healthy",
                message="Database connection successful",
                response_time_ms=response_time,
                last_check=datetime.now()
            )
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return ComponentHealth(
                status="unhealthy",
                message=f"Database connection failed: {str(e)}",
                response_time_ms=response_time,
                last_check=datetime.now()
            )
    
    async def check_alpaca_api(self) -> ComponentHealth:
        """Check Alpaca API connectivity."""
        start_time = datetime.now()
        try:
            # TODO: Implement actual Alpaca API check
            # For now, simulate a check
            await asyncio.sleep(0.05)
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return ComponentHealth(
                status="healthy",
                message="Alpaca API connection successful",
                response_time_ms=response_time,
                last_check=datetime.now()
            )
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            return ComponentHealth(
                status="unhealthy",
                message=f"Alpaca API connection failed: {str(e)}",
                response_time_ms=response_time,
                last_check=datetime.now()
            )
    
    def check_system_resources(self) -> ComponentHealth:
        """Check system resources."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Determine status based on thresholds
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 95:
                status = "unhealthy"
                message = f"High resource usage: CPU {cpu_percent}%, Memory {memory_percent}%, Disk {disk_percent}%"
            elif cpu_percent > 70 or memory_percent > 70 or disk_percent > 85:
                status = "degraded"
                message = f"Moderate resource usage: CPU {cpu_percent}%, Memory {memory_percent}%, Disk {disk_percent}%"
            else:
                status = "healthy"
                message = f"Resource usage normal: CPU {cpu_percent}%, Memory {memory_percent}%, Disk {disk_percent}%"
            
            return ComponentHealth(
                status=status,
                message=message,
                response_time_ms=1000,  # Approximate time for resource check
                last_check=datetime.now()
            )
        except Exception as e:
            return ComponentHealth(
                status="unhealthy",
                message=f"System resource check failed: {str(e)}",
                response_time_ms=0,
                last_check=datetime.now()
            )
    
    async def get_overall_health(self) -> HealthStatus:
        """Get overall system health."""
        checks = {}
        
        # Run all health checks
        db_health = await self.check_database()
        alpaca_health = await self.check_alpaca_api()
        system_health = self.check_system_resources()
        
        checks["database"] = db_health.dict()
        checks["alpaca_api"] = alpaca_health.dict()
        checks["system_resources"] = system_health.dict()
        
        # Determine overall status
        all_checks = [db_health, alpaca_health, system_health]
        unhealthy_count = sum(1 for check in all_checks if check.status == "unhealthy")
        degraded_count = sum(1 for check in all_checks if check.status == "degraded")
        
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Calculate uptime
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return HealthStatus(
            status=overall_status,
            timestamp=datetime.now(),
            uptime_seconds=uptime,
            version=self.settings.app_version,
            environment=getattr(self.settings, 'environment', 'unknown'),
            checks=checks
        )


# Global health checker instance
health_checker = HealthChecker()


@router.get("/", response_model=HealthStatus)
async def health_check():
    """Get overall system health."""
    return await health_checker.get_overall_health()


@router.get("/live")
async def liveness_probe():
    """Kubernetes liveness probe - basic service availability."""
    return {"status": "ok", "timestamp": datetime.now()}


@router.get("/ready")
async def readiness_probe():
    """Kubernetes readiness probe - service ready to handle traffic."""
    health = await health_checker.get_overall_health()
    
    if health.status == "unhealthy":
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready", "timestamp": datetime.now()}


@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus-style metrics endpoint."""
    health = await health_checker.get_overall_health()
    
    # Generate Prometheus-style metrics
    metrics = []
    
    # Health status metrics (1 = healthy, 0.5 = degraded, 0 = unhealthy)
    status_map = {"healthy": 1.0, "degraded": 0.5, "unhealthy": 0.0}
    overall_health_value = status_map.get(health.status, 0.0)
    
    metrics.append(f'algua_health_status{{component="overall"}} {overall_health_value}')
    
    # Component health metrics
    for component, check_data in health.checks.items():
        component_health_value = status_map.get(check_data["status"], 0.0)
        metrics.append(f'algua_health_status{{component="{component}"}} {component_health_value}')
        metrics.append(f'algua_health_response_time_ms{{component="{component}"}} {check_data["response_time_ms"]}')
    
    # Uptime metric
    metrics.append(f'algua_uptime_seconds {health.uptime_seconds}')
    
    # System metrics if available
    try:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        metrics.append(f'algua_cpu_usage_percent {cpu_percent}')
        metrics.append(f'algua_memory_usage_percent {memory_percent}')
    except:
        pass
    
    return {"metrics": "\\n".join(metrics), "content_type": "text/plain"}