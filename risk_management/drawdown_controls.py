"""
Drawdown monitoring and control system.
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from domain.value_objects import Symbol, Money
from .interfaces import RiskEvent, RiskLevel, RiskEventType
from .configuration import get_risk_config
from .event_system import publish_risk_event, RiskEventFactory
from utils.logging import get_logger

logger = get_logger(__name__)


class DrawdownSeverity(Enum):
    """Drawdown severity levels."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"


class RecoveryMode(Enum):
    """Portfolio recovery modes."""
    NORMAL = "normal"
    CONSERVATIVE = "conservative"
    DEFENSIVE = "defensive"
    HALT_TRADING = "halt_trading"


@dataclass
class DrawdownPeriod:
    """Drawdown period information."""
    start_date: datetime
    end_date: Optional[datetime]
    peak_value: Money
    trough_value: Money
    max_drawdown_pct: Decimal
    duration_days: int
    recovered: bool = False
    recovery_date: Optional[datetime] = None


@dataclass
class DrawdownMetrics:
    """Comprehensive drawdown metrics."""
    current_drawdown_pct: Decimal
    max_drawdown_pct: Decimal
    current_drawdown_duration_days: int
    max_drawdown_duration_days: int
    drawdown_severity: DrawdownSeverity
    recovery_mode: RecoveryMode
    underwater_curve: List[Tuple[datetime, Decimal]]
    consecutive_losing_days: int
    recovery_factor: Decimal
    time_to_recovery_estimate: Optional[int]  # days


class DrawdownController:
    """Drawdown monitoring and control system."""
    
    def __init__(self, config=None):
        self.config = config or get_risk_config()
        self.portfolio_values = deque(maxlen=1000)  # Store last 1000 portfolio values
        self.peak_value = Money(Decimal('0'))
        self.current_drawdown_start = None
        self.drawdown_periods: List[DrawdownPeriod] = []
        self.consecutive_losing_days = 0
        self.recovery_mode = RecoveryMode.NORMAL
        self.last_update = None
    
    def update_portfolio_value(
        self,
        portfolio_value: Money,
        timestamp: datetime = None
    ) -> DrawdownMetrics:
        """Update portfolio value and calculate drawdown metrics."""
        timestamp = timestamp or datetime.now()
        
        # Store portfolio value with timestamp
        self.portfolio_values.append((timestamp, portfolio_value))
        
        # Update peak value
        if portfolio_value.amount > self.peak_value.amount:
            self.peak_value = portfolio_value
            
            # End current drawdown period if we're in one
            if self.current_drawdown_start is not None:
                self._end_current_drawdown(timestamp, portfolio_value)
        
        # Calculate current drawdown
        current_drawdown_pct = self._calculate_current_drawdown(portfolio_value)
        
        # Check if we're starting a new drawdown
        if current_drawdown_pct > 0 and self.current_drawdown_start is None:
            self.current_drawdown_start = timestamp
            logger.info(f"New drawdown period started at {timestamp}")
        
        # Calculate comprehensive metrics
        metrics = self._calculate_drawdown_metrics(portfolio_value, timestamp)
        
        # Check for drawdown limit breaches
        self._check_drawdown_limits(metrics)
        
        # Update recovery mode based on drawdown severity
        self._update_recovery_mode(metrics)
        
        self.last_update = timestamp
        return metrics
    
    def _calculate_current_drawdown(self, portfolio_value: Money) -> Decimal:
        """Calculate current drawdown percentage."""
        if self.peak_value.amount == 0:
            return Decimal('0')
        
        drawdown_pct = (self.peak_value.amount - portfolio_value.amount) / self.peak_value.amount
        return max(drawdown_pct, Decimal('0'))
    
    def _calculate_drawdown_metrics(
        self,
        portfolio_value: Money,
        timestamp: datetime
    ) -> DrawdownMetrics:
        """Calculate comprehensive drawdown metrics."""
        current_drawdown_pct = self._calculate_current_drawdown(portfolio_value)
        
        # Calculate maximum drawdown from all periods
        max_drawdown_pct = max(
            [period.max_drawdown_pct for period in self.drawdown_periods] + [current_drawdown_pct]
        )
        
        # Calculate current drawdown duration
        current_duration_days = 0
        if self.current_drawdown_start:
            current_duration_days = (timestamp - self.current_drawdown_start).days
        
        # Calculate maximum drawdown duration
        max_duration_days = max(
            [period.duration_days for period in self.drawdown_periods] + [current_duration_days]
        )
        
        # Determine drawdown severity
        severity = self._determine_drawdown_severity(current_drawdown_pct)
        
        # Calculate underwater curve
        underwater_curve = self._calculate_underwater_curve()
        
        # Calculate consecutive losing days
        consecutive_losing_days = self._calculate_consecutive_losing_days()
        
        # Calculate recovery factor
        recovery_factor = self._calculate_recovery_factor()
        
        # Estimate time to recovery
        time_to_recovery = self._estimate_time_to_recovery(current_drawdown_pct)
        
        return DrawdownMetrics(
            current_drawdown_pct=current_drawdown_pct,
            max_drawdown_pct=max_drawdown_pct,
            current_drawdown_duration_days=current_duration_days,
            max_drawdown_duration_days=max_duration_days,
            drawdown_severity=severity,
            recovery_mode=self.recovery_mode,
            underwater_curve=underwater_curve,
            consecutive_losing_days=consecutive_losing_days,
            recovery_factor=recovery_factor,
            time_to_recovery_estimate=time_to_recovery
        )
    
    def _determine_drawdown_severity(self, drawdown_pct: Decimal) -> DrawdownSeverity:
        """Determine drawdown severity based on percentage."""
        if drawdown_pct == 0:
            return DrawdownSeverity.NONE
        elif drawdown_pct <= Decimal('0.02'):  # 2%
            return DrawdownSeverity.MINOR
        elif drawdown_pct <= Decimal('0.05'):  # 5%
            return DrawdownSeverity.MODERATE
        elif drawdown_pct <= Decimal('0.10'):  # 10%
            return DrawdownSeverity.SEVERE
        else:
            return DrawdownSeverity.EXTREME
    
    def _calculate_underwater_curve(self) -> List[Tuple[datetime, Decimal]]:
        """Calculate underwater curve (drawdown over time)."""
        if len(self.portfolio_values) < 2:
            return []
        
        underwater_curve = []
        running_peak = Money(Decimal('0'))
        
        for timestamp, value in self.portfolio_values:
            if value.amount > running_peak.amount:
                running_peak = value
            
            if running_peak.amount > 0:
                drawdown_pct = (running_peak.amount - value.amount) / running_peak.amount
                underwater_curve.append((timestamp, drawdown_pct))
        
        return underwater_curve[-100:]  # Return last 100 points
    
    def _calculate_consecutive_losing_days(self) -> int:
        """Calculate consecutive losing days."""
        if len(self.portfolio_values) < 2:
            return 0
        
        consecutive_days = 0
        values = list(self.portfolio_values)
        
        # Check from most recent backwards
        for i in range(len(values) - 1, 0, -1):
            current_value = values[i][1].amount
            previous_value = values[i-1][1].amount
            
            if current_value < previous_value:
                consecutive_days += 1
            else:
                break
        
        return consecutive_days
    
    def _calculate_recovery_factor(self) -> Decimal:
        """Calculate recovery factor (how quickly portfolio recovers from drawdowns)."""
        if not self.drawdown_periods:
            return Decimal('1.0')
        
        # Calculate average recovery time relative to drawdown magnitude
        recovery_factors = []
        
        for period in self.drawdown_periods[-10:]:  # Last 10 periods
            if period.recovered and period.duration_days > 0:
                # Recovery factor = drawdown_pct / recovery_time_days
                recovery_factor = period.max_drawdown_pct / Decimal(str(period.duration_days))
                recovery_factors.append(recovery_factor)
        
        if recovery_factors:
            return sum(recovery_factors) / len(recovery_factors)
        else:
            return Decimal('1.0')
    
    def _estimate_time_to_recovery(self, current_drawdown_pct: Decimal) -> Optional[int]:
        """Estimate time to recovery based on historical data."""
        if current_drawdown_pct == 0:
            return 0
        
        recovery_factor = self._calculate_recovery_factor()
        if recovery_factor > 0:
            estimated_days = int(current_drawdown_pct / recovery_factor)
            return min(estimated_days, 365)  # Cap at 1 year
        
        return None
    
    def _end_current_drawdown(self, timestamp: datetime, portfolio_value: Money) -> None:
        """End the current drawdown period."""
        if self.current_drawdown_start is None:
            return
        
        # Find the trough value during this drawdown
        trough_value = Money(float('inf'))
        max_drawdown_pct = Decimal('0')
        
        for ts, value in self.portfolio_values:
            if ts >= self.current_drawdown_start:
                if value.amount < trough_value.amount:
                    trough_value = value
                
                drawdown = (self.peak_value.amount - value.amount) / self.peak_value.amount
                if drawdown > max_drawdown_pct:
                    max_drawdown_pct = drawdown
        
        duration_days = (timestamp - self.current_drawdown_start).days
        
        drawdown_period = DrawdownPeriod(
            start_date=self.current_drawdown_start,
            end_date=timestamp,
            peak_value=self.peak_value,
            trough_value=trough_value,
            max_drawdown_pct=max_drawdown_pct,
            duration_days=duration_days,
            recovered=True,
            recovery_date=timestamp
        )
        
        self.drawdown_periods.append(drawdown_period)
        self.current_drawdown_start = None
        
        logger.info(
            f"Drawdown period ended: {max_drawdown_pct:.2%} max drawdown over {duration_days} days"
        )
    
    def _check_drawdown_limits(self, metrics: DrawdownMetrics) -> None:
        """Check drawdown limits and generate risk events."""
        # Check maximum drawdown limit
        if metrics.current_drawdown_pct > self.config.max_drawdown:
            event = RiskEventFactory.create_drawdown_limit_reached_event(
                symbol=Symbol('PORTFOLIO'),
                drawdown_pct=float(metrics.current_drawdown_pct),
                max_drawdown_pct=float(self.config.max_drawdown),
                action_taken="recovery_mode_activated"
            )
            publish_risk_event(event)
        
        # Check recovery mode trigger
        if (self.config.recovery_mode_enabled and 
            metrics.current_drawdown_pct > self.config.recovery_max_drawdown_trigger):
            
            if self.recovery_mode == RecoveryMode.NORMAL:
                logger.warning(
                    f"Recovery mode triggered: drawdown {metrics.current_drawdown_pct:.2%} "
                    f"exceeds trigger {self.config.recovery_max_drawdown_trigger:.2%}"
                )
        
        # Check consecutive losing days
        if metrics.consecutive_losing_days > self.config.max_losing_streak:
            event = RiskEvent(
                event_type=RiskEventType.DAILY_LOSS_EXCEEDED,
                timestamp=datetime.now(),
                symbol=Symbol('PORTFOLIO'),
                risk_level=RiskLevel.HIGH,
                message=f"Consecutive losing days {metrics.consecutive_losing_days} exceeds limit {self.config.max_losing_streak}",
                data={
                    'consecutive_losing_days': metrics.consecutive_losing_days,
                    'limit': self.config.max_losing_streak,
                    'current_drawdown_pct': float(metrics.current_drawdown_pct)
                }
            )
            publish_risk_event(event)
    
    def _update_recovery_mode(self, metrics: DrawdownMetrics) -> None:
        """Update recovery mode based on drawdown severity."""
        if not self.config.recovery_mode_enabled:
            return
        
        current_drawdown = metrics.current_drawdown_pct
        
        # Determine new recovery mode
        if current_drawdown == 0:
            new_mode = RecoveryMode.NORMAL
        elif current_drawdown <= Decimal('0.05'):  # 5%
            new_mode = RecoveryMode.NORMAL
        elif current_drawdown <= Decimal('0.10'):  # 10%
            new_mode = RecoveryMode.CONSERVATIVE
        elif current_drawdown <= Decimal('0.15'):  # 15%
            new_mode = RecoveryMode.DEFENSIVE
        else:
            new_mode = RecoveryMode.HALT_TRADING
        
        if new_mode != self.recovery_mode:
            old_mode = self.recovery_mode
            self.recovery_mode = new_mode
            
            logger.info(
                f"Recovery mode changed: {old_mode.value} -> {new_mode.value} "
                f"(drawdown: {current_drawdown:.2%})"
            )
    
    def get_position_size_multiplier(self) -> Decimal:
        """Get position size multiplier based on recovery mode."""
        multipliers = {
            RecoveryMode.NORMAL: Decimal('1.0'),
            RecoveryMode.CONSERVATIVE: self.config.recovery_position_size_multiplier,
            RecoveryMode.DEFENSIVE: self.config.recovery_position_size_multiplier * Decimal('0.5'),
            RecoveryMode.HALT_TRADING: Decimal('0.0')
        }
        
        return multipliers.get(self.recovery_mode, Decimal('1.0'))
    
    def should_halt_trading(self) -> bool:
        """Check if trading should be halted due to drawdown."""
        return self.recovery_mode == RecoveryMode.HALT_TRADING
    
    def get_drawdown_summary(self) -> Dict[str, any]:
        """Get drawdown summary for reporting."""
        if not self.portfolio_values:
            return {}
        
        latest_timestamp, latest_value = self.portfolio_values[-1]
        metrics = self._calculate_drawdown_metrics(latest_value, latest_timestamp)
        
        return {
            'current_drawdown_pct': float(metrics.current_drawdown_pct),
            'max_drawdown_pct': float(metrics.max_drawdown_pct),
            'current_duration_days': metrics.current_drawdown_duration_days,
            'max_duration_days': metrics.max_drawdown_duration_days,
            'drawdown_severity': metrics.drawdown_severity.value,
            'recovery_mode': metrics.recovery_mode.value,
            'consecutive_losing_days': metrics.consecutive_losing_days,
            'recovery_factor': float(metrics.recovery_factor),
            'time_to_recovery_estimate': metrics.time_to_recovery_estimate,
            'position_size_multiplier': float(self.get_position_size_multiplier()),
            'trading_halted': self.should_halt_trading(),
            'total_drawdown_periods': len(self.drawdown_periods),
            'peak_portfolio_value': float(self.peak_value.amount),
            'current_portfolio_value': float(latest_value.amount)
        }
    
    def get_historical_drawdowns(self) -> List[Dict[str, any]]:
        """Get historical drawdown periods."""
        return [
            {
                'start_date': period.start_date.isoformat(),
                'end_date': period.end_date.isoformat() if period.end_date else None,
                'max_drawdown_pct': float(period.max_drawdown_pct),
                'duration_days': period.duration_days,
                'peak_value': float(period.peak_value.amount),
                'trough_value': float(period.trough_value.amount),
                'recovered': period.recovered,
                'recovery_date': period.recovery_date.isoformat() if period.recovery_date else None
            }
            for period in self.drawdown_periods
        ]


class DrawdownAnalyzer:
    """Analyzer for drawdown patterns and statistics."""
    
    def __init__(self, drawdown_controller: DrawdownController):
        self.controller = drawdown_controller
    
    def analyze_drawdown_patterns(self) -> Dict[str, any]:
        """Analyze drawdown patterns."""
        periods = self.controller.drawdown_periods
        
        if not periods:
            return {'message': 'No drawdown periods to analyze'}
        
        # Basic statistics
        drawdowns = [p.max_drawdown_pct for p in periods]
        durations = [p.duration_days for p in periods]
        
        avg_drawdown = sum(drawdowns) / len(drawdowns)
        max_drawdown = max(drawdowns)
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        
        # Recovery statistics
        recovered_periods = [p for p in periods if p.recovered]
        recovery_rate = len(recovered_periods) / len(periods) if periods else 0
        
        if recovered_periods:
            avg_recovery_time = sum(p.duration_days for p in recovered_periods) / len(recovered_periods)
        else:
            avg_recovery_time = 0
        
        # Frequency analysis
        if len(periods) >= 2:
            time_between_drawdowns = []
            for i in range(1, len(periods)):
                if periods[i-1].end_date and periods[i].start_date:
                    days_between = (periods[i].start_date - periods[i-1].end_date).days
                    time_between_drawdowns.append(days_between)
            
            avg_time_between = sum(time_between_drawdowns) / len(time_between_drawdowns) if time_between_drawdowns else 0
        else:
            avg_time_between = 0
        
        return {
            'total_periods': len(periods),
            'average_drawdown_pct': float(avg_drawdown),
            'maximum_drawdown_pct': float(max_drawdown),
            'average_duration_days': avg_duration,
            'maximum_duration_days': max_duration,
            'recovery_rate': recovery_rate,
            'average_recovery_time_days': avg_recovery_time,
            'average_time_between_drawdowns_days': avg_time_between,
            'current_recovery_mode': self.controller.recovery_mode.value,
            'trading_halted': self.controller.should_halt_trading()
        }


# Global drawdown controller
_drawdown_controller = None


def get_drawdown_controller() -> DrawdownController:
    """Get global drawdown controller."""
    global _drawdown_controller
    if _drawdown_controller is None:
        _drawdown_controller = DrawdownController()
    return _drawdown_controller