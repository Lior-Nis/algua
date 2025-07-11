"""
Real-time alerting and notification system.
"""

from typing import Dict, List, Optional, Any, Callable, Union
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from abc import ABC, abstractmethod
from collections import defaultdict

from .error_manager import ErrorContext, ErrorSeverity, ErrorCategory, ErrorType
from .health_monitor import HealthStatus, ComponentHealth
from utils.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(Enum):
    """Types of alerts."""
    ERROR = "error"
    PERFORMANCE = "performance"
    HEALTH = "health"
    SECURITY = "security"
    BUSINESS = "business"
    SYSTEM = "system"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"


class AlertChannel(Enum):
    """Alert notification channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    LOG = "log"
    PUSH = "push"


@dataclass
class Alert:
    """Alert definition."""
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    alert_type: AlertType
    status: AlertStatus = AlertStatus.ACTIVE
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Context
    source: Optional[str] = None
    component_id: Optional[str] = None
    error_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Escalation
    escalation_level: int = 0
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Actions
    actions_taken: List[str] = field(default_factory=list)
    recovery_attempted: bool = False
    
    def acknowledge(self, user_id: Optional[str] = None) -> None:
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now()
        self.updated_at = datetime.now()
        
        if user_id:
            self.metadata['acknowledged_by'] = user_id
    
    def resolve(self, resolution_message: Optional[str] = None) -> None:
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now()
        self.updated_at = datetime.now()
        
        if resolution_message:
            self.metadata['resolution_message'] = resolution_message
    
    def suppress(self, duration: Optional[timedelta] = None) -> None:
        """Suppress the alert."""
        self.status = AlertStatus.SUPPRESSED
        self.updated_at = datetime.now()
        
        if duration:
            self.expires_at = datetime.now() + duration
    
    def escalate(self, reason: str) -> None:
        """Escalate the alert."""
        self.escalation_level += 1
        self.escalation_history.append({
            'level': self.escalation_level,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        self.updated_at = datetime.now()
    
    def is_active(self) -> bool:
        """Check if alert is active."""
        if self.status != AlertStatus.ACTIVE:
            return False
        
        if self.expires_at and datetime.now() > self.expires_at:
            self.status = AlertStatus.EXPIRED
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'title': self.title,
            'message': self.message,
            'severity': self.severity.value,
            'alert_type': self.alert_type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'source': self.source,
            'component_id': self.component_id,
            'error_id': self.error_id,
            'tags': self.tags,
            'metadata': self.metadata,
            'escalation_level': self.escalation_level,
            'escalation_history': self.escalation_history,
            'actions_taken': self.actions_taken,
            'recovery_attempted': self.recovery_attempted
        }


@dataclass
class AlertRule:
    """Alert rule definition."""
    rule_id: str
    name: str
    description: str
    
    # Conditions
    severity_threshold: AlertSeverity = AlertSeverity.MEDIUM
    error_types: List[ErrorType] = field(default_factory=list)
    error_categories: List[ErrorCategory] = field(default_factory=list)
    component_types: List[str] = field(default_factory=list)
    
    # Frequency rules
    frequency_threshold: int = 1
    time_window: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    
    # Notification settings
    channels: List[AlertChannel] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)
    
    # Escalation
    escalation_enabled: bool = False
    escalation_delay: timedelta = field(default_factory=lambda: timedelta(minutes=15))
    escalation_channels: List[AlertChannel] = field(default_factory=list)
    escalation_recipients: List[str] = field(default_factory=list)
    
    # Suppression
    suppression_enabled: bool = False
    suppression_duration: timedelta = field(default_factory=lambda: timedelta(hours=1))
    
    # State
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    def matches(self, error_context: ErrorContext) -> bool:
        """Check if rule matches the error context."""
        if not self.enabled:
            return False
        
        # Check severity
        if error_context.severity.value < self.severity_threshold.value:
            return False
        
        # Check error types
        if self.error_types and error_context.error_type not in self.error_types:
            return False
        
        # Check error categories
        if self.error_categories and error_context.category not in self.error_categories:
            return False
        
        # Check component types (if specified)
        if self.component_types and error_context.module:
            if not any(comp_type in error_context.module for comp_type in self.component_types):
                return False
        
        return True
    
    def should_trigger(self, error_context: ErrorContext) -> bool:
        """Check if rule should trigger based on frequency rules."""
        if not self.matches(error_context):
            return False
        
        # Simple frequency check (can be enhanced with sliding window)
        if self.last_triggered:
            time_since_last = datetime.now() - self.last_triggered
            if time_since_last < self.time_window:
                return False
        
        return True


class AlertNotifier(ABC):
    """Base class for alert notifiers."""
    
    @abstractmethod
    def send_alert(self, alert: Alert, recipients: List[str]) -> bool:
        """Send alert notification."""
        pass
    
    @abstractmethod
    def get_channel_type(self) -> AlertChannel:
        """Get the channel type this notifier handles."""
        pass


class EmailNotifier(AlertNotifier):
    """Email alert notifier."""
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
    
    def send_alert(self, alert: Alert, recipients: List[str]) -> bool:
        """Send email alert."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username or 'algua@trading.com'
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create email body
            body = f"""
Alert Details:
- Alert ID: {alert.alert_id}
- Severity: {alert.severity.value.upper()}
- Type: {alert.alert_type.value}
- Created: {alert.created_at}
- Source: {alert.source or 'Unknown'}
- Component: {alert.component_id or 'Unknown'}

Message:
{alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}

Actions Required:
- Acknowledge this alert to prevent escalation
- Investigate the root cause
- Take appropriate remediation actions

Alert Management URL: [Configure your alert management URL here]
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                
                if self.username and self.password:
                    server.login(self.username, self.password)
                
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def get_channel_type(self) -> AlertChannel:
        return AlertChannel.EMAIL


class ConsoleNotifier(AlertNotifier):
    """Console alert notifier."""
    
    def send_alert(self, alert: Alert, recipients: List[str]) -> bool:
        """Send console alert."""
        try:
            severity_colors = {
                AlertSeverity.CRITICAL: '\033[91m',  # Red
                AlertSeverity.HIGH: '\033[93m',      # Yellow
                AlertSeverity.MEDIUM: '\033[94m',    # Blue
                AlertSeverity.LOW: '\033[92m',       # Green
                AlertSeverity.INFO: '\033[96m'       # Cyan
            }
            
            color = severity_colors.get(alert.severity, '\033[0m')
            reset = '\033[0m'
            
            print(f"\n{color}{'='*60}{reset}")
            print(f"{color}ALERT: {alert.title}{reset}")
            print(f"{color}{'='*60}{reset}")
            print(f"ID: {alert.alert_id}")
            print(f"Severity: {color}{alert.severity.value.upper()}{reset}")
            print(f"Type: {alert.alert_type.value}")
            print(f"Time: {alert.created_at}")
            print(f"Source: {alert.source or 'Unknown'}")
            print(f"Component: {alert.component_id or 'Unknown'}")
            print(f"\nMessage:\n{alert.message}")
            
            if alert.metadata:
                print(f"\nMetadata:")
                for key, value in alert.metadata.items():
                    print(f"  {key}: {value}")
            
            print(f"{color}{'='*60}{reset}\n")
            
            logger.info(f"Console alert displayed for {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to display console alert: {e}")
            return False
    
    def get_channel_type(self) -> AlertChannel:
        return AlertChannel.CONSOLE


class WebhookNotifier(AlertNotifier):
    """Webhook alert notifier."""
    
    def __init__(self, webhook_url: str, timeout: int = 30):
        self.webhook_url = webhook_url
        self.timeout = timeout
    
    def send_alert(self, alert: Alert, recipients: List[str]) -> bool:
        """Send webhook alert."""
        try:
            import requests
            
            payload = {
                'alert': alert.to_dict(),
                'recipients': recipients,
                'timestamp': datetime.now().isoformat()
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook alert sent for {alert.alert_id}")
                return True
            else:
                logger.error(f"Webhook alert failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def get_channel_type(self) -> AlertChannel:
        return AlertChannel.WEBHOOK


class AlertManager:
    """Central alert management system."""
    
    def __init__(self):
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notifiers: Dict[AlertChannel, AlertNotifier] = {}
        self.alert_history: List[Alert] = []
        
        # Configuration
        self.max_active_alerts = 1000
        self.max_history_size = 10000
        self.auto_expire_duration = timedelta(hours=24)
        self.escalation_enabled = True
        self.deduplication_enabled = True
        self.deduplication_window = timedelta(minutes=5)
        
        # Statistics
        self.alert_stats = {
            'total_alerts': 0,
            'active_alerts': 0,
            'resolved_alerts': 0,
            'escalated_alerts': 0,
            'suppressed_alerts': 0,
            'alerts_by_severity': {severity.value: 0 for severity in AlertSeverity},
            'alerts_by_type': {alert_type.value: 0 for alert_type in AlertType},
            'notification_success_rate': 0.0
        }
        
        # State
        self.alert_counts: Dict[str, int] = defaultdict(int)
        self.recent_alerts: Dict[str, datetime] = {}
        
        # Threading
        self._lock = threading.Lock()
        self._escalation_thread = None
        self._cleanup_thread = None
        self._stop_event = threading.Event()
        
        # Initialize default notifiers
        self._initialize_default_notifiers()
        
        # Background tasks will be started when needed
        self._background_tasks_started = False
        
        logger.info("Alert manager initialized")
    
    def create_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        alert_type: AlertType,
        source: Optional[str] = None,
        component_id: Optional[str] = None,
        error_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_notify: bool = True
    ) -> Alert:
        """Create a new alert."""
        with self._lock:
            # Start background tasks if not already started
            if not self._background_tasks_started:
                self._start_background_tasks()
                self._background_tasks_started = True
            # Check for deduplication
            if self.deduplication_enabled:
                duplicate_alert = self._find_duplicate_alert(title, message, source, component_id)
                if duplicate_alert:
                    logger.debug(f"Duplicate alert suppressed: {title}")
                    return duplicate_alert
            
            # Create alert
            alert = Alert(
                alert_id=self._generate_alert_id(),
                title=title,
                message=message,
                severity=severity,
                alert_type=alert_type,
                source=source,
                component_id=component_id,
                error_id=error_id,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Set auto-expire
            if self.auto_expire_duration:
                alert.expires_at = datetime.now() + self.auto_expire_duration
            
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Update statistics
            self._update_alert_statistics(alert)
            
            # Send notifications
            if auto_notify:
                self._send_notifications(alert)
            
            logger.info(f"Alert created: {alert.alert_id} - {severity.value} - {title}")
            
            return alert
    
    def create_alert_from_error(
        self,
        error_context: ErrorContext,
        auto_notify: bool = True
    ) -> Optional[Alert]:
        """Create alert from error context."""
        # Find matching rules
        matching_rules = self._find_matching_rules(error_context)
        
        if not matching_rules:
            return None
        
        # Use the first matching rule
        rule = matching_rules[0]
        
        # Map error severity to alert severity
        severity_map = {
            ErrorSeverity.CRITICAL: AlertSeverity.CRITICAL,
            ErrorSeverity.HIGH: AlertSeverity.HIGH,
            ErrorSeverity.MEDIUM: AlertSeverity.MEDIUM,
            ErrorSeverity.LOW: AlertSeverity.LOW,
            ErrorSeverity.INFO: AlertSeverity.INFO
        }
        
        alert_severity = severity_map.get(error_context.severity, AlertSeverity.MEDIUM)
        
        # Create alert
        alert = self.create_alert(
            title=f"Error in {error_context.module or 'System'}: {error_context.error_type.value}",
            message=error_context.message,
            severity=alert_severity,
            alert_type=AlertType.ERROR,
            source=error_context.module,
            component_id=error_context.module,
            error_id=error_context.error_id,
            tags=[error_context.category.value, error_context.error_type.value],
            metadata={
                'error_context': error_context.to_dict(),
                'rule_id': rule.rule_id
            },
            auto_notify=auto_notify
        )
        
        return alert
    
    def acknowledge_alert(self, alert_id: str, user_id: Optional[str] = None) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.acknowledge(user_id)
            
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
    
    def resolve_alert(self, alert_id: str, resolution_message: Optional[str] = None) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.resolve(resolution_message)
            
            # Move to history
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert_id}")
            return True
    
    def suppress_alert(self, alert_id: str, duration: Optional[timedelta] = None) -> bool:
        """Suppress an alert."""
        with self._lock:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.suppress(duration)
            
            logger.info(f"Alert suppressed: {alert_id}")
            return True
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        with self._lock:
            self.alert_rules[rule.rule_id] = rule
            logger.info(f"Alert rule added: {rule.rule_id}")
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        with self._lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                logger.info(f"Alert rule removed: {rule_id}")
                return True
            return False
    
    def add_notifier(self, notifier: AlertNotifier) -> None:
        """Add a notification channel."""
        channel = notifier.get_channel_type()
        self.notifiers[channel] = notifier
        logger.info(f"Notifier added: {channel.value}")
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[AlertType] = None,
        component_id: Optional[str] = None
    ) -> List[Alert]:
        """Get active alerts with optional filtering."""
        with self._lock:
            alerts = list(self.active_alerts.values())
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            if alert_type:
                alerts = [a for a in alerts if a.alert_type == alert_type]
            
            if component_id:
                alerts = [a for a in alerts if a.component_id == component_id]
            
            return sorted(alerts, key=lambda x: x.created_at, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            stats = self.alert_stats.copy()
            stats['active_alerts'] = len(self.active_alerts)
            stats['total_rules'] = len(self.alert_rules)
            stats['available_channels'] = [ch.value for ch in self.notifiers.keys()]
            return stats
    
    def cleanup_expired_alerts(self) -> int:
        """Clean up expired alerts."""
        with self._lock:
            expired_count = 0
            expired_alerts = []
            
            for alert_id, alert in self.active_alerts.items():
                if not alert.is_active():
                    expired_alerts.append(alert_id)
                    expired_count += 1
            
            for alert_id in expired_alerts:
                del self.active_alerts[alert_id]
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired alerts")
            
            return expired_count
    
    def stop(self) -> None:
        """Stop the alert manager."""
        self._stop_event.set()
        
        if self._escalation_thread and self._escalation_thread.is_alive():
            self._escalation_thread.join(timeout=5)
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        logger.info("Alert manager stopped")
    
    # Private methods
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _find_duplicate_alert(
        self,
        title: str,
        message: str,
        source: Optional[str],
        component_id: Optional[str]
    ) -> Optional[Alert]:
        """Find duplicate alert within deduplication window."""
        cutoff_time = datetime.now() - self.deduplication_window
        
        for alert in self.active_alerts.values():
            if (alert.title == title and 
                alert.message == message and 
                alert.source == source and 
                alert.component_id == component_id and
                alert.created_at >= cutoff_time):
                return alert
        
        return None
    
    def _find_matching_rules(self, error_context: ErrorContext) -> List[AlertRule]:
        """Find alert rules that match the error context."""
        matching_rules = []
        
        for rule in self.alert_rules.values():
            if rule.should_trigger(error_context):
                matching_rules.append(rule)
        
        return matching_rules
    
    def _send_notifications(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        # Find rules that match this alert
        matching_rules = []
        for rule in self.alert_rules.values():
            if self._alert_matches_rule(alert, rule):
                matching_rules.append(rule)
        
        # Send notifications based on matching rules
        for rule in matching_rules:
            for channel in rule.channels:
                if channel in self.notifiers:
                    try:
                        notifier = self.notifiers[channel]
                        success = notifier.send_alert(alert, rule.recipients)
                        
                        if success:
                            alert.actions_taken.append(f"Notified via {channel.value}")
                        else:
                            alert.actions_taken.append(f"Failed to notify via {channel.value}")
                    
                    except Exception as e:
                        logger.error(f"Notification failed for {channel.value}: {e}")
    
    def _alert_matches_rule(self, alert: Alert, rule: AlertRule) -> bool:
        """Check if alert matches a rule."""
        # Simple matching logic - can be enhanced
        if alert.severity.value < rule.severity_threshold.value:
            return False
        
        return True
    
    def _update_alert_statistics(self, alert: Alert) -> None:
        """Update alert statistics."""
        self.alert_stats['total_alerts'] += 1
        self.alert_stats['alerts_by_severity'][alert.severity.value] += 1
        self.alert_stats['alerts_by_type'][alert.alert_type.value] += 1
    
    def _initialize_default_notifiers(self) -> None:
        """Initialize default notifiers."""
        # Console notifier is always available
        self.notifiers[AlertChannel.CONSOLE] = ConsoleNotifier()
        
        # Add other notifiers based on configuration
        # Email notifier would be added with proper SMTP configuration
        # Webhook notifier would be added with webhook URL
        
        logger.info("Default notifiers initialized")
    
    def _start_background_tasks(self) -> None:
        """Start background tasks."""
        self._escalation_thread = threading.Thread(
            target=self._escalation_loop,
            name="AlertEscalation",
            daemon=True
        )
        self._escalation_thread.start()
        
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="AlertCleanup",
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _escalation_loop(self) -> None:
        """Background escalation loop."""
        while not self._stop_event.is_set():
            try:
                if self.escalation_enabled:
                    self._check_escalations()
                
                # Check every minute
                if self._stop_event.wait(60):
                    break
                    
            except Exception as e:
                logger.error(f"Error in escalation loop: {e}")
                time.sleep(60)
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._stop_event.is_set():
            try:
                self.cleanup_expired_alerts()
                
                # Cleanup every 5 minutes
                if self._stop_event.wait(300):
                    break
                    
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(300)
    
    def _check_escalations(self) -> None:
        """Check for alerts that need escalation."""
        with self._lock:
            now = datetime.now()
            
            for alert in self.active_alerts.values():
                if (alert.status == AlertStatus.ACTIVE and 
                    not alert.escalation_level and  # Not yet escalated
                    alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]):
                    
                    # Check if escalation time has passed
                    escalation_time = alert.created_at + timedelta(minutes=15)  # Default escalation delay
                    
                    if now >= escalation_time:
                        alert.escalate("Automatic escalation due to unacknowledged critical alert")
                        
                        # Send escalation notifications
                        self._send_escalation_notifications(alert)
                        
                        logger.warning(f"Alert escalated: {alert.alert_id}")
    
    def _send_escalation_notifications(self, alert: Alert) -> None:
        """Send escalation notifications."""
        # Find escalation rules and send notifications
        for rule in self.alert_rules.values():
            if rule.escalation_enabled and self._alert_matches_rule(alert, rule):
                for channel in rule.escalation_channels:
                    if channel in self.notifiers:
                        try:
                            notifier = self.notifiers[channel]
                            escalation_alert = Alert(
                                alert_id=f"{alert.alert_id}-escalation",
                                title=f"ESCALATED: {alert.title}",
                                message=f"Alert has been escalated due to no acknowledgment.\n\nOriginal Alert:\n{alert.message}",
                                severity=AlertSeverity.CRITICAL,
                                alert_type=alert.alert_type,
                                source=alert.source,
                                component_id=alert.component_id,
                                metadata=alert.metadata
                            )
                            
                            notifier.send_alert(escalation_alert, rule.escalation_recipients)
                            
                        except Exception as e:
                            logger.error(f"Escalation notification failed: {e}")


# Global alert manager instance
_alert_manager = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager