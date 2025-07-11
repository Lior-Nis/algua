"""
Risk management configuration system.
"""

from decimal import Decimal
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from .interfaces import RiskConfigurationProtocol


@dataclass
class RiskConfiguration:
    """Risk management configuration."""
    
    # Position sizing limits
    max_position_size: Decimal = Decimal('0.10')  # 10% of portfolio
    min_position_size: Decimal = Decimal('0.01')  # 1% of portfolio
    
    # Risk limits
    max_daily_loss: Decimal = Decimal('0.02')  # 2% of portfolio
    max_weekly_loss: Decimal = Decimal('0.05')  # 5% of portfolio
    max_monthly_loss: Decimal = Decimal('0.10')  # 10% of portfolio
    max_drawdown: Decimal = Decimal('0.15')  # 15% of portfolio
    
    # Portfolio limits
    max_portfolio_exposure: Decimal = Decimal('0.95')  # 95% of portfolio
    max_sector_exposure: Decimal = Decimal('0.30')  # 30% per sector
    max_single_stock_exposure: Decimal = Decimal('0.15')  # 15% per stock
    
    # Correlation limits
    max_correlation_exposure: Decimal = Decimal('0.40')  # 40% in correlated positions
    correlation_threshold: Decimal = Decimal('0.70')  # 70% correlation threshold
    
    # Stop loss settings
    default_stop_loss_pct: Decimal = Decimal('0.02')  # 2% stop loss
    max_stop_loss_pct: Decimal = Decimal('0.05')  # 5% max stop loss
    trailing_stop_enabled: bool = True
    trailing_stop_pct: Decimal = Decimal('0.01')  # 1% trailing stop
    
    # Time-based limits
    max_position_hold_days: int = 30  # Maximum days to hold position
    max_losing_streak: int = 5  # Maximum consecutive losing trades
    
    # Volatility settings
    volatility_lookback_days: int = 20  # Days for volatility calculation
    volatility_multiplier: Decimal = Decimal('2.0')  # ATR multiplier for stops
    
    # Risk metrics
    risk_free_rate: Decimal = Decimal('0.02')  # 2% annual risk-free rate
    var_confidence_level: Decimal = Decimal('0.95')  # 95% VaR confidence
    var_lookback_days: int = 250  # Days for VaR calculation
    
    # Monitoring settings
    risk_check_interval_minutes: int = 5  # Risk check frequency
    alert_threshold_breach_count: int = 3  # Alerts before action
    
    # Recovery settings
    recovery_mode_enabled: bool = True
    recovery_position_size_multiplier: Decimal = Decimal('0.5')  # 50% position size in recovery
    recovery_max_drawdown_trigger: Decimal = Decimal('0.10')  # 10% drawdown triggers recovery
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'max_position_size': float(self.max_position_size),
            'min_position_size': float(self.min_position_size),
            'max_daily_loss': float(self.max_daily_loss),
            'max_weekly_loss': float(self.max_weekly_loss),
            'max_monthly_loss': float(self.max_monthly_loss),
            'max_drawdown': float(self.max_drawdown),
            'max_portfolio_exposure': float(self.max_portfolio_exposure),
            'max_sector_exposure': float(self.max_sector_exposure),
            'max_single_stock_exposure': float(self.max_single_stock_exposure),
            'max_correlation_exposure': float(self.max_correlation_exposure),
            'correlation_threshold': float(self.correlation_threshold),
            'default_stop_loss_pct': float(self.default_stop_loss_pct),
            'max_stop_loss_pct': float(self.max_stop_loss_pct),
            'trailing_stop_enabled': self.trailing_stop_enabled,
            'trailing_stop_pct': float(self.trailing_stop_pct),
            'max_position_hold_days': self.max_position_hold_days,
            'max_losing_streak': self.max_losing_streak,
            'volatility_lookback_days': self.volatility_lookback_days,
            'volatility_multiplier': float(self.volatility_multiplier),
            'risk_free_rate': float(self.risk_free_rate),
            'var_confidence_level': float(self.var_confidence_level),
            'var_lookback_days': self.var_lookback_days,
            'risk_check_interval_minutes': self.risk_check_interval_minutes,
            'alert_threshold_breach_count': self.alert_threshold_breach_count,
            'recovery_mode_enabled': self.recovery_mode_enabled,
            'recovery_position_size_multiplier': float(self.recovery_position_size_multiplier),
            'recovery_max_drawdown_trigger': float(self.recovery_max_drawdown_trigger)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskConfiguration':
        """Create configuration from dictionary."""
        return cls(
            max_position_size=Decimal(str(data.get('max_position_size', '0.10'))),
            min_position_size=Decimal(str(data.get('min_position_size', '0.01'))),
            max_daily_loss=Decimal(str(data.get('max_daily_loss', '0.02'))),
            max_weekly_loss=Decimal(str(data.get('max_weekly_loss', '0.05'))),
            max_monthly_loss=Decimal(str(data.get('max_monthly_loss', '0.10'))),
            max_drawdown=Decimal(str(data.get('max_drawdown', '0.15'))),
            max_portfolio_exposure=Decimal(str(data.get('max_portfolio_exposure', '0.95'))),
            max_sector_exposure=Decimal(str(data.get('max_sector_exposure', '0.30'))),
            max_single_stock_exposure=Decimal(str(data.get('max_single_stock_exposure', '0.15'))),
            max_correlation_exposure=Decimal(str(data.get('max_correlation_exposure', '0.40'))),
            correlation_threshold=Decimal(str(data.get('correlation_threshold', '0.70'))),
            default_stop_loss_pct=Decimal(str(data.get('default_stop_loss_pct', '0.02'))),
            max_stop_loss_pct=Decimal(str(data.get('max_stop_loss_pct', '0.05'))),
            trailing_stop_enabled=data.get('trailing_stop_enabled', True),
            trailing_stop_pct=Decimal(str(data.get('trailing_stop_pct', '0.01'))),
            max_position_hold_days=data.get('max_position_hold_days', 30),
            max_losing_streak=data.get('max_losing_streak', 5),
            volatility_lookback_days=data.get('volatility_lookback_days', 20),
            volatility_multiplier=Decimal(str(data.get('volatility_multiplier', '2.0'))),
            risk_free_rate=Decimal(str(data.get('risk_free_rate', '0.02'))),
            var_confidence_level=Decimal(str(data.get('var_confidence_level', '0.95'))),
            var_lookback_days=data.get('var_lookback_days', 250),
            risk_check_interval_minutes=data.get('risk_check_interval_minutes', 5),
            alert_threshold_breach_count=data.get('alert_threshold_breach_count', 3),
            recovery_mode_enabled=data.get('recovery_mode_enabled', True),
            recovery_position_size_multiplier=Decimal(str(data.get('recovery_position_size_multiplier', '0.5'))),
            recovery_max_drawdown_trigger=Decimal(str(data.get('recovery_max_drawdown_trigger', '0.10')))
        )
    
    def save_to_file(self, file_path: Path) -> None:
        """Save configuration to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'RiskConfiguration':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class RiskConfigurationManager:
    """Manager for risk configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or Path("configs/risk_config.json")
        self._config = self._load_config()
    
    def _load_config(self) -> RiskConfiguration:
        """Load configuration from file or create default."""
        if self.config_path.exists():
            return RiskConfiguration.load_from_file(self.config_path)
        else:
            # Create default configuration
            config = RiskConfiguration()
            self._save_config(config)
            return config
    
    def _save_config(self, config: RiskConfiguration) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        config.save_to_file(self.config_path)
    
    def get_config(self) -> RiskConfiguration:
        """Get current configuration."""
        return self._config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        config_dict = self._config.to_dict()
        config_dict.update(kwargs)
        self._config = RiskConfiguration.from_dict(config_dict)
        self._save_config(self._config)
    
    def reset_to_default(self) -> None:
        """Reset configuration to default values."""
        self._config = RiskConfiguration()
        self._save_config(self._config)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        # Check logical consistency
        if self._config.max_position_size < self._config.min_position_size:
            issues.append("max_position_size must be >= min_position_size")
        
        if self._config.max_daily_loss >= self._config.max_weekly_loss:
            issues.append("max_weekly_loss must be > max_daily_loss")
        
        if self._config.max_weekly_loss >= self._config.max_monthly_loss:
            issues.append("max_monthly_loss must be > max_weekly_loss")
        
        if self._config.max_stop_loss_pct < self._config.default_stop_loss_pct:
            issues.append("max_stop_loss_pct must be >= default_stop_loss_pct")
        
        if self._config.correlation_threshold <= 0 or self._config.correlation_threshold > 1:
            issues.append("correlation_threshold must be between 0 and 1")
        
        if self._config.var_confidence_level <= 0 or self._config.var_confidence_level >= 1:
            issues.append("var_confidence_level must be between 0 and 1")
        
        # Check reasonable ranges
        if self._config.max_position_size > Decimal('0.5'):
            issues.append("max_position_size > 50% may be too risky")
        
        if self._config.max_daily_loss > Decimal('0.05'):
            issues.append("max_daily_loss > 5% may be too risky")
        
        return issues


# Global configuration instance
_config_manager = None


def get_risk_config() -> RiskConfiguration:
    """Get global risk configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = RiskConfigurationManager()
    return _config_manager.get_config()


def update_risk_config(**kwargs) -> None:
    """Update global risk configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = RiskConfigurationManager()
    _config_manager.update_config(**kwargs)