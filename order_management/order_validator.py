"""
Order validation system with pluggable validation rules.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from decimal import Decimal
from datetime import datetime, time
from dataclasses import dataclass
from enum import Enum

from domain.value_objects import Symbol, Price, Quantity, Money
from .order_types import Order, OrderSide, OrderType, MarketOrder, LimitOrder
from risk_management import get_risk_config, get_portfolio_limiter
from utils.logging import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    rule_name: str
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of order validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int = 0
    errors_count: int = 0
    
    def __post_init__(self):
        self.warnings_count = len([i for i in self.issues if i.severity == ValidationSeverity.WARNING])
        self.errors_count = len([i for i in self.issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]])
        self.is_valid = self.errors_count == 0
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add validation issue."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.WARNING:
            self.warnings_count += 1
        elif issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.errors_count += 1
        self.is_valid = self.errors_count == 0
    
    def get_errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
    
    def get_warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]


class ValidationRule(ABC):
    """Abstract base class for validation rules."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
    
    @abstractmethod
    def validate(self, order: Order, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate order and return issues."""
        pass
    
    def is_applicable(self, order: Order) -> bool:
        """Check if rule applies to this order."""
        return True


class BasicOrderValidationRule(ValidationRule):
    """Basic order field validation."""
    
    def __init__(self):
        super().__init__("BasicOrderValidation")
    
    def validate(self, order: Order, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate basic order fields."""
        issues = []
        
        # Check symbol
        if not order.symbol or str(order.symbol).strip() == "":
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message="Symbol is required",
                field="symbol"
            ))
        
        # Check quantity
        if order.quantity.value <= 0:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message="Quantity must be positive",
                field="quantity",
                suggested_fix="Set quantity > 0"
            ))
        
        # Check limit order price
        if isinstance(order, LimitOrder):
            if order.price.value <= 0:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    message="Limit price must be positive",
                    field="price",
                    suggested_fix="Set price > 0"
                ))
        
        # Check for reasonable order sizes
        if order.quantity.value > Decimal('1000000'):  # 1M shares
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.WARNING,
                message="Very large order quantity detected",
                field="quantity",
                suggested_fix="Consider breaking into smaller orders"
            ))
        
        return issues


class RiskValidationRule(ValidationRule):
    """Risk management validation rule."""
    
    def __init__(self):
        super().__init__("RiskValidation")
        self.risk_config = get_risk_config()
        self.portfolio_limiter = get_portfolio_limiter()
    
    def validate(self, order: Order, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate order against risk limits."""
        issues = []
        
        portfolio_value = context.get('portfolio_value', Money(Decimal('100000')))
        current_positions = context.get('current_positions', [])
        
        # Estimate order value
        order_value = self._estimate_order_value(order, context)
        if not order_value:
            return issues
        
        # Check position size limit
        position_pct = order_value.amount / portfolio_value.amount
        if position_pct > self.risk_config.max_position_size:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"Order size {position_pct:.2%} exceeds position limit {self.risk_config.max_position_size:.2%}",
                field="quantity",
                suggested_fix=f"Reduce quantity to stay within {self.risk_config.max_position_size:.2%} limit"
            ))
        
        # Check portfolio exposure limits
        can_add, warnings = self.portfolio_limiter.can_add_position(
            order.symbol,
            order_value,
            current_positions,
            portfolio_value
        )
        
        if not can_add:
            for warning in warnings:
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.ERROR,
                    message=warning,
                    suggested_fix="Reduce position size or review portfolio allocation"
                ))
        
        # Check for concentration risk
        if position_pct > Decimal('0.05'):  # 5% warning threshold
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.WARNING,
                message=f"Large position size {position_pct:.2%} may increase concentration risk",
                suggested_fix="Consider portfolio diversification"
            ))
        
        return issues
    
    def _estimate_order_value(self, order: Order, context: Dict[str, Any]) -> Optional[Money]:
        """Estimate order value for validation."""
        if isinstance(order, LimitOrder):
            return Money(order.quantity.value * order.price.value)
        elif isinstance(order, MarketOrder):
            # Use current market price
            current_price = context.get('current_price', Decimal('100'))
            return Money(order.quantity.value * current_price)
        else:
            # For stop orders, use stop price as estimate
            if hasattr(order, 'stop_price'):
                return Money(order.quantity.value * order.stop_price.value)
        
        return None


class MarketHoursValidationRule(ValidationRule):
    """Market hours validation rule."""
    
    def __init__(self):
        super().__init__("MarketHoursValidation")
        self.market_open = time(9, 30)   # 9:30 AM
        self.market_close = time(16, 0)  # 4:00 PM
        self.allow_extended_hours = True
        self.extended_open = time(4, 0)  # 4:00 AM
        self.extended_close = time(20, 0)  # 8:00 PM
    
    def validate(self, order: Order, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate order against market hours."""
        issues = []
        
        current_time = datetime.now().time()
        is_weekend = datetime.now().weekday() >= 5  # Saturday = 5, Sunday = 6
        
        # Check if it's weekend
        if is_weekend:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.WARNING,
                message="Markets are closed on weekends",
                suggested_fix="Order will be queued for next trading day"
            ))
            return issues
        
        # Check market hours
        if self.allow_extended_hours:
            # Extended hours trading
            if not (self.extended_open <= current_time <= self.extended_close):
                issues.append(ValidationIssue(
                    rule_name=self.name,
                    severity=ValidationSeverity.WARNING,
                    message="Order placed outside extended trading hours",
                    suggested_fix="Order will be queued for next trading session"
                ))
        else:
            # Regular hours only
            if not (self.market_open <= current_time <= self.market_close):
                if isinstance(order, MarketOrder):
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        severity=ValidationSeverity.ERROR,
                        message="Market orders cannot be placed outside regular hours",
                        suggested_fix="Use limit order or wait for market open"
                    ))
                else:
                    issues.append(ValidationIssue(
                        rule_name=self.name,
                        severity=ValidationSeverity.WARNING,
                        message="Order placed outside regular trading hours",
                        suggested_fix="Order will be queued for market open"
                    ))
        
        return issues


class PositionSizeValidationRule(ValidationRule):
    """Position sizing validation rule."""
    
    def __init__(self):
        super().__init__("PositionSizeValidation")
        self.min_order_value = Money(Decimal('100'))    # $100 minimum
        self.max_order_value = Money(Decimal('1000000'))  # $1M maximum
    
    def validate(self, order: Order, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate position sizing."""
        issues = []
        
        # Estimate order value
        order_value = self._estimate_order_value(order, context)
        if not order_value:
            return issues
        
        # Check minimum order size
        if order_value.amount < self.min_order_value.amount:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.WARNING,
                message=f"Order value ${order_value.amount:.2f} below minimum ${self.min_order_value.amount:.2f}",
                suggested_fix="Increase order size or combine with other orders"
            ))
        
        # Check maximum order size
        if order_value.amount > self.max_order_value.amount:
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.ERROR,
                message=f"Order value ${order_value.amount:.2f} exceeds maximum ${self.max_order_value.amount:.2f}",
                field="quantity",
                suggested_fix="Break order into smaller chunks"
            ))
        
        # Check for odd lots (less than 100 shares for stocks)
        if order.quantity.value < Decimal('100'):
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.INFO,
                message="Odd lot order (< 100 shares) may have poor execution",
                suggested_fix="Consider rounding to 100 share increments"
            ))
        
        return issues
    
    def _estimate_order_value(self, order: Order, context: Dict[str, Any]) -> Optional[Money]:
        """Estimate order value."""
        if isinstance(order, LimitOrder):
            return Money(order.quantity.value * order.price.value)
        elif isinstance(order, MarketOrder):
            current_price = context.get('current_price', Decimal('100'))
            return Money(order.quantity.value * current_price)
        return None


class LiquidityValidationRule(ValidationRule):
    """Liquidity validation rule."""
    
    def __init__(self):
        super().__init__("LiquidityValidation")
    
    def validate(self, order: Order, context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate order against liquidity constraints."""
        issues = []
        
        daily_volume = context.get('daily_volume', Decimal('1000000'))
        avg_volume = context.get('avg_volume', daily_volume)
        current_price = context.get('current_price', Decimal('100'))
        
        # Calculate order as percentage of daily volume
        order_volume = order.quantity.value
        volume_ratio = order_volume / daily_volume if daily_volume > 0 else Decimal('1')
        
        # Check if order is large relative to daily volume
        if volume_ratio > Decimal('0.10'):  # 10% of daily volume
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.WARNING,
                message=f"Order size is {volume_ratio:.1%} of daily volume, may cause significant market impact",
                suggested_fix="Consider breaking into smaller orders or using TWAP/VWAP strategy"
            ))
        elif volume_ratio > Decimal('0.05'):  # 5% of daily volume
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.INFO,
                message=f"Order size is {volume_ratio:.1%} of daily volume",
                suggested_fix="Monitor for market impact"
            ))
        
        # Check for very low volume stocks
        if daily_volume < Decimal('10000'):  # Less than 10k shares daily
            issues.append(ValidationIssue(
                rule_name=self.name,
                severity=ValidationSeverity.WARNING,
                message="Low liquidity stock detected",
                suggested_fix="Expect wider spreads and potential execution delays"
            ))
        
        return issues


class OrderValidator:
    """Main order validator with pluggable rules."""
    
    def __init__(self):
        self.rules: List[ValidationRule] = []
        self.validation_stats = {
            'total_validations': 0,
            'valid_orders': 0,
            'invalid_orders': 0,
            'warnings_generated': 0,
            'errors_generated': 0
        }
        
        # Add default rules
        self._add_default_rules()
    
    def _add_default_rules(self) -> None:
        """Add default validation rules."""
        self.rules = [
            BasicOrderValidationRule(),
            RiskValidationRule(),
            MarketHoursValidationRule(),
            PositionSizeValidationRule(),
            LiquidityValidationRule()
        ]
    
    def validate_order(self, order: Order, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate order using all applicable rules."""
        context = context or {}
        result = ValidationResult(is_valid=True, issues=[])
        
        logger.debug(f"Validating order {order.order_id} ({order.order_type.value})")
        
        # Run all applicable rules
        for rule in self.rules:
            if rule.enabled and rule.is_applicable(order):
                try:
                    issues = rule.validate(order, context)
                    for issue in issues:
                        result.add_issue(issue)
                except Exception as e:
                    logger.error(f"Validation rule {rule.name} failed: {e}")
                    result.add_issue(ValidationIssue(
                        rule_name=rule.name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Validation rule failed: {e}"
                    ))
        
        # Update statistics
        self._update_validation_stats(result)
        
        # Log results
        if not result.is_valid:
            logger.warning(
                f"Order {order.order_id} validation failed: {result.errors_count} errors, "
                f"{result.warnings_count} warnings"
            )
            for error in result.get_errors():
                logger.warning(f"  ERROR: {error.message}")
        elif result.warnings_count > 0:
            logger.info(f"Order {order.order_id} has {result.warnings_count} warnings")
        else:
            logger.debug(f"Order {order.order_id} validation passed")
        
        return result
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add custom validation rule."""
        self.rules.append(rule)
        logger.info(f"Added validation rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove validation rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                logger.info(f"Removed validation rule: {rule_name}")
                return True
        return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable validation rule."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True
                logger.info(f"Enabled validation rule: {rule_name}")
                return True
        return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable validation rule."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.info(f"Disabled validation rule: {rule_name}")
                return True
        return False
    
    def get_enabled_rules(self) -> List[str]:
        """Get list of enabled rule names."""
        return [rule.name for rule in self.rules if rule.enabled]
    
    def _update_validation_stats(self, result: ValidationResult) -> None:
        """Update validation statistics."""
        self.validation_stats['total_validations'] += 1
        
        if result.is_valid:
            self.validation_stats['valid_orders'] += 1
        else:
            self.validation_stats['invalid_orders'] += 1
        
        self.validation_stats['warnings_generated'] += result.warnings_count
        self.validation_stats['errors_generated'] += result.errors_count
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self.validation_stats.copy()
        
        if stats['total_validations'] > 0:
            stats['validation_success_rate'] = stats['valid_orders'] / stats['total_validations']
            stats['average_warnings_per_order'] = stats['warnings_generated'] / stats['total_validations']
            stats['average_errors_per_order'] = stats['errors_generated'] / stats['total_validations']
        else:
            stats['validation_success_rate'] = 0
            stats['average_warnings_per_order'] = 0
            stats['average_errors_per_order'] = 0
        
        stats['enabled_rules'] = self.get_enabled_rules()
        stats['total_rules'] = len(self.rules)
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'valid_orders': 0,
            'invalid_orders': 0,
            'warnings_generated': 0,
            'errors_generated': 0
        }
        logger.info("Validation statistics reset")


# Global validator instance
_order_validator = None


def get_order_validator() -> OrderValidator:
    """Get global order validator."""
    global _order_validator
    if _order_validator is None:
        _order_validator = OrderValidator()
    return _order_validator