"""
Portfolio risk limits and exposure management.
"""

from typing import Dict, List, Optional, Set
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from domain.value_objects import Symbol, Money
from .interfaces import PositionRisk, RiskEvent, RiskLevel, RiskEventType
from .configuration import get_risk_config
from .event_system import publish_risk_event, RiskEventFactory
from utils.logging import get_logger

logger = get_logger(__name__)


class ExposureType(Enum):
    """Types of portfolio exposure."""
    TOTAL_PORTFOLIO = "total_portfolio"
    SINGLE_POSITION = "single_position"
    SECTOR = "sector"
    ASSET_CLASS = "asset_class"
    CORRELATION_GROUP = "correlation_group"
    GEOGRAPHIC_REGION = "geographic_region"


@dataclass
class ExposureLimit:
    """Exposure limit configuration."""
    exposure_type: ExposureType
    identifier: str  # sector name, asset class, etc.
    max_exposure_pct: Decimal
    current_exposure_pct: Decimal = Decimal('0')
    positions_count: int = 0
    warning_threshold_pct: Optional[Decimal] = None


@dataclass
class PortfolioExposure:
    """Portfolio exposure summary."""
    total_value: Money
    total_exposure_pct: Decimal
    cash_pct: Decimal
    position_count: int
    exposures_by_type: Dict[ExposureType, List[ExposureLimit]]
    concentration_risk_score: Decimal


class PortfolioRiskLimiter:
    """Portfolio risk limits enforcement."""
    
    def __init__(self, config=None):
        self.config = config or get_risk_config()
        self.exposure_limits = self._initialize_exposure_limits()
        self.sector_mappings = {}  # symbol -> sector
        self.asset_class_mappings = {}  # symbol -> asset_class
        self.correlation_groups = {}  # symbol -> correlation_group
    
    def _initialize_exposure_limits(self) -> Dict[str, ExposureLimit]:
        """Initialize exposure limits from configuration."""
        limits = {}
        
        # Portfolio-wide limits
        limits['total_exposure'] = ExposureLimit(
            exposure_type=ExposureType.TOTAL_PORTFOLIO,
            identifier='total',
            max_exposure_pct=self.config.max_portfolio_exposure,
            warning_threshold_pct=self.config.max_portfolio_exposure * Decimal('0.9')
        )
        
        # Single position limits
        limits['single_position'] = ExposureLimit(
            exposure_type=ExposureType.SINGLE_POSITION,
            identifier='any',
            max_exposure_pct=self.config.max_single_stock_exposure,
            warning_threshold_pct=self.config.max_single_stock_exposure * Decimal('0.8')
        )
        
        # Sector limits
        limits['sector_default'] = ExposureLimit(
            exposure_type=ExposureType.SECTOR,
            identifier='default',
            max_exposure_pct=self.config.max_sector_exposure,
            warning_threshold_pct=self.config.max_sector_exposure * Decimal('0.8')
        )
        
        # Correlation limits
        limits['correlation_default'] = ExposureLimit(
            exposure_type=ExposureType.CORRELATION_GROUP,
            identifier='default',
            max_exposure_pct=self.config.max_correlation_exposure,
            warning_threshold_pct=self.config.max_correlation_exposure * Decimal('0.8')
        )
        
        return limits
    
    def add_sector_mapping(self, symbol: Symbol, sector: str) -> None:
        """Add sector mapping for a symbol."""
        self.sector_mappings[symbol] = sector
        logger.debug(f"Added sector mapping: {symbol} -> {sector}")
    
    def add_asset_class_mapping(self, symbol: Symbol, asset_class: str) -> None:
        """Add asset class mapping for a symbol."""
        self.asset_class_mappings[symbol] = asset_class
        logger.debug(f"Added asset class mapping: {symbol} -> {asset_class}")
    
    def add_correlation_group(self, symbol: Symbol, group: str) -> None:
        """Add correlation group mapping for a symbol."""
        self.correlation_groups[symbol] = group
        logger.debug(f"Added correlation group mapping: {symbol} -> {group}")
    
    def calculate_portfolio_exposure(
        self,
        positions: List[PositionRisk],
        portfolio_value: Money
    ) -> PortfolioExposure:
        """Calculate current portfolio exposure."""
        total_position_value = Money(sum(pos.market_value.amount for pos in positions))
        total_exposure_pct = total_position_value.amount / portfolio_value.amount
        cash_pct = Decimal('1') - total_exposure_pct
        
        # Calculate exposures by type
        exposures_by_type = defaultdict(list)
        
        # Sector exposures
        sector_exposures = self._calculate_sector_exposures(positions, portfolio_value)
        exposures_by_type[ExposureType.SECTOR] = sector_exposures
        
        # Asset class exposures
        asset_class_exposures = self._calculate_asset_class_exposures(positions, portfolio_value)
        exposures_by_type[ExposureType.ASSET_CLASS] = asset_class_exposures
        
        # Correlation group exposures
        correlation_exposures = self._calculate_correlation_exposures(positions, portfolio_value)
        exposures_by_type[ExposureType.CORRELATION_GROUP] = correlation_exposures
        
        # Calculate concentration risk score
        concentration_score = self._calculate_concentration_risk(positions, portfolio_value)
        
        return PortfolioExposure(
            total_value=portfolio_value,
            total_exposure_pct=total_exposure_pct,
            cash_pct=cash_pct,
            position_count=len(positions),
            exposures_by_type=dict(exposures_by_type),
            concentration_risk_score=concentration_score
        )
    
    def _calculate_sector_exposures(
        self,
        positions: List[PositionRisk],
        portfolio_value: Money
    ) -> List[ExposureLimit]:
        """Calculate sector exposures."""
        sector_values = defaultdict(Decimal)
        sector_counts = defaultdict(int)
        
        for position in positions:
            sector = self.sector_mappings.get(position.symbol, 'Unknown')
            sector_values[sector] += position.market_value.amount
            sector_counts[sector] += 1
        
        exposures = []
        for sector, value in sector_values.items():
            exposure_pct = value / portfolio_value.amount
            max_exposure = self.config.max_sector_exposure
            
            exposure = ExposureLimit(
                exposure_type=ExposureType.SECTOR,
                identifier=sector,
                max_exposure_pct=max_exposure,
                current_exposure_pct=exposure_pct,
                positions_count=sector_counts[sector],
                warning_threshold_pct=max_exposure * Decimal('0.8')
            )
            exposures.append(exposure)
        
        return exposures
    
    def _calculate_asset_class_exposures(
        self,
        positions: List[PositionRisk],
        portfolio_value: Money
    ) -> List[ExposureLimit]:
        """Calculate asset class exposures."""
        class_values = defaultdict(Decimal)
        class_counts = defaultdict(int)
        
        for position in positions:
            asset_class = self.asset_class_mappings.get(position.symbol, 'Equity')
            class_values[asset_class] += position.market_value.amount
            class_counts[asset_class] += 1
        
        exposures = []
        for asset_class, value in class_values.items():
            exposure_pct = value / portfolio_value.amount
            # Different asset classes may have different limits
            max_exposure = Decimal('1.0')  # Default 100% for single asset class
            
            exposure = ExposureLimit(
                exposure_type=ExposureType.ASSET_CLASS,
                identifier=asset_class,
                max_exposure_pct=max_exposure,
                current_exposure_pct=exposure_pct,
                positions_count=class_counts[asset_class]
            )
            exposures.append(exposure)
        
        return exposures
    
    def _calculate_correlation_exposures(
        self,
        positions: List[PositionRisk],
        portfolio_value: Money
    ) -> List[ExposureLimit]:
        """Calculate correlation group exposures."""
        group_values = defaultdict(Decimal)
        group_counts = defaultdict(int)
        
        for position in positions:
            group = self.correlation_groups.get(position.symbol, 'default')
            group_values[group] += position.market_value.amount
            group_counts[group] += 1
        
        exposures = []
        for group, value in group_values.items():
            exposure_pct = value / portfolio_value.amount
            max_exposure = self.config.max_correlation_exposure
            
            exposure = ExposureLimit(
                exposure_type=ExposureType.CORRELATION_GROUP,
                identifier=group,
                max_exposure_pct=max_exposure,
                current_exposure_pct=exposure_pct,
                positions_count=group_counts[group],
                warning_threshold_pct=max_exposure * Decimal('0.8')
            )
            exposures.append(exposure)
        
        return exposures
    
    def _calculate_concentration_risk(
        self,
        positions: List[PositionRisk],
        portfolio_value: Money
    ) -> Decimal:
        """Calculate concentration risk score (0-100)."""
        if not positions:
            return Decimal('0')
        
        # Calculate Herfindahl-Hirschman Index
        hhi = Decimal('0')
        for position in positions:
            weight = position.market_value.amount / portfolio_value.amount
            hhi += weight ** 2
        
        # Normalize to 0-100 scale
        # HHI ranges from 1/n (perfectly diversified) to 1 (single position)
        n = len(positions)
        min_hhi = Decimal('1') / Decimal(str(n))
        max_hhi = Decimal('1')
        
        if max_hhi > min_hhi:
            concentration_score = (hhi - min_hhi) / (max_hhi - min_hhi) * 100
        else:
            concentration_score = Decimal('0')
        
        return min(concentration_score, Decimal('100'))
    
    def check_exposure_limits(
        self,
        positions: List[PositionRisk],
        portfolio_value: Money
    ) -> List[RiskEvent]:
        """Check all exposure limits and generate risk events."""
        risk_events = []
        exposure = self.calculate_portfolio_exposure(positions, portfolio_value)
        
        # Check total portfolio exposure
        if exposure.total_exposure_pct > self.config.max_portfolio_exposure:
            event = RiskEvent(
                event_type=RiskEventType.CONCENTRATION_RISK,
                timestamp=datetime.now(),
                symbol=Symbol('PORTFOLIO'),
                risk_level=RiskLevel.HIGH,
                message=f"Total portfolio exposure {exposure.total_exposure_pct:.2%} exceeds limit {self.config.max_portfolio_exposure:.2%}",
                data={
                    'exposure_type': 'total_portfolio',
                    'current_exposure': float(exposure.total_exposure_pct),
                    'limit': float(self.config.max_portfolio_exposure)
                }
            )
            risk_events.append(event)
        
        # Check individual position limits
        for position in positions:
            position_pct = position.market_value.amount / portfolio_value.amount
            if position_pct > self.config.max_single_stock_exposure:
                event = RiskEvent(
                    event_type=RiskEventType.POSITION_SIZE_EXCEEDED,
                    timestamp=datetime.now(),
                    symbol=position.symbol,
                    risk_level=RiskLevel.HIGH,
                    message=f"Position size {position_pct:.2%} exceeds single stock limit {self.config.max_single_stock_exposure:.2%}",
                    data={
                        'exposure_type': 'single_position',
                        'current_exposure': float(position_pct),
                        'limit': float(self.config.max_single_stock_exposure)
                    }
                )
                risk_events.append(event)
        
        # Check sector limits
        for sector_exposure in exposure.exposures_by_type.get(ExposureType.SECTOR, []):
            if sector_exposure.current_exposure_pct > sector_exposure.max_exposure_pct:
                event = RiskEvent(
                    event_type=RiskEventType.CONCENTRATION_RISK,
                    timestamp=datetime.now(),
                    symbol=Symbol(f"SECTOR_{sector_exposure.identifier}"),
                    risk_level=RiskLevel.MEDIUM,
                    message=f"Sector exposure {sector_exposure.current_exposure_pct:.2%} exceeds limit {sector_exposure.max_exposure_pct:.2%}",
                    data={
                        'exposure_type': 'sector',
                        'sector': sector_exposure.identifier,
                        'current_exposure': float(sector_exposure.current_exposure_pct),
                        'limit': float(sector_exposure.max_exposure_pct)
                    }
                )
                risk_events.append(event)
        
        # Check correlation limits
        for corr_exposure in exposure.exposures_by_type.get(ExposureType.CORRELATION_GROUP, []):
            if corr_exposure.current_exposure_pct > corr_exposure.max_exposure_pct:
                event = RiskEvent(
                    event_type=RiskEventType.CORRELATION_RISK,
                    timestamp=datetime.now(),
                    symbol=Symbol(f"CORR_{corr_exposure.identifier}"),
                    risk_level=RiskLevel.MEDIUM,
                    message=f"Correlation group exposure {corr_exposure.current_exposure_pct:.2%} exceeds limit {corr_exposure.max_exposure_pct:.2%}",
                    data={
                        'exposure_type': 'correlation_group',
                        'group': corr_exposure.identifier,
                        'current_exposure': float(corr_exposure.current_exposure_pct),
                        'limit': float(corr_exposure.max_exposure_pct)
                    }
                )
                risk_events.append(event)
        
        # Publish all risk events
        for event in risk_events:
            publish_risk_event(event)
        
        if risk_events:
            logger.warning(f"Portfolio exposure limit violations: {len(risk_events)} events")
        
        return risk_events
    
    def can_add_position(
        self,
        symbol: Symbol,
        position_value: Money,
        current_positions: List[PositionRisk],
        portfolio_value: Money
    ) -> tuple[bool, List[str]]:
        """Check if a new position can be added without violating limits."""
        warnings = []
        can_add = True
        
        position_pct = position_value.amount / portfolio_value.amount
        
        # Check single position limit
        if position_pct > self.config.max_single_stock_exposure:
            warnings.append(
                f"Position size {position_pct:.2%} exceeds single stock limit {self.config.max_single_stock_exposure:.2%}"
            )
            can_add = False
        
        # Check total portfolio exposure
        current_exposure = sum(pos.market_value.amount for pos in current_positions)
        new_total_exposure = (current_exposure + position_value.amount) / portfolio_value.amount
        
        if new_total_exposure > self.config.max_portfolio_exposure:
            warnings.append(
                f"New total exposure {new_total_exposure:.2%} would exceed limit {self.config.max_portfolio_exposure:.2%}"
            )
            can_add = False
        
        # Check sector limits
        symbol_sector = self.sector_mappings.get(symbol, 'Unknown')
        sector_exposure = sum(
            pos.market_value.amount for pos in current_positions
            if self.sector_mappings.get(pos.symbol, 'Unknown') == symbol_sector
        )
        new_sector_exposure = (sector_exposure + position_value.amount) / portfolio_value.amount
        
        if new_sector_exposure > self.config.max_sector_exposure:
            warnings.append(
                f"New sector exposure for {symbol_sector} {new_sector_exposure:.2%} would exceed limit {self.config.max_sector_exposure:.2%}"
            )
            can_add = False
        
        # Check correlation limits
        symbol_group = self.correlation_groups.get(symbol, 'default')
        group_exposure = sum(
            pos.market_value.amount for pos in current_positions
            if self.correlation_groups.get(pos.symbol, 'default') == symbol_group
        )
        new_group_exposure = (group_exposure + position_value.amount) / portfolio_value.amount
        
        if new_group_exposure > self.config.max_correlation_exposure:
            warnings.append(
                f"New correlation group exposure {new_group_exposure:.2%} would exceed limit {self.config.max_correlation_exposure:.2%}"
            )
            can_add = False
        
        return can_add, warnings
    
    def suggest_position_reduction(
        self,
        positions: List[PositionRisk],
        portfolio_value: Money
    ) -> List[tuple[Symbol, Decimal]]:
        """Suggest position reductions to comply with limits."""
        suggestions = []
        
        # Find positions that exceed single position limit
        for position in positions:
            position_pct = position.market_value.amount / portfolio_value.amount
            if position_pct > self.config.max_single_stock_exposure:
                target_pct = self.config.max_single_stock_exposure * Decimal('0.9')  # 10% buffer
                reduction_pct = (position_pct - target_pct) / position_pct
                suggestions.append((position.symbol, reduction_pct))
        
        # Find sector overexposure
        sector_exposures = self._calculate_sector_exposures(positions, portfolio_value)
        for sector_exp in sector_exposures:
            if sector_exp.current_exposure_pct > sector_exp.max_exposure_pct:
                # Suggest reducing all positions in this sector proportionally
                sector_positions = [
                    pos for pos in positions
                    if self.sector_mappings.get(pos.symbol, 'Unknown') == sector_exp.identifier
                ]
                
                target_pct = sector_exp.max_exposure_pct * Decimal('0.9')  # 10% buffer
                reduction_factor = (sector_exp.current_exposure_pct - target_pct) / sector_exp.current_exposure_pct
                
                for pos in sector_positions:
                    suggestions.append((pos.symbol, reduction_factor))
        
        return suggestions
    
    def get_exposure_summary(
        self,
        positions: List[PositionRisk],
        portfolio_value: Money
    ) -> Dict[str, any]:
        """Get portfolio exposure summary."""
        exposure = self.calculate_portfolio_exposure(positions, portfolio_value)
        
        return {
            'total_exposure_pct': float(exposure.total_exposure_pct),
            'cash_pct': float(exposure.cash_pct),
            'position_count': exposure.position_count,
            'concentration_risk_score': float(exposure.concentration_risk_score),
            'sector_exposures': [
                {
                    'sector': exp.identifier,
                    'exposure_pct': float(exp.current_exposure_pct),
                    'limit_pct': float(exp.max_exposure_pct),
                    'positions_count': exp.positions_count
                }
                for exp in exposure.exposures_by_type.get(ExposureType.SECTOR, [])
            ],
            'top_positions': [
                {
                    'symbol': str(pos.symbol),
                    'exposure_pct': float(pos.market_value.amount / portfolio_value.amount),
                    'market_value': float(pos.market_value.amount)
                }
                for pos in sorted(positions, key=lambda p: p.market_value.amount, reverse=True)[:10]
            ]
        }


# Global portfolio risk limiter
_portfolio_limiter = None


def get_portfolio_limiter() -> PortfolioRiskLimiter:
    """Get global portfolio risk limiter."""
    global _portfolio_limiter
    if _portfolio_limiter is None:
        _portfolio_limiter = PortfolioRiskLimiter()
    return _portfolio_limiter