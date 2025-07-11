"""
Fill handling and slippage calculation for order execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import random

from domain.value_objects import Symbol, Price, Quantity, Money
from utils.logging import get_logger

logger = get_logger(__name__)


class FillType(Enum):
    """Types of fills."""
    FULL = "full"
    PARTIAL = "partial"
    SLIPPED = "slipped"


@dataclass
class Fill:
    """Individual fill record."""
    fill_id: str
    order_id: str
    symbol: Symbol
    quantity: Quantity
    price: Price
    timestamp: datetime
    fill_type: FillType
    commission: Money
    venue: Optional[str] = None
    slippage: Optional[Money] = None
    market_impact: Optional[Decimal] = None
    
    def get_total_value(self) -> Money:
        """Get total value of the fill including commission."""
        gross_value = Money(self.quantity.value * self.price.value)
        return Money(gross_value.amount + self.commission.amount)
    
    def get_net_value(self) -> Money:
        """Get net value of the fill excluding commission."""
        return Money(self.quantity.value * self.price.value)


@dataclass
class PartialFill:
    """Partial fill information."""
    remaining_quantity: Quantity
    fill_reason: str
    next_attempt_time: Optional[datetime] = None


class SlippageModel(ABC):
    """Abstract base class for slippage models."""
    
    @abstractmethod
    def calculate_slippage(
        self,
        symbol: Symbol,
        quantity: Quantity,
        order_side: str,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate slippage as percentage of price."""
        pass


class LinearSlippageModel(SlippageModel):
    """Linear slippage model based on order size."""
    
    def __init__(self, base_slippage: Decimal = Decimal('0.001'), size_factor: Decimal = Decimal('0.00001')):
        self.base_slippage = base_slippage  # 0.1% base slippage
        self.size_factor = size_factor      # Additional slippage per $1000 of order
    
    def calculate_slippage(
        self,
        symbol: Symbol,
        quantity: Quantity,
        order_side: str,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate linear slippage."""
        current_price = market_data.get('current_price', Decimal('100'))
        order_value = quantity.value * current_price
        
        # Base slippage + size impact
        size_impact = (order_value / Decimal('1000')) * self.size_factor
        total_slippage = self.base_slippage + size_impact
        
        # Cap slippage at reasonable levels
        return min(total_slippage, Decimal('0.01'))  # Max 1% slippage


class SquareRootSlippageModel(SlippageModel):
    """Square root slippage model (more realistic market impact)."""
    
    def __init__(self, impact_factor: Decimal = Decimal('0.0001')):
        self.impact_factor = impact_factor
    
    def calculate_slippage(
        self,
        symbol: Symbol,
        quantity: Quantity,
        order_side: str,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate square root slippage."""
        current_price = market_data.get('current_price', Decimal('100'))
        daily_volume = market_data.get('daily_volume', Decimal('1000000'))
        
        if daily_volume <= 0:
            return Decimal('0.001')  # Default slippage
        
        # Order size as percentage of daily volume
        order_value = quantity.value * current_price
        volume_ratio = order_value / daily_volume
        
        # Square root impact model
        slippage = self.impact_factor * (volume_ratio ** Decimal('0.5'))
        
        return min(slippage, Decimal('0.02'))  # Max 2% slippage


class MarketImpactModel(SlippageModel):
    """Market impact model with bid-ask spread consideration."""
    
    def __init__(self, spread_factor: Decimal = Decimal('0.5')):
        self.spread_factor = spread_factor  # Portion of spread to pay
    
    def calculate_slippage(
        self,
        symbol: Symbol,
        quantity: Quantity,
        order_side: str,
        market_data: Dict[str, Any]
    ) -> Decimal:
        """Calculate market impact slippage."""
        bid = market_data.get('bid', Decimal('100'))
        ask = market_data.get('ask', Decimal('100.10'))
        current_price = (bid + ask) / Decimal('2')
        
        # Bid-ask spread impact
        spread = ask - bid
        spread_cost = spread * self.spread_factor
        
        # Additional impact based on order size
        daily_volume = market_data.get('daily_volume', Decimal('1000000'))
        order_value = quantity.value * current_price
        
        if daily_volume > 0:
            volume_impact = (order_value / daily_volume) * Decimal('0.001')
        else:
            volume_impact = Decimal('0')
        
        total_slippage = (spread_cost + volume_impact) / current_price
        
        return min(total_slippage, Decimal('0.015'))  # Max 1.5% slippage


class SlippageCalculator:
    """Calculator for order slippage."""
    
    def __init__(self, model: SlippageModel = None):
        self.model = model or LinearSlippageModel()
        self.volatility_factor = Decimal('1.0')
        self.time_factor = Decimal('1.0')
    
    def calculate_execution_price(
        self,
        intended_price: Price,
        symbol: Symbol,
        quantity: Quantity,
        order_side: str,
        market_data: Dict[str, Any]
    ) -> Tuple[Price, Decimal]:
        """Calculate actual execution price with slippage."""
        # Get base slippage from model
        base_slippage = self.model.calculate_slippage(symbol, quantity, order_side, market_data)
        
        # Apply volatility factor
        volatility = market_data.get('volatility', Decimal('0.2'))
        volatility_adjustment = volatility * Decimal('0.1')  # 10% of volatility
        
        # Apply time factor (market conditions)
        time_adjustment = self._get_time_factor() * Decimal('0.05')
        
        # Total slippage
        total_slippage = base_slippage + volatility_adjustment + time_adjustment
        
        # Apply slippage direction
        if order_side.lower() == 'buy':
            # Buying typically causes price to go up
            slipped_price = intended_price.value * (Decimal('1') + total_slippage)
        else:
            # Selling typically causes price to go down
            slipped_price = intended_price.value * (Decimal('1') - total_slippage)
        
        return Price(slipped_price), total_slippage
    
    def _get_time_factor(self) -> Decimal:
        """Get time-based slippage factor."""
        # Simplified: use random factor to simulate market conditions
        return Decimal(str(random.uniform(0.8, 1.2)))
    
    def set_market_conditions(self, volatility_factor: Decimal, time_factor: Decimal) -> None:
        """Set market condition factors."""
        self.volatility_factor = volatility_factor
        self.time_factor = time_factor


class FillHandler:
    """Handler for order fills and execution simulation."""
    
    def __init__(self, slippage_calculator: SlippageCalculator = None):
        self.slippage_calculator = slippage_calculator or SlippageCalculator()
        self.commission_rate = Decimal('0.001')  # 0.1% commission
        self.min_commission = Money(Decimal('1.00'))  # $1 minimum
        self.fill_counter = 0
    
    def execute_order(
        self,
        order: 'Order',
        market_data: Dict[str, Any],
        partial_fill_probability: Decimal = Decimal('0.1')
    ) -> Tuple[List[Fill], Optional[PartialFill]]:
        """Execute an order and return fills."""
        fills = []
        partial_fill = None
        
        # Check if order should be partially filled
        if random.random() < float(partial_fill_probability) and order.quantity.value > Decimal('10'):
            # Partial fill
            fill_ratio = Decimal(str(random.uniform(0.3, 0.8)))
            fill_quantity = Quantity(order.quantity.value * fill_ratio)
            remaining = Quantity(order.quantity.value - fill_quantity.value)
            
            partial_fill = PartialFill(
                remaining_quantity=remaining,
                fill_reason="Liquidity constraints",
                next_attempt_time=datetime.now()
            )
        else:
            # Full fill
            fill_quantity = order.quantity
        
        # Calculate execution price with slippage
        intended_price = order.get_execution_price(Price(market_data.get('current_price', Decimal('100'))))
        execution_price, slippage_pct = self.slippage_calculator.calculate_execution_price(
            intended_price,
            order.symbol,
            fill_quantity,
            order.side.value,
            market_data
        )
        
        # Calculate commission
        commission = self._calculate_commission(fill_quantity, execution_price)
        
        # Calculate slippage amount
        slippage_amount = Money(abs(execution_price.value - intended_price.value) * fill_quantity.value)
        
        # Create fill
        self.fill_counter += 1
        fill = Fill(
            fill_id=f"fill_{self.fill_counter}",
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=fill_quantity,
            price=execution_price,
            timestamp=datetime.now(),
            fill_type=FillType.PARTIAL if partial_fill else FillType.FULL,
            commission=commission,
            slippage=slippage_amount,
            market_impact=slippage_pct
        )
        
        fills.append(fill)
        
        logger.info(
            f"Order {order.order_id} executed: {fill_quantity.value} @ {execution_price.value:.2f} "
            f"(slippage: {slippage_pct:.4f}, commission: ${commission.amount:.2f})"
        )
        
        return fills, partial_fill
    
    def _calculate_commission(self, quantity: Quantity, price: Price) -> Money:
        """Calculate commission for a fill."""
        order_value = quantity.value * price.value
        commission = order_value * self.commission_rate
        
        # Apply minimum commission
        return Money(max(commission, self.min_commission.amount))
    
    def simulate_realistic_fills(
        self,
        order: 'Order',
        market_data: Dict[str, Any],
        fill_strategy: str = "aggressive"
    ) -> List[Fill]:
        """Simulate realistic fills with multiple partial fills."""
        fills = []
        remaining_quantity = order.quantity.value
        
        if fill_strategy == "aggressive":
            # Fill in 1-2 chunks
            fill_count = random.randint(1, 2)
        elif fill_strategy == "passive":
            # Fill in 2-5 chunks
            fill_count = random.randint(2, 5)
        else:
            # Conservative: fill in 3-8 chunks
            fill_count = random.randint(3, 8)
        
        # Don't over-chunk small orders
        fill_count = min(fill_count, max(1, int(remaining_quantity / Decimal('10'))))
        
        for i in range(fill_count):
            if remaining_quantity <= 0:
                break
            
            # Determine fill size
            if i == fill_count - 1:
                # Last fill gets remaining quantity
                fill_qty = remaining_quantity
            else:
                # Random portion of remaining
                max_fill = remaining_quantity * Decimal('0.6')
                min_fill = remaining_quantity * Decimal('0.1')
                fill_qty = Decimal(str(random.uniform(float(min_fill), float(max_fill))))
            
            # Slight price variation for each fill
            base_price = market_data.get('current_price', Decimal('100'))
            price_variation = Decimal(str(random.uniform(-0.002, 0.002)))  # ±0.2% variation
            varied_price = base_price * (Decimal('1') + price_variation)
            market_data_copy = market_data.copy()
            market_data_copy['current_price'] = varied_price
            
            # Create temporary order for this fill
            temp_order = type(order)(
                order.symbol,
                order.side,
                Quantity(fill_qty),
                order.order_type if hasattr(order, 'order_type') else None
            )
            temp_order.order_id = order.order_id
            
            # Execute fill
            fill_list, _ = self.execute_order(temp_order, market_data_copy, Decimal('0'))
            if fill_list:
                fills.extend(fill_list)
                remaining_quantity -= fill_qty
            
            # Small delay simulation
            import time
            time.sleep(0.001)  # 1ms delay
        
        return fills
    
    def get_fill_statistics(self, fills: List[Fill]) -> Dict[str, Any]:
        """Get statistics for a list of fills."""
        if not fills:
            return {}
        
        total_quantity = sum(fill.quantity.value for fill in fills)
        total_value = sum(fill.get_net_value().amount for fill in fills)
        total_commission = sum(fill.commission.amount for fill in fills)
        total_slippage = sum(fill.slippage.amount for fill in fills if fill.slippage)
        
        average_price = total_value / total_quantity if total_quantity > 0 else Decimal('0')
        
        return {
            'fill_count': len(fills),
            'total_quantity': float(total_quantity),
            'total_value': float(total_value),
            'average_price': float(average_price),
            'total_commission': float(total_commission),
            'total_slippage': float(total_slippage),
            'commission_rate': float(total_commission / total_value) if total_value > 0 else 0,
            'slippage_rate': float(total_slippage / total_value) if total_value > 0 else 0,
            'first_fill_time': fills[0].timestamp.isoformat(),
            'last_fill_time': fills[-1].timestamp.isoformat()
        }


# Import Order here to avoid circular imports
if False:  # Type checking only
    from .order_types import Order