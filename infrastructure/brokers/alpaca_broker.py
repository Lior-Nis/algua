"""
Alpaca broker implementation.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
import logging

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None

from domain.value_objects import Symbol, Money, Price, Quantity
from domain.entities import Order, Position
from infrastructure.interfaces import (
    BrokerInterface, OrderType, OrderSide, OrderStatus, BrokerFactory
)
from utils.logging import get_logger

logger = get_logger(__name__)


class AlpacaBroker(BrokerInterface):
    """Alpaca broker implementation."""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        """
        Initialize Alpaca broker.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            base_url: Alpaca API base URL (paper or live)
        """
        if tradeapi is None:
            raise ImportError("alpaca-trade-api package is required for AlpacaBroker")
        
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        
        # Initialize API client
        self.api = tradeapi.REST(
            key_id=api_key,
            secret_key=secret_key,
            base_url=base_url,
            api_version='v2'
        )
        
        # Order type mapping
        self._order_type_map = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit"
        }
        
        # Order side mapping
        self._order_side_map = {
            OrderSide.BUY: "buy",
            OrderSide.SELL: "sell"
        }
        
        logger.info(f"AlpacaBroker initialized with base URL: {base_url}")
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account details
        """
        try:
            account = self.api.get_account()
            return {
                'account_id': account.id,
                'cash': float(account.cash),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'transfers_blocked': account.transfers_blocked,
                'account_blocked': account.account_blocked,
                'created_at': account.created_at,
                'currency': account.currency
            }
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            raise
    
    def get_positions(self) -> List[Position]:
        """
        Get all current positions.
        
        Returns:
            List of current positions
        """
        try:
            positions = []
            alpaca_positions = self.api.list_positions()
            
            for pos in alpaca_positions:
                position = self._convert_alpaca_position(pos)
                positions.append(position)
            
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise
    
    def get_position(self, symbol: Symbol) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position if exists, None otherwise
        """
        try:
            pos = self.api.get_position(str(symbol))
            return self._convert_alpaca_position(pos)
        except Exception as e:
            if "position does not exist" in str(e).lower():
                return None
            logger.error(f"Error getting position for {symbol}: {str(e)}")
            raise
    
    def place_order(
        self,
        symbol: Symbol,
        side: OrderSide,
        order_type: OrderType,
        quantity: Quantity,
        price: Optional[Price] = None,
        stop_price: Optional[Price] = None,
        time_in_force: str = "day"
    ) -> str:
        """
        Place a trading order.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            order_type: Type of order
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            
        Returns:
            Order ID
        """
        try:
            order_params = {
                'symbol': str(symbol),
                'qty': str(quantity.value),
                'side': self._order_side_map[side],
                'type': self._order_type_map[order_type],
                'time_in_force': time_in_force.lower()
            }
            
            # Add price parameters based on order type
            if order_type == OrderType.LIMIT and price is not None:
                order_params['limit_price'] = str(price.value)
            elif order_type == OrderType.STOP and stop_price is not None:
                order_params['stop_price'] = str(stop_price.value)
            elif order_type == OrderType.STOP_LIMIT and price is not None and stop_price is not None:
                order_params['limit_price'] = str(price.value)
                order_params['stop_price'] = str(stop_price.value)
            
            order = self.api.submit_order(**order_params)
            
            logger.info(f"Order placed: {order.id} for {symbol}")
            return order.id
            
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {str(e)}")
            raise
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order canceled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {str(e)}")
            return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status
        """
        try:
            order = self.api.get_order(order_id)
            return self._convert_alpaca_order_status(order.status)
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {str(e)}")
            raise
    
    def get_order_history(self, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Order]:
        """
        Get order history.
        
        Args:
            start_date: Start date for history
            end_date: End date for history
            
        Returns:
            List of orders
        """
        try:
            # If no dates provided, get last 30 days
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()
            
            orders = []
            alpaca_orders = self.api.list_orders(
                status='all',
                after=start_date.isoformat(),
                until=end_date.isoformat()
            )
            
            for order in alpaca_orders:
                converted_order = self._convert_alpaca_order(order)
                orders.append(converted_order)
            
            return orders
        except Exception as e:
            logger.error(f"Error getting order history: {str(e)}")
            raise
    
    def get_buying_power(self) -> Money:
        """
        Get available buying power.
        
        Returns:
            Available buying power
        """
        try:
            account = self.api.get_account()
            return Money(Decimal(str(account.buying_power)))
        except Exception as e:
            logger.error(f"Error getting buying power: {str(e)}")
            raise
    
    def is_connected(self) -> bool:
        """
        Check if connected to broker.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            self.api.get_account()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def _convert_alpaca_position(self, alpaca_pos) -> Position:
        """Convert Alpaca position to domain Position."""
        from domain.entities.position import PositionSide, PositionStatus
        
        symbol = Symbol(alpaca_pos.symbol)
        side = PositionSide.LONG if float(alpaca_pos.qty) > 0 else PositionSide.SHORT
        quantity = Quantity(abs(float(alpaca_pos.qty)))
        avg_price = Price(Decimal(str(alpaca_pos.avg_cost)))
        current_price = Price(Decimal(str(alpaca_pos.market_value)) / Decimal(str(alpaca_pos.qty)))
        
        return Position(
            id=f"alpaca_{alpaca_pos.symbol}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            avg_price=avg_price,
            current_price=current_price,
            status=PositionStatus.OPEN
        )
    
    def _convert_alpaca_order(self, alpaca_order) -> Order:
        """Convert Alpaca order to domain Order."""
        from domain.entities.order import OrderSide as DomainOrderSide, OrderStatus as DomainOrderStatus
        
        symbol = Symbol(alpaca_order.symbol)
        side = DomainOrderSide.BUY if alpaca_order.side == "buy" else DomainOrderSide.SELL
        quantity = Quantity(Decimal(str(alpaca_order.qty)))
        price = Price(Decimal(str(alpaca_order.limit_price))) if alpaca_order.limit_price else None
        
        return Order(
            id=alpaca_order.id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=alpaca_order.type,
            status=self._convert_alpaca_order_status(alpaca_order.status),
            created_at=alpaca_order.created_at,
            updated_at=alpaca_order.updated_at
        )
    
    def _convert_alpaca_order_status(self, alpaca_status: str) -> OrderStatus:
        """Convert Alpaca order status to domain OrderStatus."""
        status_map = {
            'new': OrderStatus.PENDING,
            'accepted': OrderStatus.PENDING,
            'filled': OrderStatus.FILLED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'canceled': OrderStatus.CANCELED,
            'rejected': OrderStatus.REJECTED,
            'pending_cancel': OrderStatus.PENDING,
            'pending_replace': OrderStatus.PENDING,
            'replaced': OrderStatus.PENDING,
            'done_for_day': OrderStatus.CANCELED,
            'expired': OrderStatus.CANCELED
        }
        
        return status_map.get(alpaca_status.lower(), OrderStatus.PENDING)


# Register the broker
BrokerFactory.register("alpaca", AlpacaBroker)