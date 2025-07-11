"""
Abstract interfaces for pluggable data providers and brokers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum

try:
    import pandas as pd
except ImportError:
    pd = None

from domain.value_objects import Symbol, Money, Price, Quantity


class TimeFrame(Enum):
    """Time frame for historical data."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


class OrderType(Enum):
    """Order types for trading."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side for trading."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class MarketDataProvider(ABC):
    """Abstract interface for market data providers."""
    
    @abstractmethod
    def get_historical_data(
        self, 
        symbol: Symbol, 
        start_date: datetime, 
        end_date: datetime, 
        timeframe: TimeFrame = TimeFrame.DAY_1
    ) -> Union[List[Dict], Any]:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            timeframe: Time frame for the data
            
        Returns:
            List of dictionaries or DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: Symbol) -> Price:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price
        """
        pass
    
    @abstractmethod
    def get_multiple_prices(self, symbols: List[Symbol]) -> Dict[Symbol, Price]:
        """
        Get current prices for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary mapping symbols to current prices
        """
        pass
    
    @abstractmethod
    def search_symbols(self, query: str) -> List[Symbol]:
        """
        Search for symbols matching a query.
        
        Args:
            query: Search query
            
        Returns:
            List of matching symbols
        """
        pass
    
    @abstractmethod
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.
        
        Returns:
            True if market is open, False otherwise
        """
        pass


class BrokerInterface(ABC):
    """Abstract interface for broker integration."""
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account details (cash, equity, etc.)
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Any]:
        """
        Get all current positions.
        
        Returns:
            List of current positions
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: Symbol) -> Optional[Any]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position if exists, None otherwise
        """
        pass
    
    @abstractmethod
    def place_order(
        self,
        symbol: Symbol,
        side: OrderSide,
        order_type: OrderType,
        quantity: Quantity,
        price: Optional[Price] = None,
        stop_price: Optional[Price] = None,
        time_in_force: str = "DAY"
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
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status
        """
        pass
    
    @abstractmethod
    def get_order_history(self, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Any]:
        """
        Get order history.
        
        Args:
            start_date: Start date for history
            end_date: End date for history
            
        Returns:
            List of orders
        """
        pass
    
    @abstractmethod
    def get_buying_power(self) -> Money:
        """
        Get available buying power.
        
        Returns:
            Available buying power
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if connected to broker.
        
        Returns:
            True if connected, False otherwise
        """
        pass


class DataProviderFactory:
    """Factory for creating data providers."""
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: type):
        """Register a data provider."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> MarketDataProvider:
        """Create a data provider instance."""
        if name not in cls._providers:
            raise ValueError(f"Unknown data provider: {name}")
        
        return cls._providers[name](**kwargs)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List available data providers."""
        return list(cls._providers.keys())


class BrokerFactory:
    """Factory for creating brokers."""
    
    _brokers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, broker_class: type):
        """Register a broker."""
        cls._brokers[name] = broker_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BrokerInterface:
        """Create a broker instance."""
        if name not in cls._brokers:
            raise ValueError(f"Unknown broker: {name}")
        
        return cls._brokers[name](**kwargs)
    
    @classmethod
    def list_brokers(cls) -> List[str]:
        """List available brokers."""
        return list(cls._brokers.keys())