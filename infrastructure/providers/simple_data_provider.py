"""
Simple data provider that works without pandas using basic Python structures.
"""

import requests
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import logging

from domain.value_objects import Symbol, Price
from infrastructure.interfaces import MarketDataProvider, TimeFrame, DataProviderFactory
from utils.logging import get_logger

logger = get_logger(__name__)


class SimpleDataProvider(MarketDataProvider):
    """Simple data provider using basic Python structures."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the provider."""
        self.api_key = api_key
        self.session = requests.Session()
        logger.info("SimpleDataProvider initialized")
    
    def get_historical_data(
        self, 
        symbol: Symbol, 
        start_date: datetime, 
        end_date: datetime, 
        timeframe: TimeFrame = TimeFrame.DAY_1
    ) -> List[Dict]:
        """
        Get historical OHLCV data.
        
        Returns:
            List of dictionaries with OHLCV data
        """
        try:
            # For demo purposes, generate synthetic data
            # In a real implementation, this would call an actual API
            data = self._generate_synthetic_data(str(symbol), start_date, end_date)
            
            logger.info(f"Retrieved {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return []
    
    def get_current_price(self, symbol: Symbol) -> Price:
        """Get current price for a symbol."""
        try:
            # For demo purposes, return a random price
            # In a real implementation, this would call an actual API
            base_prices = {
                'AAPL': 150.0,
                'MSFT': 300.0,
                'GOOGL': 120.0,
                'TSLA': 200.0,
                'AMZN': 100.0
            }
            
            base_price = base_prices.get(str(symbol).upper(), 100.0)
            # Add some random variation
            import random
            variation = random.uniform(-0.05, 0.05)  # ±5%
            current_price = base_price * (1 + variation)
            
            return Price(Decimal(str(round(current_price, 2))))
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            raise
    
    def get_multiple_prices(self, symbols: List[Symbol]) -> Dict[Symbol, Price]:
        """Get current prices for multiple symbols."""
        prices = {}
        
        for symbol in symbols:
            try:
                price = self.get_current_price(symbol)
                prices[symbol] = price
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {str(e)}")
                continue
        
        return prices
    
    def search_symbols(self, query: str) -> List[Symbol]:
        """Search for symbols matching a query."""
        # Simple symbol database
        known_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 
            'NVDA', 'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'PYPL'
        ]
        
        matching_symbols = []
        query_upper = query.upper()
        
        for symbol_str in known_symbols:
            if query_upper in symbol_str:
                matching_symbols.append(Symbol(symbol_str))
        
        return matching_symbols
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        # Simplified market hours check
        import datetime
        now = datetime.datetime.now()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check if it's within market hours (9:30 AM to 4:00 PM)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _generate_synthetic_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Generate synthetic OHLCV data for testing."""
        data = []
        current_date = start_date
        
        # Base price for different symbols
        base_prices = {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 120.0,
            'TSLA': 200.0,
            'AMZN': 100.0
        }
        
        current_price = base_prices.get(symbol.upper(), 100.0)
        
        import random
        random.seed(42)  # For reproducible results
        
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                # Generate daily price movement
                change = random.uniform(-0.03, 0.03)  # ±3% daily change
                current_price *= (1 + change)
                
                # Generate OHLC from current price
                daily_range = current_price * random.uniform(0.01, 0.04)  # 1-4% daily range
                
                high = current_price + random.uniform(0, daily_range * 0.6)
                low = current_price - random.uniform(0, daily_range * 0.4)
                open_price = low + random.uniform(0, high - low)
                close = current_price
                
                volume = random.randint(1000000, 10000000)
                
                data.append({
                    'Timestamp': current_date,
                    'Open': round(open_price, 2),
                    'High': round(high, 2),
                    'Low': round(low, 2),
                    'Close': round(close, 2),
                    'Volume': volume
                })
            
            current_date += timedelta(days=1)
        
        return data


# Register the provider
DataProviderFactory.register("simple", SimpleDataProvider)