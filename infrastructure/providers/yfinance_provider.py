"""
Yahoo Finance data provider implementation.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
import logging

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import yfinance as yf
except ImportError:
    yf = None

from domain.value_objects import Symbol, Price
from infrastructure.interfaces import MarketDataProvider, TimeFrame, DataProviderFactory
from utils.logging import get_logger

logger = get_logger(__name__)


class YFinanceProvider(MarketDataProvider):
    """Yahoo Finance data provider implementation."""
    
    def __init__(self):
        """Initialize the provider."""
        if yf is None:
            raise ImportError("yfinance package is required for YFinanceProvider")
        
        if pd is None:
            raise ImportError("pandas package is required for YFinanceProvider")
        
        self._timeframe_map = {
            TimeFrame.MINUTE_1: "1m",
            TimeFrame.MINUTE_5: "5m",
            TimeFrame.MINUTE_15: "15m",
            TimeFrame.MINUTE_30: "30m",
            TimeFrame.HOUR_1: "1h",
            TimeFrame.DAY_1: "1d",
            TimeFrame.WEEK_1: "1wk",
            TimeFrame.MONTH_1: "1mo"
        }
        
        logger.info("YFinanceProvider initialized")
    
    def get_historical_data(
        self, 
        symbol: Symbol, 
        start_date: datetime, 
        end_date: datetime, 
        timeframe: TimeFrame = TimeFrame.DAY_1
    ) -> Any:
        """
        Get historical OHLCV data from Yahoo Finance.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            timeframe: Time frame for the data
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Timestamp
        """
        try:
            ticker = yf.Ticker(str(symbol))
            interval = self._timeframe_map.get(timeframe, "1d")
            
            # Download data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                logger.warning(f"No data found for {symbol} from {start_date} to {end_date}")
                return pd.DataFrame() if pd else []
            
            # Standardize column names
            data = data.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # Add timestamp column
            data['Timestamp'] = data.index
            
            # Reset index to make timestamp a column
            data = data.reset_index(drop=True)
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Timestamp']
            for col in required_columns:
                if col not in data.columns:
                    logger.error(f"Missing required column: {col}")
                    return pd.DataFrame() if pd else []
            
            logger.info(f"Retrieved {len(data)} rows of data for {symbol}")
            return data[required_columns]
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame() if pd else []
    
    def get_current_price(self, symbol: Symbol) -> Price:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price
        """
        try:
            ticker = yf.Ticker(str(symbol))
            info = ticker.info
            
            # Try different price fields
            price_value = None
            for field in ['regularMarketPrice', 'currentPrice', 'price', 'previousClose']:
                if field in info and info[field] is not None:
                    price_value = info[field]
                    break
            
            if price_value is None:
                # Fallback to recent data
                data = ticker.history(period="1d", interval="1m")
                if not data.empty:
                    price_value = data['Close'].iloc[-1]
            
            if price_value is None:
                raise ValueError(f"Could not get current price for {symbol}")
            
            return Price(Decimal(str(price_value)))
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            raise
    
    def get_multiple_prices(self, symbols: List[Symbol]) -> Dict[Symbol, Price]:
        """
        Get current prices for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary mapping symbols to current prices
        """
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
        """
        Search for symbols matching a query.
        
        Note: Yahoo Finance doesn't have a direct search API,
        so this is a basic implementation that tries common formats.
        
        Args:
            query: Search query
            
        Returns:
            List of matching symbols
        """
        # Basic symbol search - try the query as-is and with common suffixes
        potential_symbols = [
            query.upper(),
            f"{query.upper()}.L",  # London Stock Exchange
            f"{query.upper()}.TO", # Toronto Stock Exchange
            f"{query.upper()}.AX", # Australian Securities Exchange
        ]
        
        valid_symbols = []
        
        for symbol_str in potential_symbols:
            try:
                symbol = Symbol(symbol_str)
                ticker = yf.Ticker(symbol_str)
                
                # Try to get basic info to validate the symbol
                info = ticker.info
                if info and 'symbol' in info:
                    valid_symbols.append(symbol)
                    break  # Found a valid symbol, stop searching
                    
            except Exception:
                continue
        
        return valid_symbols
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.
        
        Note: This is a simplified implementation.
        For more accurate market hours, consider using a dedicated market calendar.
        
        Returns:
            True if market is open, False otherwise
        """
        try:
            # Get current time in EST (US market timezone)
            import pytz
            est = pytz.timezone('US/Eastern')
            current_time = datetime.now(est)
            
            # Check if it's a weekday
            if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check if it's within market hours (9:30 AM to 4:00 PM EST)
            market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
            return market_open <= current_time <= market_close
            
        except ImportError:
            # If pytz is not available, assume market is open during weekdays
            current_time = datetime.now()
            return current_time.weekday() < 5
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            return False


# Register the provider
DataProviderFactory.register("yfinance", YFinanceProvider)