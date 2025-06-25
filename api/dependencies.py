"""
FastAPI dependency injection functions.
"""

import os
from typing import Generator

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import Environment

from utils.config import get_settings
from utils.logging import get_logger

logger = get_logger(__name__)


def get_settings():
    """Get application settings."""
    from utils.config import Settings
    return Settings()


def get_trading_client() -> Generator[TradingClient, None, None]:
    """
    Get Alpaca trading client.
    
    Yields:
        TradingClient: Configured Alpaca trading client
    """
    settings = get_settings()
    
    try:
        # TODO: Get credentials from environment or settings
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            raise ValueError("Alpaca API credentials not found in environment")
        
        # Use paper trading by default for safety
        environment = Environment.PAPER if settings.paper_trading else Environment.LIVE
        
        client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            environment=environment
        )
        
        logger.info(f"Initialized Alpaca client (env: {environment.value})")
        yield client
        
    except Exception as e:
        logger.error(f"Failed to initialize trading client: {e}")
        raise
    
    finally:
        # Clean up if needed
        pass


def get_portfolio_manager():
    """
    Get portfolio manager instance.
    
    TODO: Implement portfolio management class
    """
    # TODO: Initialize and return portfolio manager
    class DummyPortfolioManager:
        def get_summary(self):
            return {"message": "Portfolio manager not implemented"}
    
    return DummyPortfolioManager()


def get_data_client():
    """
    Get market data client.
    
    TODO: Implement market data client for TradingView, etc.
    """
    # TODO: Initialize market data client
    class DummyDataClient:
        def get_quote(self, symbol: str):
            return {"symbol": symbol, "price": 0.0}
    
    return DummyDataClient()


def get_strategy_manager():
    """
    Get strategy manager instance.
    
    TODO: Implement strategy management system
    """
    class DummyStrategyManager:
        def list_strategies(self):
            return []
        
        def get_strategy(self, name: str):
            return None
    
    return DummyStrategyManager()


def get_risk_manager():
    """
    Get risk management instance.
    
    TODO: Implement risk management system
    """
    class DummyRiskManager:
        def check_trade(self, trade_request):
            return True
    
    return DummyRiskManager() 