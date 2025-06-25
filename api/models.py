"""
Pydantic models for API request/response objects.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, validator


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"


class TradeRequest(BaseModel):
    """Model for trade execution requests."""
    symbol: str = Field(..., description="Trading symbol (e.g., 'AAPL')")
    side: OrderSide = Field(..., description="Order side (buy/sell)")
    quantity: float = Field(..., gt=0, description="Number of shares")
    order_type: OrderType = Field(default=OrderType.MARKET, description="Order type")
    limit_price: Optional[float] = Field(None, description="Limit price for limit orders")
    stop_price: Optional[float] = Field(None, description="Stop price for stop orders")
    time_in_force: str = Field(default="day", description="Time in force")
    
    @validator('symbol')
    def symbol_must_be_uppercase(cls, v):
        return v.upper()


class TradeResponse(BaseModel):
    """Model for trade execution responses."""
    order_id: str = Field(..., description="Unique order identifier")
    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Order side")
    quantity: float = Field(..., description="Order quantity")
    status: OrderStatus = Field(..., description="Order status")
    timestamp: datetime = Field(..., description="Order timestamp")
    filled_qty: Optional[float] = Field(None, description="Filled quantity")
    avg_fill_price: Optional[float] = Field(None, description="Average fill price")


class Position(BaseModel):
    """Model for portfolio positions."""
    symbol: str = Field(..., description="Trading symbol")
    quantity: float = Field(..., description="Position size")
    market_value: float = Field(..., description="Current market value")
    avg_cost: float = Field(..., description="Average cost basis")
    unrealized_pnl: float = Field(..., description="Unrealized P&L")
    percent_change: float = Field(..., description="Percent change from cost basis")


class PortfolioSummary(BaseModel):
    """Model for portfolio summary."""
    total_value: float = Field(..., description="Total portfolio value")
    available_cash: float = Field(..., description="Available cash")
    positions: List[Position] = Field(default=[], description="Current positions")
    daily_pnl: float = Field(..., description="Daily P&L")
    total_pnl: float = Field(..., description="Total unrealized P&L")
    buying_power: Optional[float] = Field(None, description="Available buying power")


class MarketData(BaseModel):
    """Model for market data."""
    symbol: str = Field(..., description="Trading symbol")
    price: float = Field(..., description="Current price")
    bid: Optional[float] = Field(None, description="Bid price")
    ask: Optional[float] = Field(None, description="Ask price")
    volume: Optional[int] = Field(None, description="Volume")
    timestamp: datetime = Field(..., description="Data timestamp")


class StrategyConfig(BaseModel):
    """Model for strategy configuration."""
    name: str = Field(..., description="Strategy name")
    enabled: bool = Field(default=False, description="Strategy enabled status")
    parameters: Dict[str, Any] = Field(default={}, description="Strategy parameters")
    symbols: List[str] = Field(default=[], description="Trading symbols")
    risk_limits: Dict[str, float] = Field(default={}, description="Risk limits")


class BacktestRequest(BaseModel):
    """Model for backtest requests."""
    strategy_name: str = Field(..., description="Strategy to backtest")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: float = Field(default=100000.0, description="Initial capital")
    symbols: List[str] = Field(..., description="Symbols to trade")
    parameters: Dict[str, Any] = Field(default={}, description="Strategy parameters")


class BacktestResult(BaseModel):
    """Model for backtest results."""
    strategy_name: str = Field(..., description="Strategy name")
    total_return: float = Field(..., description="Total return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    win_rate: float = Field(..., description="Win rate percentage")
    total_trades: int = Field(..., description="Total number of trades")
    profit_factor: float = Field(..., description="Profit factor")
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata") 