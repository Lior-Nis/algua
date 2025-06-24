"""
Main FastAPI application for Algua trading platform.
"""

from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from api.models import TradeRequest, TradeResponse, PortfolioSummary
from api.dependencies import get_trading_client, get_portfolio_manager
from utils.logging import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Algua Trading API",
    description="REST API for quantitative trading platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def ping() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "algua-api"
    }


@app.get("/portfolio")
async def get_portfolio(
    portfolio_manager=Depends(get_portfolio_manager)
) -> PortfolioSummary:
    """Get current portfolio summary."""
    try:
        # TODO: Implement portfolio retrieval logic
        return PortfolioSummary(
            total_value=100000.0,
            available_cash=50000.0,
            positions=[],
            daily_pnl=0.0,
            total_pnl=0.0
        )
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch portfolio")


@app.post("/trade")
async def execute_trade(
    trade_request: TradeRequest,
    trading_client=Depends(get_trading_client)
) -> TradeResponse:
    """Execute a trade order."""
    try:
        logger.info(f"Executing trade: {trade_request}")
        
        # TODO: Implement trade execution logic
        # - Validate trade request
        # - Check risk limits
        # - Execute via Alpaca API
        # - Record in database
        # - Send alerts if needed
        
        return TradeResponse(
            order_id="dummy_order_123",
            symbol=trade_request.symbol,
            side=trade_request.side,
            quantity=trade_request.quantity,
            status="submitted",
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        raise HTTPException(status_code=400, detail=f"Trade failed: {str(e)}")


@app.get("/positions")
async def get_positions(trading_client=Depends(get_trading_client)) -> List[Dict]:
    """Get current positions."""
    try:
        # TODO: Implement position retrieval from Alpaca
        return []
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch positions")


@app.get("/market-data/{symbol}")
async def get_market_data(symbol: str) -> Dict:
    """Get real-time market data for a symbol."""
    try:
        # TODO: Implement market data retrieval
        return {
            "symbol": symbol,
            "price": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch market data")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 