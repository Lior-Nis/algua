#!/usr/bin/env python3
"""
Script to run backtests for trading strategies.
"""

import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.append('.')

from backtesting.engine import BacktestEngine
from models.strategies import StrategyFactory
from data_ingestion.collectors import DataCollectorFactory
from utils.logging import get_logger
from utils.config import get_settings

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run strategy backtests')
    
    parser.add_argument(
        '--strategy', 
        type=str, 
        default='mean_reversion',
        choices=StrategyFactory.list_strategies(),
        help='Strategy to backtest'
    )
    
    parser.add_argument(
        '--symbols', 
        type=str, 
        nargs='+', 
        default=['AAPL'],
        help='Symbols to trade'
    )
    
    parser.add_argument(
        '--start-date', 
        type=str, 
        default='2023-01-01',
        help='Backtest start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date', 
        type=str, 
        default='2023-12-31',
        help='Backtest end date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--initial-capital', 
        type=float, 
        default=100000.0,
        help='Initial capital for backtest'
    )
    
    parser.add_argument(
        '--data-source', 
        type=str, 
        default='yfinance',
        choices=DataCollectorFactory.list_sources(),
        help='Data source for market data'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        help='Output file for results (CSV)'
    )
    
    return parser.parse_args()


async def run_backtest(args):
    """
    Run a single backtest.
    
    Args:
        args: Parsed command line arguments
    """
    logger.info(f"Starting backtest for {args.strategy} strategy")
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Initialize components
    data_collector = DataCollectorFactory.create_collector(args.data_source)
    strategy = StrategyFactory.create_strategy(args.strategy)
    backtest_engine = BacktestEngine(initial_capital=args.initial_capital)
    
    results = []
    
    for symbol in args.symbols:
        logger.info(f"Running backtest for {symbol}")
        
        try:
            # Get market data
            data = await data_collector.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                continue
            
            # Generate strategy signals
            signals = strategy.generate_signals(data)
            
            # Run backtest
            result = backtest_engine.run_backtest(
                data=data,
                strategy_signals=signals,
                strategy_name=f"{args.strategy}_{symbol}"
            )
            
            result['symbol'] = symbol
            results.append(result)
            
            logger.info(f"Backtest completed for {symbol}")
            logger.info(f"  Total Return: {result['total_return']:.2%}")
            logger.info(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            logger.info(f"  Max Drawdown: {result['max_drawdown']:.2%}")
            
        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}")
            continue
    
    # Create comparison report
    if results:
        comparison_df = backtest_engine.compare_strategies(results)
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        print(comparison_df.to_string(index=False))
        
        # Save results if output file specified
        if args.output:
            comparison_df.to_csv(args.output, index=False)
            logger.info(f"Results saved to {args.output}")
    
    else:
        logger.error("No successful backtests completed")


def main():
    """Main entry point."""
    import asyncio
    
    args = parse_args()
    
    try:
        asyncio.run(run_backtest(args))
    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user")
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 