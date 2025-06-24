#!/usr/bin/env python3
"""
Optuna-based strategy parameter optimization script.
"""

import sys
import argparse
import optuna
import pandas as pd
from datetime import datetime
from typing import Dict, Any

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
    parser = argparse.ArgumentParser(description='Optimize strategy parameters')
    
    parser.add_argument(
        '--strategy', 
        type=str, 
        default='mean_reversion',
        help='Strategy to optimize'
    )
    
    parser.add_argument(
        '--symbol', 
        type=str, 
        default='AAPL',
        help='Symbol to optimize on'
    )
    
    parser.add_argument(
        '--n-trials', 
        type=int, 
        default=100,
        help='Number of optimization trials'
    )
    
    parser.add_argument(
        '--study-name', 
        type=str, 
        help='Optuna study name'
    )
    
    parser.add_argument(
        '--metric', 
        type=str, 
        default='sharpe_ratio',
        choices=['sharpe_ratio', 'total_return', 'max_drawdown'],
        help='Optimization metric'
    )
    
    return parser.parse_args()


class StrategyOptimizer:
    """Optuna-based strategy parameter optimizer."""
    
    def __init__(self, strategy_name: str, symbol: str, metric: str = 'sharpe_ratio'):
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.metric = metric
        self.data_collector = DataCollectorFactory.create_collector('yfinance')
        self.backtest_engine = BacktestEngine()
        self.market_data = None
        
    async def load_data(self):
        """Load market data for optimization."""
        logger.info(f"Loading data for {self.symbol}")
        
        end_date = datetime.now()
        start_date = datetime(end_date.year - 2, end_date.month, end_date.day)
        
        self.market_data = await self.data_collector.get_historical_data(
            symbol=self.symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"Loaded {len(self.market_data)} data points")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for parameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value to optimize
        """
        try:
            # Define parameter ranges based on strategy
            if self.strategy_name == 'mean_reversion':
                params = {
                    'window': trial.suggest_int('window', 10, 50),
                    'std_dev': trial.suggest_float('std_dev', 1.0, 3.0)
                }
            elif self.strategy_name == 'momentum':
                params = {
                    'fast_period': trial.suggest_int('fast_period', 5, 20),
                    'slow_period': trial.suggest_int('slow_period', 20, 60)
                }
            elif self.strategy_name == 'rsi':
                params = {
                    'rsi_period': trial.suggest_int('rsi_period', 10, 25),
                    'oversold': trial.suggest_float('oversold', 20, 35),
                    'overbought': trial.suggest_float('overbought', 65, 80)
                }
            else:
                # Default parameters
                params = {}
            
            # Create strategy with trial parameters
            strategy = StrategyFactory.create_strategy(self.strategy_name, **params)
            
            # Generate signals
            signals = strategy.generate_signals(self.market_data)
            
            # Run backtest
            result = self.backtest_engine.run_backtest(
                data=self.market_data,
                strategy_signals=signals,
                strategy_name=f"{self.strategy_name}_trial_{trial.number}"
            )
            
            # Return metric to optimize
            metric_value = result.get(self.metric, 0.0)
            
            # For max_drawdown, we want to minimize (less negative is better)
            if self.metric == 'max_drawdown':
                metric_value = -metric_value
            
            # Log trial results
            trial.set_user_attr('total_return', result.get('total_return', 0.0))
            trial.set_user_attr('sharpe_ratio', result.get('sharpe_ratio', 0.0))
            trial.set_user_attr('max_drawdown', result.get('max_drawdown', 0.0))
            trial.set_user_attr('win_rate', result.get('win_rate', 0.0))
            
            return metric_value
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float('-inf') if self.metric != 'max_drawdown' else float('inf')
    
    def optimize(self, n_trials: int = 100, study_name: str = None) -> Dict[str, Any]:
        """
        Run parameter optimization.
        
        Args:
            n_trials: Number of optimization trials
            study_name: Name of the study
            
        Returns:
            Optimization results
        """
        if study_name is None:
            study_name = f"{self.strategy_name}_{self.symbol}_{self.metric}"
        
        logger.info(f"Starting optimization: {study_name}")
        logger.info(f"Strategy: {self.strategy_name}")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Metric: {self.metric}")
        logger.info(f"Trials: {n_trials}")
        
        # Create study
        direction = 'maximize' if self.metric != 'max_drawdown' else 'minimize'
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)
        
        # Get results
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Optimization completed!")
        logger.info(f"Best {self.metric}: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Return results
        results = {
            'study_name': study_name,
            'strategy': self.strategy_name,
            'symbol': self.symbol,
            'metric': self.metric,
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': n_trials,
            'study': study  # For further analysis
        }
        
        return results


async def main():
    """Main optimization routine."""
    args = parse_args()
    
    # Initialize optimizer
    optimizer = StrategyOptimizer(
        strategy_name=args.strategy,
        symbol=args.symbol,
        metric=args.metric
    )
    
    # Load data
    await optimizer.load_data()
    
    # Run optimization
    results = optimizer.optimize(
        n_trials=args.n_trials,
        study_name=args.study_name
    )
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Strategy: {results['strategy']}")
    print(f"Symbol: {results['symbol']}")
    print(f"Metric: {results['metric']}")
    print(f"Best Value: {results['best_value']:.4f}")
    print(f"Best Parameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    
    # TODO: Save results to file or database
    # TODO: Upload to W&B if enabled


if __name__ == "__main__":
    import asyncio
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1) 