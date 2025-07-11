"""
Backtesting engine using VectorBT for strategy evaluation.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import vectorbt as vbt
except ImportError:
    vbt = None

from utils.logging import get_logger
from utils.config import get_settings
from domain.entities.strategy import Strategy, StrategyPerformance
from domain.value_objects import Money

logger = get_logger(__name__)


class BacktestEngine:
    """
    VectorBT-based backtesting engine.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital for backtests
        """
        if vbt is None:
            logger.warning("VectorBT not available, using fallback implementation")
        
        if pd is None:
            logger.warning("pandas not available, using basic Python data structures")
        
        if np is None:
            logger.warning("numpy not available, using basic Python math")
        
        self.initial_capital = initial_capital
        self.settings = get_settings()
        logger.info(f"Initialized backtesting engine with ${initial_capital:,.2f}")
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_signals: pd.DataFrame,
        strategy_name: str = "Strategy",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a backtest using VectorBT or fallback implementation.
        
        Args:
            data: OHLCV price data with columns: Open, High, Low, Close, Volume, Timestamp
            strategy_signals: Buy/sell signals with columns: buy_signal, sell_signal
            strategy_name: Name of the strategy
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Validate inputs
            if data.empty:
                raise ValueError("Data cannot be empty")
            
            if strategy_signals.empty:
                raise ValueError("Strategy signals cannot be empty")
            
            required_data_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_data_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required data columns: {missing_columns}")
            
            if 'buy_signal' not in strategy_signals.columns or 'sell_signal' not in strategy_signals.columns:
                raise ValueError("Strategy signals must have 'buy_signal' and 'sell_signal' columns")
            
            # Align data and signals
            if len(data) != len(strategy_signals):
                min_len = min(len(data), len(strategy_signals))
                data = data.iloc[:min_len]
                strategy_signals = strategy_signals.iloc[:min_len]
            
            if vbt is not None:
                results = self._run_vectorbt_backtest(data, strategy_signals, strategy_name, **kwargs)
            else:
                results = self._run_fallback_backtest(data, strategy_signals, strategy_name, **kwargs)
            
            logger.info(f"Backtest completed for {strategy_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest for {strategy_name}: {str(e)}")
            raise
    
    def _run_vectorbt_backtest(
        self,
        data: pd.DataFrame,
        strategy_signals: pd.DataFrame,
        strategy_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run backtest using VectorBT."""
        # Set up VectorBT portfolio
        close_prices = data['Close'].values
        buy_signals = strategy_signals['buy_signal'].values
        sell_signals = strategy_signals['sell_signal'].values
        
        # Create portfolio
        portfolio = vbt.Portfolio.from_signals(
            close=close_prices,
            entries=buy_signals,
            exits=sell_signals,
            init_cash=self.initial_capital,
            freq='1D'  # Assume daily frequency, can be adjusted
        )
        
        # Calculate performance metrics
        results = self._calculate_vectorbt_metrics(portfolio, data, strategy_name)
        return results
    
    def _run_fallback_backtest(
        self,
        data: pd.DataFrame,
        strategy_signals: pd.DataFrame,
        strategy_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run backtest using fallback implementation."""
        logger.info(f"Running fallback backtest for {strategy_name}")
        
        # Simple backtest simulation
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        positions = 0
        trades = []
        portfolio_values = []
        
        for i in range(len(data)):
            current_price = data.iloc[i]['Close']
            buy_signal = strategy_signals.iloc[i]['buy_signal']
            sell_signal = strategy_signals.iloc[i]['sell_signal']
            
            # Execute trades
            if buy_signal and cash > 0:
                shares_to_buy = cash // current_price
                if shares_to_buy > 0:
                    positions += shares_to_buy
                    cash -= shares_to_buy * current_price
                    trades.append({
                        'type': 'BUY',
                        'price': current_price,
                        'shares': shares_to_buy,
                        'date': data.iloc[i].get('Timestamp', i)
                    })
            
            elif sell_signal and positions > 0:
                cash += positions * current_price
                trades.append({
                    'type': 'SELL',
                    'price': current_price,
                    'shares': positions,
                    'date': data.iloc[i].get('Timestamp', i)
                })
                positions = 0
            
            # Calculate portfolio value
            portfolio_value = cash + positions * current_price
            portfolio_values.append(portfolio_value)
        
        # Calculate metrics
        results = self._calculate_fallback_metrics(
            portfolio_values, trades, data, strategy_name
        )
        
        return results
    
    def _calculate_vectorbt_metrics(
        self, 
        portfolio: 'vbt.Portfolio', 
        data: pd.DataFrame, 
        strategy_name: str
    ) -> Dict[str, Any]:
        """Calculate performance metrics from VectorBT portfolio."""
        # Basic portfolio stats
        stats = portfolio.stats()
        
        # Calculate additional metrics
        returns = portfolio.returns()
        total_return = portfolio.total_return()
        
        # Sharpe ratio (assuming 252 trading days and 0% risk-free rate)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        max_drawdown = portfolio.max_drawdown()
        
        # Win rate and profit factor
        trades = portfolio.trades.records_readable
        if len(trades) > 0:
            winning_trades = trades[trades['PnL'] > 0]
            losing_trades = trades[trades['PnL'] < 0]
            
            total_trades = len(trades)
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
            
            total_wins = winning_trades['PnL'].sum() if len(winning_trades) > 0 else 0
            total_losses = abs(losing_trades['PnL'].sum()) if len(losing_trades) > 0 else 0
            
            profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
            
            avg_win = winning_trades['PnL'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['PnL'].mean() if len(losing_trades) > 0 else 0
            
            largest_win = winning_trades['PnL'].max() if len(winning_trades) > 0 else 0
            largest_loss = losing_trades['PnL'].min() if len(losing_trades) > 0 else 0
        else:
            total_trades = win_count = loss_count = 0
            win_rate = profit_factor = avg_win = avg_loss = largest_win = largest_loss = 0
        
        # Create performance object
        performance = StrategyPerformance(
            total_return=Decimal(str(total_return * 100)),  # Convert to percentage
            sharpe_ratio=Decimal(str(sharpe_ratio)),
            max_drawdown=Decimal(str(max_drawdown * 100)),  # Convert to percentage
            win_rate=Decimal(str(win_rate)),
            profit_factor=Decimal(str(profit_factor)),
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            avg_win=Decimal(str(avg_win)),
            avg_loss=Decimal(str(avg_loss)),
            largest_win=Decimal(str(largest_win)),
            largest_loss=Decimal(str(largest_loss))
        )
        
        # Compile results
        results = {
            'strategy_name': strategy_name,
            'initial_capital': self.initial_capital,
            'final_portfolio_value': float(portfolio.value().iloc[-1]),
            'total_return_pct': float(total_return * 100),
            'total_return_dollars': float(portfolio.value().iloc[-1] - self.initial_capital),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown_pct': float(max_drawdown * 100),
            'win_rate_pct': float(win_rate),
            'profit_factor': float(profit_factor),
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'largest_win': float(largest_win),
            'largest_loss': float(largest_loss),
            'expectancy': float(performance.expectancy),
            'performance_object': performance,
            'portfolio_value_series': portfolio.value(),
            'returns_series': returns,
            'trades_df': trades if len(trades) > 0 else pd.DataFrame(),
            'start_date': data.index[0] if hasattr(data, 'index') else data.iloc[0].get('Timestamp', 'N/A'),
            'end_date': data.index[-1] if hasattr(data, 'index') else data.iloc[-1].get('Timestamp', 'N/A'),
            'backtest_duration_days': len(data)
        }
        
        return results
    
    def _calculate_fallback_metrics(
        self,
        portfolio_values: List[float],
        trades: List[Dict],
        data: pd.DataFrame,
        strategy_name: str
    ) -> Dict[str, Any]:
        """Calculate performance metrics from fallback backtest."""
        portfolio_series = pd.Series(portfolio_values)
        
        # Basic metrics
        total_return = (portfolio_series.iloc[-1] / self.initial_capital) - 1
        returns = portfolio_series.pct_change().dropna()
        
        # Sharpe ratio
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade analysis
        if trades:
            # Group trades into buy-sell pairs
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            
            trade_pairs = []
            for i in range(min(len(buy_trades), len(sell_trades))):
                pnl = sell_trades[i]['price'] * sell_trades[i]['shares'] - buy_trades[i]['price'] * buy_trades[i]['shares']
                trade_pairs.append(pnl)
            
            if trade_pairs:
                winning_trades = [t for t in trade_pairs if t > 0]
                losing_trades = [t for t in trade_pairs if t < 0]
                
                total_trades = len(trade_pairs)
                win_count = len(winning_trades)
                loss_count = len(losing_trades)
                
                win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
                
                total_wins = sum(winning_trades) if winning_trades else 0
                total_losses = abs(sum(losing_trades)) if losing_trades else 0
                
                profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
                
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = np.mean(losing_trades) if losing_trades else 0
                
                largest_win = max(winning_trades) if winning_trades else 0
                largest_loss = min(losing_trades) if losing_trades else 0
            else:
                total_trades = win_count = loss_count = 0
                win_rate = profit_factor = avg_win = avg_loss = largest_win = largest_loss = 0
        else:
            total_trades = win_count = loss_count = 0
            win_rate = profit_factor = avg_win = avg_loss = largest_win = largest_loss = 0
        
        # Create performance object
        performance = StrategyPerformance(
            total_return=Decimal(str(total_return * 100)),
            sharpe_ratio=Decimal(str(sharpe_ratio)),
            max_drawdown=Decimal(str(max_drawdown * 100)),
            win_rate=Decimal(str(win_rate)),
            profit_factor=Decimal(str(profit_factor)),
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            avg_win=Decimal(str(avg_win)),
            avg_loss=Decimal(str(avg_loss)),
            largest_win=Decimal(str(largest_win)),
            largest_loss=Decimal(str(largest_loss))
        )
        
        results = {
            'strategy_name': strategy_name,
            'initial_capital': self.initial_capital,
            'final_portfolio_value': portfolio_series.iloc[-1],
            'total_return_pct': total_return * 100,
            'total_return_dollars': portfolio_series.iloc[-1] - self.initial_capital,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'expectancy': float(performance.expectancy),
            'performance_object': performance,
            'portfolio_value_series': portfolio_series,
            'returns_series': returns,
            'trades_df': pd.DataFrame(trades),
            'start_date': data.iloc[0].get('Timestamp', 'N/A'),
            'end_date': data.iloc[-1].get('Timestamp', 'N/A'),
            'backtest_duration_days': len(data)
        }
        
        return results
    
    def run_optimization(
        self,
        data: pd.DataFrame,
        strategy_func: callable,
        param_grid: Dict[str, List[Any]],
        strategy_name: str = "Strategy",
        metric: str = "sharpe_ratio"
    ) -> Dict[str, Any]:
        """
        Run parameter optimization for a strategy.
        
        Args:
            data: OHLCV price data
            strategy_func: Function that generates signals given parameters
            param_grid: Dictionary of parameter names and their possible values
            strategy_name: Name of the strategy
            metric: Metric to optimize for
            
        Returns:
            Dictionary with optimization results
        """
        try:
            best_params = None
            best_score = float('-inf')
            all_results = []
            
            # Generate all parameter combinations
            import itertools
            
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            for param_combo in itertools.product(*param_values):
                params = dict(zip(param_names, param_combo))
                
                try:
                    # Generate signals with current parameters
                    signals = strategy_func(data, **params)
                    
                    # Run backtest
                    results = self.run_backtest(data, signals, f"{strategy_name}_{params}")
                    
                    # Get score for this combination
                    score = results.get(metric, float('-inf'))
                    
                    # Store results
                    result_entry = {
                        'parameters': params,
                        'score': score,
                        'results': results
                    }
                    all_results.append(result_entry)
                    
                    # Update best if this is better
                    if score > best_score:
                        best_score = score
                        best_params = params
                    
                except Exception as e:
                    logger.warning(f"Error testing parameters {params}: {str(e)}")
                    continue
            
            if best_params is None:
                raise ValueError("No valid parameter combinations found")
            
            # Run final backtest with best parameters
            best_signals = strategy_func(data, **best_params)
            best_results = self.run_backtest(data, best_signals, f"{strategy_name}_best")
            
            optimization_results = {
                'best_parameters': best_params,
                'best_score': best_score,
                'best_results': best_results,
                'all_results': all_results,
                'total_combinations_tested': len(all_results),
                'optimization_metric': metric
            }
            
            logger.info(f"Optimization completed for {strategy_name}. Best {metric}: {best_score:.4f}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error running optimization for {strategy_name}: {str(e)}")
            raise