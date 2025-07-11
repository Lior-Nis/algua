"""
Simple Moving Average Crossover Strategy.

This strategy generates buy signals when a fast SMA crosses above a slow SMA,
and sell signals when the fast SMA crosses below the slow SMA.
"""

from typing import Dict, List, Optional, Any
from decimal import Decimal
import logging
from datetime import datetime

from domain.entities.strategy import Strategy, StrategyStatus
from domain.value_objects import Symbol, Money
from utils.logging import get_logger

logger = get_logger(__name__)


class SMACrossoverStrategy:
    """
    Simple Moving Average Crossover Strategy.
    
    Parameters:
    - fast_period: Period for fast moving average (default: 10)
    - slow_period: Period for slow moving average (default: 30)
    - min_volume: Minimum volume required for signals (default: 100000)
    """
    
    def __init__(self, 
                 fast_period: int = 10, 
                 slow_period: int = 30,
                 min_volume: int = 100000):
        """
        Initialize the SMA crossover strategy.
        
        Args:
            fast_period: Period for fast moving average
            slow_period: Period for slow moving average
            min_volume: Minimum volume required for signals
        """
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.min_volume = min_volume
        
        logger.info(f"SMA Crossover Strategy initialized: fast={fast_period}, slow={slow_period}")
    
    def generate_signals(self, data: List[Dict]) -> List[Dict]:
        """
        Generate buy/sell signals based on SMA crossover.
        
        Args:
            data: List of OHLCV dictionaries
            
        Returns:
            List of signal dictionaries with buy_signal and sell_signal columns
        """
        if len(data) < self.slow_period:
            logger.warning(f"Insufficient data for SMA calculation. Need {self.slow_period}, got {len(data)}")
            return [{'buy_signal': False, 'sell_signal': False} for _ in data]
        
        signals = []
        fast_sma_values = []
        slow_sma_values = []
        
        # Calculate moving averages
        for i in range(len(data)):
            # Calculate fast SMA
            if i >= self.fast_period - 1:
                fast_sum = sum(data[j]['Close'] for j in range(i - self.fast_period + 1, i + 1))
                fast_sma = fast_sum / self.fast_period
                fast_sma_values.append(fast_sma)
            else:
                fast_sma_values.append(None)
            
            # Calculate slow SMA
            if i >= self.slow_period - 1:
                slow_sum = sum(data[j]['Close'] for j in range(i - self.slow_period + 1, i + 1))
                slow_sma = slow_sum / self.slow_period
                slow_sma_values.append(slow_sma)
            else:
                slow_sma_values.append(None)
        
        # Generate signals
        for i in range(len(data)):
            buy_signal = False
            sell_signal = False
            
            # Need at least 2 periods to detect crossover
            if (i > 0 and 
                fast_sma_values[i] is not None and 
                slow_sma_values[i] is not None and
                fast_sma_values[i-1] is not None and 
                slow_sma_values[i-1] is not None):
                
                current_fast = fast_sma_values[i]
                current_slow = slow_sma_values[i]
                prev_fast = fast_sma_values[i-1]
                prev_slow = slow_sma_values[i-1]
                
                # Check volume requirement
                volume_ok = data[i]['Volume'] >= self.min_volume
                
                # Buy signal: fast SMA crosses above slow SMA
                if (prev_fast <= prev_slow and current_fast > current_slow and volume_ok):
                    buy_signal = True
                    logger.info(f"Buy signal generated at index {i}: fast_sma={current_fast:.2f}, slow_sma={current_slow:.2f}")
                
                # Sell signal: fast SMA crosses below slow SMA
                elif (prev_fast >= prev_slow and current_fast < current_slow and volume_ok):
                    sell_signal = True
                    logger.info(f"Sell signal generated at index {i}: fast_sma={current_fast:.2f}, slow_sma={current_slow:.2f}")
            
            signals.append({
                'buy_signal': buy_signal,
                'sell_signal': sell_signal,
                'fast_sma': fast_sma_values[i],
                'slow_sma': slow_sma_values[i],
                'volume': data[i]['Volume'],
                'close': data[i]['Close']
            })
        
        total_buy_signals = sum(1 for s in signals if s['buy_signal'])
        total_sell_signals = sum(1 for s in signals if s['sell_signal'])
        
        logger.info(f"Generated {total_buy_signals} buy signals and {total_sell_signals} sell signals")
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'min_volume': self.min_volume,
            'strategy_type': 'sma_crossover'
        }
    
    def set_parameters(self, **params) -> None:
        """Set strategy parameters."""
        if 'fast_period' in params:
            self.fast_period = params['fast_period']
        if 'slow_period' in params:
            self.slow_period = params['slow_period']
        if 'min_volume' in params:
            self.min_volume = params['min_volume']
        
        # Validate parameters
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        logger.info(f"Strategy parameters updated: {self.get_parameters()}")
    
    def validate_data(self, data: List[Dict]) -> bool:
        """Validate that data has required fields."""
        if not data:
            return False
        
        required_fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for field in required_fields:
            if field not in data[0]:
                logger.error(f"Missing required field: {field}")
                return False
        
        return True
    
    def get_signal_summary(self, signals: List[Dict]) -> Dict[str, Any]:
        """Get summary statistics of generated signals."""
        if not signals:
            return {}
        
        buy_signals = sum(1 for s in signals if s['buy_signal'])
        sell_signals = sum(1 for s in signals if s['sell_signal'])
        
        # Find signal periods
        buy_indices = [i for i, s in enumerate(signals) if s['buy_signal']]
        sell_indices = [i for i, s in enumerate(signals) if s['sell_signal']]
        
        return {
            'total_periods': len(signals),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'signal_frequency': (buy_signals + sell_signals) / len(signals) * 100,
            'buy_indices': buy_indices,
            'sell_indices': sell_indices,
            'first_signal_period': min(buy_indices + sell_indices) if (buy_indices or sell_indices) else None,
            'last_signal_period': max(buy_indices + sell_indices) if (buy_indices or sell_indices) else None
        }


def create_sma_strategy(**params) -> SMACrossoverStrategy:
    """Factory function to create SMA crossover strategy."""
    return SMACrossoverStrategy(**params)


# Strategy metadata for registration
STRATEGY_METADATA = {
    'name': 'SMA Crossover',
    'description': 'Simple Moving Average Crossover Strategy',
    'type': 'trend_following',
    'factory': create_sma_strategy,
    'default_params': {
        'fast_period': 10,
        'slow_period': 30,
        'min_volume': 100000
    },
    'param_ranges': {
        'fast_period': (5, 50),
        'slow_period': (20, 200),
        'min_volume': (0, 10000000)
    }
}