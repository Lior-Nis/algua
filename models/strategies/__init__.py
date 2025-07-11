"""
Trading strategies package.
"""

from .sma_crossover import SMACrossoverStrategy, create_sma_strategy, STRATEGY_METADATA

__all__ = ['SMACrossoverStrategy', 'create_sma_strategy', 'STRATEGY_METADATA']