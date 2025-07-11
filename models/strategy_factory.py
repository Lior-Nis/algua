"""
Strategy factory for creating and managing trading strategies.
"""

from typing import Dict, List, Optional, Any, Type
import importlib
import logging

from utils.logging import get_logger

logger = get_logger(__name__)


class StrategyFactory:
    """Factory for creating and managing trading strategies."""
    
    _strategies: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register_strategy(cls, name: str, strategy_metadata: Dict[str, Any]):
        """
        Register a strategy with the factory.
        
        Args:
            name: Strategy name
            strategy_metadata: Dictionary containing strategy information
        """
        required_keys = ['name', 'description', 'type', 'factory', 'default_params']
        
        for key in required_keys:
            if key not in strategy_metadata:
                raise ValueError(f"Strategy metadata missing required key: {key}")
        
        cls._strategies[name] = strategy_metadata
        logger.info(f"Registered strategy: {name}")
    
    @classmethod
    def create_strategy(cls, name: str, **params):
        """
        Create a strategy instance.
        
        Args:
            name: Strategy name
            **params: Strategy parameters
            
        Returns:
            Strategy instance
        """
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(cls._strategies.keys())}")
        
        strategy_info = cls._strategies[name]
        factory_func = strategy_info['factory']
        
        # Merge default params with provided params
        final_params = strategy_info['default_params'].copy()
        final_params.update(params)
        
        # Validate parameters
        if 'param_ranges' in strategy_info:
            cls._validate_parameters(final_params, strategy_info['param_ranges'])
        
        try:
            strategy = factory_func(**final_params)
            logger.info(f"Created strategy: {name} with params: {final_params}")
            return strategy
        except Exception as e:
            logger.error(f"Error creating strategy {name}: {str(e)}")
            raise
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all available strategies."""
        return list(cls._strategies.keys())
    
    @classmethod
    def get_strategy_info(cls, name: str) -> Dict[str, Any]:
        """Get information about a strategy."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        
        return cls._strategies[name].copy()
    
    @classmethod
    def get_all_strategies_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all strategies."""
        return cls._strategies.copy()
    
    @classmethod
    def _validate_parameters(cls, params: Dict[str, Any], param_ranges: Dict[str, tuple]):
        """Validate strategy parameters against allowed ranges."""
        for param_name, value in params.items():
            if param_name in param_ranges:
                min_val, max_val = param_ranges[param_name]
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"Parameter {param_name}={value} out of range [{min_val}, {max_val}]"
                    )
    
    @classmethod
    def optimize_strategy(cls, name: str, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate parameter combinations for strategy optimization.
        
        Args:
            name: Strategy name
            param_grid: Dictionary of parameter names and their possible values
            
        Returns:
            List of parameter combinations
        """
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}")
        
        import itertools
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = []
        for param_combo in itertools.product(*param_values):
            params = dict(zip(param_names, param_combo))
            
            # Validate against ranges if available
            strategy_info = cls._strategies[name]
            if 'param_ranges' in strategy_info:
                try:
                    cls._validate_parameters(params, strategy_info['param_ranges'])
                    combinations.append(params)
                except ValueError:
                    # Skip invalid parameter combinations
                    continue
            else:
                combinations.append(params)
        
        logger.info(f"Generated {len(combinations)} parameter combinations for {name}")
        return combinations


def auto_register_strategies():
    """Automatically register all strategies from the strategies package."""
    try:
        # Import SMA crossover strategy
        from models.strategies.sma_crossover import STRATEGY_METADATA as sma_metadata
        StrategyFactory.register_strategy('sma_crossover', sma_metadata)
        
        logger.info("Auto-registered all strategies")
        
    except Exception as e:
        logger.error(f"Error auto-registering strategies: {str(e)}")


# Auto-register strategies when module is imported
auto_register_strategies()