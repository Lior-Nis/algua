"""
Development environment configuration.
"""

from .base import BaseConfig


class DevelopmentConfig(BaseConfig):
    """Development configuration."""
    
    debug: bool = True
    api_reload: bool = True
    log_level: str = "DEBUG"
    
    # Use paper trading by default
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    
    # Relaxed limits for development
    max_position_size: float = 0.05  # 5% for dev
    max_daily_loss: float = 0.01     # 1% for dev
    
    # Short cache for development
    cache_ttl_seconds: int = 60
    
    # Development database
    database_url: str = "sqlite:///./algua_dev.db"
    
    # Disable some features in dev
    enable_metrics: bool = False
    
    class Config(BaseConfig.Config):
        env_file = ".env.development"