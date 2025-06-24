"""
Testing environment configuration.
"""

from .base import BaseConfig


class TestingConfig(BaseConfig):
    """Testing configuration."""
    
    debug: bool = True
    log_level: str = "DEBUG"
    
    # Test database
    database_url: str = "sqlite:///:memory:"
    
    # Disable external services
    enable_metrics: bool = False
    
    # Fast cache expiry for tests
    cache_ttl_seconds: int = 1
    
    # Mock secrets for testing
    secret_key: str = "test-secret-key"
    alpaca_api_key: str = "test-api-key"
    alpaca_secret_key: str = "test-secret-key"
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    
    # Minimal limits for testing
    max_position_size: float = 0.01
    max_daily_loss: float = 0.001
    
    # Short retention for tests
    data_retention_days: int = 1
    
    class Config(BaseConfig.Config):
        env_file = ".env.testing"