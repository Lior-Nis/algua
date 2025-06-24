"""
Production environment configuration.
"""

from .base import BaseConfig


class ProductionConfig(BaseConfig):
    """Production configuration."""
    
    debug: bool = False
    api_reload: bool = False
    log_level: str = "INFO"
    
    # Production Alpaca settings
    alpaca_base_url: str = "https://api.alpaca.markets"
    
    # Conservative limits for production
    max_position_size: float = 0.02  # 2% for production
    max_daily_loss: float = 0.005    # 0.5% for production
    
    # Longer cache for production
    cache_ttl_seconds: int = 600
    
    # Production database - should be set via env var
    database_url: str = None  # Must be provided via env
    
    # Enable all monitoring in production
    enable_metrics: bool = True
    
    # Stricter risk limits
    max_drawdown: float = 0.10       # 10% max drawdown
    var_confidence: float = 0.99     # 99% VaR confidence
    
    # Production logging
    log_file: str = "/var/log/algua/algua.log"
    
    class Config(BaseConfig.Config):
        env_file = ".env.production"
        
    def validate_production_requirements(self) -> list[str]:
        """Additional validation for production."""
        issues = self.validate_required_secrets()
        
        if not self.database_url:
            issues.append("DATABASE_URL must be set in production")
            
        if self.secret_key == "dev-secret-key":
            issues.append("SECRET_KEY must be changed from default")
            
        if not self.telegram_bot_token:
            issues.append("TELEGRAM_BOT_TOKEN recommended for alerts")
            
        return issues