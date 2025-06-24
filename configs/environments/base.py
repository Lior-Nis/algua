"""
Base configuration settings.
"""

from pydantic import Field
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from typing import Optional, List
import os


class BaseConfig(BaseSettings):
    """Base configuration for all environments."""
    
    # Application
    app_name: str = "Algua Trading Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    
    # Database
    database_url: Optional[str] = None
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Trading
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.02    # 2% daily loss limit
    default_currency: str = "USD"
    
    # Alpaca
    alpaca_api_key: Optional[str] = Field(None, env="ALPACA_API_KEY")
    alpaca_secret_key: Optional[str] = Field(None, env="ALPACA_SECRET_KEY")
    alpaca_base_url: Optional[str] = Field(None, env="ALPACA_BASE_URL")
    
    # Data
    data_retention_days: int = 365
    cache_ttl_seconds: int = 300
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Wandb
    wandb_project: Optional[str] = Field(None, env="WANDB_PROJECT")
    wandb_entity: Optional[str] = Field(None, env="WANDB_ENTITY")
    
    # Notifications
    telegram_bot_token: Optional[str] = Field(None, env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(None, env="TELEGRAM_CHAT_ID")
    
    # Risk Management
    max_drawdown: float = 0.15      # 15% max drawdown
    risk_free_rate: float = 0.03    # 3% risk-free rate
    var_confidence: float = 0.95    # 95% VaR confidence
    
    # Strategy
    default_strategy: str = "mean_reversion"
    backtest_start_date: str = "2023-01-01"
    strategy_rebalance_frequency: str = "daily"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def get_database_url(self) -> str:
        """Get database URL with fallback."""
        if self.database_url:
            return self.database_url
        return "sqlite:///./algua.db"
    
    def get_alpaca_config(self) -> dict:
        """Get Alpaca configuration."""
        return {
            "api_key": self.alpaca_api_key,
            "secret_key": self.alpaca_secret_key,
            "base_url": self.alpaca_base_url or "https://paper-api.alpaca.markets"
        }
    
    def validate_required_secrets(self) -> List[str]:
        """Validate required secrets are present."""
        missing = []
        
        if not self.secret_key:
            missing.append("SECRET_KEY")
        
        if not self.alpaca_api_key:
            missing.append("ALPACA_API_KEY")
            
        if not self.alpaca_secret_key:
            missing.append("ALPACA_SECRET_KEY")
        
        return missing