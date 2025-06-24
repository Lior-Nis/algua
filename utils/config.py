"""
Configuration management for the Algua trading platform.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    alpaca_api_key: str = Field(..., env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(..., env="ALPACA_SECRET_KEY") 
    alpaca_base_url: str = Field(
        "https://paper-api.alpaca.markets", 
        env="ALPACA_BASE_URL"
    )
    
    wandb_api_key: Optional[str] = Field(None, env="WANDB_API_KEY")
    wandb_project: str = Field("algua-trading", env="WANDB_PROJECT")
    wandb_entity: Optional[str] = Field(None, env="WANDB_ENTITY")
    
    telegram_bot_token: Optional[str] = Field(None, env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(None, env="TELEGRAM_CHAT_ID")
    
    # Trading Configuration
    max_position_size: float = Field(0.02, env="MAX_POSITION_SIZE")
    max_portfolio_risk: float = Field(0.1, env="MAX_PORTFOLIO_RISK")
    stop_loss_percent: float = Field(0.05, env="STOP_LOSS_PERCENT")
    
    market_open_hour: int = Field(14, env="MARKET_OPEN_HOUR")  # UTC
    market_close_hour: int = Field(21, env="MARKET_CLOSE_HOUR")  # UTC
    
    trading_enabled: bool = Field(False, env="TRADING_ENABLED")
    paper_trading: bool = Field(True, env="PAPER_TRADING")
    min_trade_amount: float = Field(100.0, env="MIN_TRADE_AMOUNT")
    
    # Data Storage
    data_dir: str = Field("./data", env="DATA_DIR")
    logs_dir: str = Field("./logs", env="LOGS_DIR")
    models_dir: str = Field("./models/saved", env="MODELS_DIR")
    
    gcs_bucket: Optional[str] = Field(None, env="GCS_BUCKET")
    aws_bucket: Optional[str] = Field(None, env="AWS_BUCKET")
    
    # API Settings
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(1, env="API_WORKERS")
    
    dashboard_host: str = Field("localhost", env="DASHBOARD_HOST")
    dashboard_port: int = Field(8501, env="DASHBOARD_PORT")
    
    # Logging & Monitoring
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    enable_telegram_alerts: bool = Field(False, env="ENABLE_TELEGRAM_ALERTS")
    enable_wandb_logging: bool = Field(True, env="ENABLE_WANDB_LOGGING")
    
    # Development
    debug: bool = Field(False, env="DEBUG")
    testing: bool = Field(False, env="TESTING")
    vectorbt_pro_license_key: Optional[str] = Field(
        None, env="VECTORBT_PRO_LICENSE_KEY"
    )
    
    # Data Sources
    firstrate_data_path: str = Field("./data/firstrate", env="FIRSTRATE_DATA_PATH")
    databento_data_path: str = Field("./data/databento", env="DATABENTO_DATA_PATH")
    
    default_timeframe: str = Field("1h", env="DEFAULT_TIMEFRAME")
    lookback_days: int = Field(252, env="LOOKBACK_DAYS")
    data_refresh_interval: int = Field(300, env="DATA_REFRESH_INTERVAL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get application settings (singleton pattern).
    
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_file: Path to config file (YAML or JSON)
        
    Returns:
        Configuration dictionary
        
    TODO: Implement YAML/JSON config file loading
    """
    if config_file and os.path.exists(config_file):
        # TODO: Load from YAML or JSON file
        pass
    
    # Return environment-based config for now
    settings = get_settings()
    return settings.dict()


def validate_config() -> bool:
    """
    Validate the current configuration.
    
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    settings = get_settings()
    
    # Validate API keys for live trading
    if not settings.paper_trading and settings.trading_enabled:
        if not settings.alpaca_api_key or not settings.alpaca_secret_key:
            raise ValueError("Alpaca API credentials required for live trading")
    
    # Validate risk management settings
    if settings.max_position_size <= 0 or settings.max_position_size > 1:
        raise ValueError("max_position_size must be between 0 and 1")
    
    if settings.max_portfolio_risk <= 0 or settings.max_portfolio_risk > 1:
        raise ValueError("max_portfolio_risk must be between 0 and 1")
    
    # Validate directories exist
    for directory in [settings.data_dir, settings.logs_dir, settings.models_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return True


def is_market_hours() -> bool:
    """
    Check if current time is within market hours.
    
    Returns:
        True if markets are open
        
    TODO: Implement proper market hours checking with holidays
    """
    from datetime import datetime, timezone
    
    settings = get_settings()
    now_utc = datetime.now(timezone.utc)
    current_hour = now_utc.hour
    
    # Simple check - weekdays between market open/close hours
    if now_utc.weekday() >= 5:  # Weekend
        return False
    
    return settings.market_open_hour <= current_hour < settings.market_close_hour 