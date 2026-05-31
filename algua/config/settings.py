from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALGUA_", env_file=".env", extra="ignore")

    db_path: Path = Path("data/algua.db")
    data_dir: Path = Path("data")
    exchange: str = "XNYS"
    timezone: str = "America/New_York"
    alpaca_api_key: str | None = None
    alpaca_api_secret: str | None = None
    alpaca_data_url: str = "https://data.alpaca.markets/v2"
    mlflow_tracking_uri: str = "mlruns"


def get_settings() -> Settings:
    """Fresh Settings each call (re-reads env; keeps tests isolated)."""
    return Settings()
