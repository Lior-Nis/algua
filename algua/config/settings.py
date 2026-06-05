from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Alpaca's paper-trading API host. The hard live boundary is enforced at config load:
# the paper URL must point here, never at the live host (api.alpaca.markets).
_ALPACA_PAPER_HOST = "paper-api.alpaca.markets"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALGUA_", env_file=".env", extra="ignore")

    db_path: Path = Path("data/algua.db")
    data_dir: Path = Path("data")
    # Root of the Obsidian-mountable knowledge base (vault). Strategy docs live under
    # `<knowledge_dir>/strategies/`; other domains (research, news, …) are added as they arrive.
    knowledge_dir: Path = Path("kb")
    exchange: str = "XNYS"
    alpaca_api_key: str | None = None
    alpaca_api_secret: str | None = None
    alpaca_data_url: str = "https://data.alpaca.markets/v2"
    alpaca_paper_url: str = "https://paper-api.alpaca.markets"
    mlflow_tracking_uri: str = "mlruns"

    @field_validator("db_path", "data_dir")
    @classmethod
    def _path_must_be_nonempty(cls, value: Path) -> Path:
        if not str(value).strip() or value == Path("."):
            raise ValueError("path must be a non-empty filesystem path")
        return value

    @field_validator("alpaca_paper_url")
    @classmethod
    def _paper_url_is_paper_endpoint(cls, value: str) -> str:
        host = urlparse(value).hostname
        if host != _ALPACA_PAPER_HOST:
            raise ValueError(
                f"alpaca_paper_url must point at the paper endpoint "
                f"({_ALPACA_PAPER_HOST}); got host {host!r}. "
                "The live endpoint is forbidden by the paper-only boundary."
            )
        return value


def get_settings() -> Settings:
    """Fresh Settings each call (re-reads env; keeps tests isolated)."""
    return Settings()
