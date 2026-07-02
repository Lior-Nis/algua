from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from algua.contracts.net import require_https_allowlisted_host

# Alpaca's paper-trading API host. The hard live boundary is enforced at config load:
# the paper URL must point here, never at the live host (api.alpaca.markets).
_ALPACA_PAPER_HOST = "paper-api.alpaca.markets"
_ALPACA_LIVE_HOST = "api.alpaca.markets"
# Alpaca's market-data host. Its credentials are the same account-scoped broker secrets, so the
# data URL is pinned https + host at config load — defense-in-depth ahead of AlpacaBarProvider's
# own constructor guard (#394).
_ALPACA_DATA_HOST = "data.alpaca.markets"


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
    alpaca_live_api_key: str | None = None
    alpaca_live_api_secret: str | None = None
    alpaca_live_url: str = "https://api.alpaca.markets"
    mlflow_tracking_uri: str = "mlruns"

    @field_validator("db_path", "data_dir")
    @classmethod
    def _path_must_be_nonempty(cls, value: Path) -> Path:
        if not str(value).strip() or value == Path("."):
            raise ValueError("path must be a non-empty filesystem path")
        return value

    @field_validator("alpaca_data_url")
    @classmethod
    def _data_url_is_data_endpoint(cls, value: str) -> str:
        # https-only to the Alpaca data host — the data URL carries the same broker credentials,
        # so a plaintext/foreign host would leak them (#394).
        try:
            require_https_allowlisted_host(value, frozenset({_ALPACA_DATA_HOST}))
        except ValueError as exc:
            raise ValueError(
                f"alpaca_data_url must be https to the data endpoint ({_ALPACA_DATA_HOST}); "
                f"got {value!r}"
            ) from exc
        return value

    @field_validator("alpaca_paper_url")
    @classmethod
    def _paper_url_is_paper_endpoint(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.hostname != _ALPACA_PAPER_HOST:
            raise ValueError(
                f"alpaca_paper_url must be https to the paper endpoint "
                f"({_ALPACA_PAPER_HOST}); got {value!r}. "
                "The live endpoint is forbidden by the paper-only boundary."
            )
        return value

    @field_validator("alpaca_live_url")
    @classmethod
    def _live_url_is_live_endpoint(cls, value: str) -> str:
        # https-only: API keys must never travel over plaintext (codex review).
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.hostname != _ALPACA_LIVE_HOST:
            raise ValueError(
                f"alpaca_live_url must be https to the live endpoint ({_ALPACA_LIVE_HOST}); "
                f"got {value!r}"
            )
        return value


def get_settings() -> Settings:
    """Fresh Settings each call (re-reads env; keeps tests isolated)."""
    return Settings()
