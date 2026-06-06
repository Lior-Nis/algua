from pathlib import Path

import pytest
from pydantic import ValidationError

from algua.config.settings import Settings, get_settings


def test_defaults():
    s = Settings()
    assert s.exchange == "XNYS"
    assert s.db_path == Path("data/algua.db")


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "x.db"))
    monkeypatch.setenv("ALGUA_EXCHANGE", "XLON")
    s = get_settings()
    assert s.exchange == "XLON"
    assert s.db_path == tmp_path / "x.db"


def test_alpaca_env_override(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "secret")
    monkeypatch.setenv("ALGUA_ALPACA_DATA_URL", "https://example.test")
    s = get_settings()
    assert s.alpaca_api_key == "key"
    assert s.alpaca_api_secret == "secret"
    assert s.alpaca_data_url == "https://example.test"


def test_mlflow_tracking_uri_default_and_override(monkeypatch):
    from algua.config.settings import get_settings
    assert get_settings().mlflow_tracking_uri == "mlruns"
    monkeypatch.setenv("ALGUA_MLFLOW_TRACKING_URI", "/tmp/x/mlruns")
    assert get_settings().mlflow_tracking_uri == "/tmp/x/mlruns"


def test_alpaca_paper_url_default():
    from algua.config.settings import get_settings

    assert get_settings().alpaca_paper_url == "https://paper-api.alpaca.markets"


def test_knowledge_dir_default_and_override(monkeypatch):
    from algua.config.settings import Settings

    assert Settings().knowledge_dir == Path("kb")
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", "/tmp/vault")
    assert Settings().knowledge_dir == Path("/tmp/vault")


def test_alpaca_paper_url_rejects_live_endpoint(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_PAPER_URL", "https://api.alpaca.markets")
    with pytest.raises(ValidationError):
        get_settings()


def test_alpaca_paper_url_rejects_non_paper_host(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_PAPER_URL", "https://example.test")
    with pytest.raises(ValidationError):
        get_settings()


def test_alpaca_paper_url_accepts_paper_host(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_PAPER_URL", "https://paper-api.alpaca.markets/v2")
    assert get_settings().alpaca_paper_url == "https://paper-api.alpaca.markets/v2"


def test_db_path_rejects_empty():
    with pytest.raises(ValidationError):
        Settings(db_path=Path(""))


def test_data_dir_rejects_empty():
    with pytest.raises(ValidationError):
        Settings(data_dir=Path(""))


def test_alpaca_live_url_validator(monkeypatch):
    import pytest

    from algua.config.settings import Settings
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_URL", "https://paper-api.alpaca.markets")
    with pytest.raises(ValueError):
        Settings()
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_URL", "https://api.alpaca.markets")
    assert Settings().alpaca_live_url == "https://api.alpaca.markets"


def test_alpaca_urls_require_https(monkeypatch):
    import pytest

    from algua.config.settings import Settings
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_URL", "http://api.alpaca.markets")
    with pytest.raises(ValueError):
        Settings()
    monkeypatch.delenv("ALGUA_ALPACA_LIVE_URL", raising=False)
    monkeypatch.setenv("ALGUA_ALPACA_PAPER_URL", "http://paper-api.alpaca.markets")
    with pytest.raises(ValueError):
        Settings()
