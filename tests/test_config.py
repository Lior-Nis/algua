from pathlib import Path

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
