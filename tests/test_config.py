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
