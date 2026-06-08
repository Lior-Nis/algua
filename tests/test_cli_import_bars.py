import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path / "data"))


def _firstrate_dirs(tmp_path):
    raw = tmp_path / "raw"
    adj = tmp_path / "adj"
    raw.mkdir()
    adj.mkdir()
    for sym, rprice, aprice in [("AAPL", 100, 50), ("MSFT", 200, 180)]:
        (raw / f"{sym}_full_1day_UNADJUSTED.txt").write_text(
            f"2024-07-01,{rprice},{rprice},{rprice},{rprice},10\n", encoding="utf-8")
        (adj / f"{sym}_full_1day_adjsplitdiv.txt").write_text(
            f"2024-07-01,{aprice},{aprice},{aprice},{aprice},10\n", encoding="utf-8")
    return raw, adj


def test_import_bars_happy_path(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    result = runner.invoke(app, [
        "data", "import-bars", "--vendor", "firstrate",
        "--raw-dir", str(raw), "--adjusted-dir", str(adj),
        "--timeframe", "1d", "--as-of", "2024-07-02T00:00:00+00:00",
        "--adjustment", "split_div",
    ])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    snap = payload["snapshot"]
    assert snap["provider"] == "firstrate"
    assert snap["adjustment"] == "split_div"
    assert snap["symbols"] == ["AAPL", "MSFT"]
    assert snap["row_count"] == 2
    assert snap["source_metadata"]["vendor"] == "firstratedata"
    assert snap["source_metadata"]["raw_dir"] == "raw"
    assert snap["source_metadata"]["adjusted_dir"] == "adj"


def test_import_bars_unknown_vendor_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    result = runner.invoke(app, [
        "data", "import-bars", "--vendor", "nope",
        "--raw-dir", str(raw), "--adjusted-dir", str(adj),
        "--as-of", "2024-07-02T00:00:00+00:00",
    ])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_import_bars_symbols_filter(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    result = runner.invoke(app, [
        "data", "import-bars", "--vendor", "firstrate",
        "--raw-dir", str(raw), "--adjusted-dir", str(adj),
        "--as-of", "2024-07-02T00:00:00+00:00", "--symbols", "AAPL",
    ])
    assert result.exit_code == 0, result.stdout
    snap = json.loads(result.stdout)["snapshot"]
    assert snap["symbols"] == ["AAPL"]
    assert snap["row_count"] == 1


def test_import_bars_requested_bounds_mismatch_errors(tmp_path):
    raw, adj = _firstrate_dirs(tmp_path)
    result = runner.invoke(app, [
        "data", "import-bars", "--vendor", "firstrate",
        "--raw-dir", str(raw), "--adjusted-dir", str(adj),
        "--as-of", "2024-07-02T00:00:00+00:00",
        "--start", "2020-01-01", "--end", "2024-07-01",
    ])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
