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


def test_import_bars_intraday_roundtrip(tmp_path):
    raw = tmp_path / "raw"
    adj = tmp_path / "adj"
    raw.mkdir()
    adj.mkdir()
    (raw / "AAPL_full_1min_UNADJUSTED.txt").write_text(
        "2024-07-01 09:30:00,100,110,95,105,10\n2024-07-01 09:31:00,105,120,100,115,20\n",
        encoding="utf-8")
    (adj / "AAPL_full_1min_adjsplitdiv.txt").write_text(
        "2024-07-01 09:30:00,50,55,47,52,10\n2024-07-01 09:31:00,52,60,50,57,20\n",
        encoding="utf-8")
    result = runner.invoke(app, [
        "data", "import-bars", "--vendor", "firstrate",
        "--raw-dir", str(raw), "--adjusted-dir", str(adj),
        "--timeframe", "1m", "--as-of", "2024-07-02T00:00:00+00:00",
        "--adjustment", "split_div",
    ])
    assert result.exit_code == 0, result.stdout
    snap = json.loads(result.stdout)["snapshot"]
    assert snap["timeframe"] == "1m"
    assert snap["row_count"] == 2

    # Read it back through the serving seam: time-of-day preserved, half-open [start, end).
    import os
    from datetime import datetime
    from pathlib import Path

    from algua.data.serve import StoreBackedProvider
    from algua.data.store import DataStore

    store = DataStore(Path(os.environ["ALGUA_DATA_DIR"]))
    provider = StoreBackedProvider(store, snap["snapshot_id"])
    bars = provider.get_bars(
        ["AAPL"], datetime(2024, 7, 1), datetime(2024, 7, 1, 13, 31), "1m"
    )
    # 09:30 ET -> 13:30 UTC is included; 09:31 ET -> 13:31 UTC is excluded by the half-open end.
    assert [str(ts) for ts in bars.index] == ["2024-07-01 13:30:00+00:00"]
    assert bars["close"].iloc[0] == 105.0
    assert bars["adj_close"].iloc[0] == 52.0


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
