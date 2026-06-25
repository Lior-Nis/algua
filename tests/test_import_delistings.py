import json

from typer.testing import CliRunner

from algua.cli.main import app
from algua.data.store import DataStore


def test_import_delistings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    csv = tmp_path / "d.csv"
    csv.write_text("symbol,delisting_date,delisting_value\nENRN,2001-11-28,0.25\n")
    res = CliRunner().invoke(
        app, ["data", "import-delistings", "--file", str(csv), "--source", "vendor"]
    )
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["ok"] is True
    recs = DataStore(tmp_path).read_delistings()
    assert recs["ENRN"][0].terminal_price == 0.25


def test_import_delistings_rejects_nonpositive(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    csv = tmp_path / "d.csv"
    csv.write_text("symbol,delisting_date,delisting_value\nX,2001-01-01,0\n")
    res = CliRunner().invoke(app, ["data", "import-delistings", "--file", str(csv)])
    assert res.exit_code != 0
