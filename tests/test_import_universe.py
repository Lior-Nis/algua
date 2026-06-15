import json
from datetime import date

from typer.testing import CliRunner

from algua.cli.main import app
from algua.data.store import DataStore


def _csv(path, text):
    path.write_text(text)
    return path


def test_import_universe_builds_timeline(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    csv = _csv(tmp_path / "c.csv",
               "symbol,add_date,drop_date\nAAPL,1998-01-02,\nENRN,1998-01-02,2001-11-28\n")
    res = CliRunner().invoke(app, ["data", "import-universe", "SP", "--file", str(csv)])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout)
    assert payload["ok"] is True
    assert payload["snapshots_written"] == 2

    timeline = DataStore(tmp_path).read_universe("SP")
    eff = {s.effective_date: s.symbols for s in timeline}
    assert eff[date(1998, 1, 2)] == frozenset({"AAPL", "ENRN"})
    assert eff[date(2001, 11, 28)] == frozenset({"AAPL"})


def test_import_universe_same_name_correction_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    c1 = _csv(tmp_path / "a.csv", "symbol,add_date,drop_date\nAAPL,2000-01-01,\n")
    assert CliRunner().invoke(app, ["data", "import-universe", "U", "--file", str(c1)]).exit_code == 0
    c2 = _csv(tmp_path / "b.csv", "symbol,add_date,drop_date\nMSFT,2000-01-01,\n")
    res = CliRunner().invoke(app, ["data", "import-universe", "U", "--file", str(c2)])
    assert res.exit_code != 0
    assert "immutab" in res.output.lower() or "different membership" in res.output.lower()


def test_import_universe_empty_membership_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    # ENRN added then dropped with no other members -> empty membership at the drop date.
    csv = _csv(tmp_path / "e.csv", "symbol,add_date,drop_date\nENRN,1998-01-02,2001-11-28\n")
    res = CliRunner().invoke(app, ["data", "import-universe", "X", "--file", str(csv)])
    assert res.exit_code != 0
    assert "empty" in res.output.lower()
