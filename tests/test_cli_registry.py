import json
import pytest
from typer.testing import CliRunner
from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def _json(result):
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_add_and_list():
    _json(runner.invoke(app, ["registry", "add", "alpha"]))
    listed = _json(runner.invoke(app, ["registry", "list"]))
    assert [s["name"] for s in listed] == ["alpha"]
    assert listed[0]["stage"] == "idea"


def test_transition_legal():
    runner.invoke(app, ["registry", "add", "alpha"])
    out = _json(runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "backtested",
              "--actor", "agent", "--reason", "ran"]))
    assert out["stage"] == "backtested"


def test_transition_illegal_exits_nonzero():
    runner.invoke(app, ["registry", "add", "alpha"])
    result = runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "live", "--actor", "agent"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_full_path_to_live_with_approval():
    runner.invoke(app, ["registry", "add", "alpha"])
    for stage in ("backtested", "shortlisted", "paper"):
        runner.invoke(app, ["registry", "transition", "alpha",
                            "--to", stage, "--actor", "agent"])
    runner.invoke(app, ["registry", "approve", "alpha",
                        "--code-hash", "c1", "--config-hash", "g1", "--by", "lior"])
    out = _json(runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "live", "--actor", "human",
              "--code-hash", "c1", "--config-hash", "g1"]))
    assert out["stage"] == "live"
