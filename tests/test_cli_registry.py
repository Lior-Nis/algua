import json
import sqlite3

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
    # The live gate pins the recomputed hash of the loaded module, so this exercises a real,
    # loadable strategy and supplies no caller-controlled hashes.
    strategy = "cross_sectional_momentum"
    runner.invoke(app, ["registry", "add", strategy])
    for stage in ("backtested", "shortlisted", "paper"):
        runner.invoke(app, ["registry", "transition", strategy,
                            "--to", stage, "--actor", "agent"])
    runner.invoke(app, ["registry", "approve", strategy, "--by", "lior"])
    out = _json(runner.invoke(
        app, ["registry", "transition", strategy, "--to", "live", "--actor", "human"]))
    assert out["stage"] == "live"


def test_unknown_strategy_emits_json_error():
    result = runner.invoke(app, ["registry", "show", "ghost"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "ghost" in payload["error"]


def test_invalid_stage_value_emits_json_error():
    runner.invoke(app, ["registry", "add", "alpha"])
    result = runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "bogus", "--actor", "agent"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False


def test_duplicate_add_emits_json_error():
    runner.invoke(app, ["registry", "add", "alpha"])
    result = runner.invoke(app, ["registry", "add", "alpha"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False


def test_registry_command_closes_connection(monkeypatch, tmp_path):
    from algua.cli import registry_cmd

    closed = []

    class TrackingConnection(sqlite3.Connection):
        def close(self) -> None:
            closed.append(True)
            super().close()

    def connect_tracking(_db_path):
        conn = sqlite3.connect(tmp_path / "tracked.db", factory=TrackingConnection)
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(registry_cmd, "connect", connect_tracking)

    result = runner.invoke(app, ["registry", "add", "alpha"])

    assert result.exit_code == 0, result.stdout
    assert closed == [True]
