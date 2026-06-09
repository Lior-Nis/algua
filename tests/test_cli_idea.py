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


def _add(*args):
    base = ["research", "idea", "add", "--title", "low vol anomaly",
            "--hypothesis", "low vol names outperform risk adjusted",
            "--family", "vol", "--source-type", "paper", "--source-ref", "http://x",
            "--required-data", "ohlcv"]
    return runner.invoke(app, base + list(args))


def test_add_and_show():
    added = _json(_add())
    assert added["ok"] is True
    assert added["status"] == "open"
    shown = _json(runner.invoke(app, ["research", "idea", "show", str(added["id"])]))
    assert shown["title"] == "low vol anomaly"


def test_needs_data_parking():
    out = _json(runner.invoke(app, [
        "research", "idea", "add", "--title", "whale 13f", "--hypothesis", "follow institutions",
        "--family", "flow", "--source-type", "filing", "--required-data", "form_13f"]))
    assert out["status"] == "needs_data"


def test_unknown_capability_errors():
    result = runner.invoke(app, [
        "research", "idea", "add", "--title", "x", "--hypothesis", "y",
        "--source-type", "manual", "--required-data", "satellite_imagery"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_dedup_collision_fails_closed():
    _add()
    result = _add("--title", "the low vol anomaly",
                  "--hypothesis", "low vol stocks outperform on a risk adjusted basis")
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert len(payload["collisions"]) == 1


def test_allow_duplicate_override():
    _add()
    out = _json(_add("--title", "the low vol anomaly",
                     "--hypothesis", "low vol stocks outperform on a risk adjusted basis",
                     "--allow-duplicate", "--reason", "distinct universe"))
    assert out["duplicate_of_idea_id"] == 1
    assert out["override_reason"] == "distinct universe"


def test_allow_duplicate_requires_reason():
    _add()
    result = _add("--title", "the low vol anomaly",
                  "--hypothesis", "low vol stocks outperform on a risk adjusted basis",
                  "--allow-duplicate")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_refuted_collision_not_overridable():
    added = _json(_add())
    runner.invoke(app, ["registry", "add", "lowvol_v1", "--family", "vol"])
    _json(runner.invoke(app, [
        "research", "idea", "set-status", str(added["id"]), "--to", "authored",
        "--strategy", "lowvol_v1"]))
    # Refute the linked strategy; its authored idea now surfaces as effective REFUTED.
    _json(runner.invoke(app, [
        "registry", "set", "lowvol_v1", "--hypothesis-status", "refuted"]))
    # Even with --allow-duplicate --reason the refuted wall must hold.
    result = _add("--title", "the low vol anomaly",
                  "--hypothesis", "low vol stocks outperform on a risk adjusted basis",
                  "--allow-duplicate", "--reason", "x")
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "refuted" in payload["error"]


def test_list_is_bare_array():
    _add()
    listed = _json(runner.invoke(app, ["research", "idea", "list"]))
    assert isinstance(listed, list)
    assert listed[0]["title"] == "low vol anomaly"


def test_dedup_check_no_write():
    _add()
    out = _json(runner.invoke(app, [
        "research", "idea", "dedup-check", "--title", "the low vol anomaly",
        "--hypothesis", "low vol stocks outperform on a risk adjusted basis", "--family", "vol"]))
    assert out["is_novel"] is False
    assert len(out["collisions"]) == 1
    # nothing was written
    assert len(_json(runner.invoke(app, ["research", "idea", "list"]))) == 1


def test_set_status_authored_requires_strategy():
    added = _json(_add())
    result = runner.invoke(app, [
        "research", "idea", "set-status", str(added["id"]), "--to", "authored"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_set_status_authored_links_strategy():
    added = _json(_add())
    runner.invoke(app, ["registry", "add", "lowvol_v1", "--family", "vol"])
    out = _json(runner.invoke(app, [
        "research", "idea", "set-status", str(added["id"]), "--to", "authored",
        "--strategy", "lowvol_v1"]))
    assert out["status"] == "authored"
    assert out["authored_strategy_id"] is not None


def test_stats_counts_by_status():
    _add()
    out = _json(runner.invoke(app, ["research", "idea", "stats"]))
    assert out["counts"]["open"] == 1
    assert out["counts"]["total"] == 1
