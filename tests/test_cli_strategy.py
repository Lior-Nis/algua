import json

from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


def test_strategy_list_includes_bundled():
    result = runner.invoke(app, ["strategy", "list"])
    assert result.exit_code == 0, result.stdout
    assert "cross_sectional_momentum" in json.loads(result.stdout)


def test_strategy_new_scaffolds_loadable_module(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["strategy", "new", "my_strat"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert (tmp_path / payload["path"]).exists()


def test_strategy_new_rejects_unsafe_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for bad in ["../evil", "bad-name", "with space", "1abc", "class"]:
        result = runner.invoke(app, ["strategy", "new", bad])
        assert result.exit_code == 1, (bad, result.stdout)
        assert json.loads(result.stdout)["ok"] is False


def test_strategy_new_scaffolds_doc_and_family(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    result = runner.invoke(
        app, ["strategy", "new", "alpha", "--family", "momentum", "--derived-from", "seed"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert (tmp_path / "vault" / "alpha.md").exists()
    assert (tmp_path / "vault" / "families" / "momentum.md").exists()
    assert payload["doc"].endswith("alpha.md")


def test_strategy_new_rejects_unsafe_family(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    result = runner.invoke(app, ["strategy", "new", "alpha", "--family", "../evil"])
    assert result.exit_code == 1, result.stdout
    assert json.loads(result.stdout)["ok"] is False


def test_strategy_doc_syncs_and_builds_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))

    assert runner.invoke(app, ["strategy", "new", "alpha"]).exit_code == 0
    assert runner.invoke(app, ["registry", "add", "alpha"]).exit_code == 0
    assert runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "backtested",
              "--actor", "agent", "--reason", "x"]
    ).exit_code == 0

    result = runner.invoke(app, ["strategy", "doc", "--all"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["ok"] is True
    assert "[[alpha]]" in (tmp_path / "vault" / "_index.md").read_text()
    assert "stage: backtested" in (tmp_path / "vault" / "alpha.md").read_text()


def test_strategy_doc_missing_doc_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    result = runner.invoke(app, ["strategy", "doc", "ghost"])
    assert result.exit_code == 1, result.stdout
    assert json.loads(result.stdout)["ok"] is False


def test_strategy_doc_single_refreshes_family_roster(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))

    assert runner.invoke(app, ["strategy", "new", "alpha", "--family", "mom"]).exit_code == 0
    assert runner.invoke(app, ["registry", "add", "alpha"]).exit_code == 0
    assert runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "backtested",
              "--actor", "agent", "--reason", "x"]
    ).exit_code == 0

    # A single-strategy sync (not --all) must still refresh that strategy's family roster.
    result = runner.invoke(app, ["strategy", "doc", "alpha"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["families"] == ["mom"]
    assert "backtested 1" in (tmp_path / "vault" / "families" / "mom.md").read_text()
