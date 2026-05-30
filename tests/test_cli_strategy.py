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
