import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


def test_strategy_list_includes_bundled():
    result = runner.invoke(app, ["strategy", "list"])
    assert result.exit_code == 0, result.stdout
    assert "cross_sectional_momentum" in json.loads(result.stdout)


@pytest.fixture()
def _cleanup_scaffolded():
    """Remove strategy files created in the package tree during tests."""
    created: list[Path] = []
    yield created
    for p in created:
        p.unlink(missing_ok=True)


def test_strategy_new_scaffolds_loadable_module(tmp_path, monkeypatch, _cleanup_scaffolded):
    # CWD does NOT matter — path must be package-relative, not CWD-relative (#74)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["strategy", "new", "my_strat"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    # The returned path should be absolute and point inside the installed package,
    # not relative to the test's temporary working directory.
    p = Path(payload["path"])
    _cleanup_scaffolded.append(p)
    assert p.is_absolute(), f"expected absolute path, got {p}"
    assert p.exists(), f"scaffold file not created: {p}"


def test_strategy_new_path_is_package_relative(tmp_path, monkeypatch, _cleanup_scaffolded):
    """Path must resolve to the package strategies/examples dir regardless of CWD (#74)."""
    monkeypatch.chdir(tmp_path)  # CWD is a temp dir, NOT the project root
    result = runner.invoke(app, ["strategy", "new", "cwd_test_strat"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    p = Path(payload["path"])
    _cleanup_scaffolded.append(p)
    # Must be absolute and NOT inside tmp_path
    assert p.is_absolute()
    assert not str(p).startswith(str(tmp_path)), (
        f"path {p} is inside tmp_path {tmp_path} — still CWD-relative"
    )
    # Must be inside the algua.strategies.examples package dir
    import algua.strategies  # noqa: PLC0415
    expected_dir = Path(algua.strategies.__file__).parent / "examples"
    assert str(p).startswith(str(expected_dir)), (
        f"path {p} is not under strategies/examples {expected_dir}"
    )


def test_strategy_new_rejects_unsafe_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for bad in ["../evil", "bad-name", "with space", "1abc", "class"]:
        result = runner.invoke(app, ["strategy", "new", bad])
        assert result.exit_code == 1, (bad, result.stdout)
        assert json.loads(result.stdout)["ok"] is False


def test_strategy_new_scaffolds_doc_and_family(tmp_path, monkeypatch, _cleanup_scaffolded):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    result = runner.invoke(
        app, ["strategy", "new", "kb_new_strat", "--family", "momentum", "--derived-from", "seed"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    _cleanup_scaffolded.append(Path(payload["path"]))
    assert (tmp_path / "vault" / "kb_new_strat.md").exists()
    assert (tmp_path / "vault" / "families" / "momentum.md").exists()
    assert payload["doc"].endswith("kb_new_strat.md")


def test_strategy_new_rejects_unsafe_family(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    result = runner.invoke(app, ["strategy", "new", "alpha", "--family", "../evil"])
    assert result.exit_code == 1, result.stdout
    assert json.loads(result.stdout)["ok"] is False


def test_strategy_doc_syncs_and_builds_index(tmp_path, monkeypatch, _cleanup_scaffolded):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))

    new_result = runner.invoke(app, ["strategy", "new", "kb_sync_strat"])
    assert new_result.exit_code == 0, new_result.stdout
    _cleanup_scaffolded.append(Path(json.loads(new_result.stdout)["path"]))
    assert runner.invoke(app, ["registry", "add", "kb_sync_strat"]).exit_code == 0
    assert runner.invoke(
        app, ["registry", "transition", "kb_sync_strat", "--to", "backtested",
              "--actor", "agent", "--reason", "x"]
    ).exit_code == 0

    result = runner.invoke(app, ["strategy", "doc", "--all"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["ok"] is True
    assert "[[kb_sync_strat]]" in (tmp_path / "vault" / "_index.md").read_text()
    assert "stage: backtested" in (tmp_path / "vault" / "kb_sync_strat.md").read_text()


def test_strategy_doc_missing_doc_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    result = runner.invoke(app, ["strategy", "doc", "ghost"])
    assert result.exit_code == 1, result.stdout
    assert json.loads(result.stdout)["ok"] is False


def test_strategy_doc_single_refreshes_family_roster(tmp_path, monkeypatch, _cleanup_scaffolded):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))

    new_result = runner.invoke(app, ["strategy", "new", "kb_fam_strat", "--family", "mom"])
    assert new_result.exit_code == 0, new_result.stdout
    _cleanup_scaffolded.append(Path(json.loads(new_result.stdout)["path"]))
    assert runner.invoke(app, ["registry", "add", "kb_fam_strat"]).exit_code == 0
    assert runner.invoke(
        app, ["registry", "transition", "kb_fam_strat", "--to", "backtested",
              "--actor", "agent", "--reason", "x"]
    ).exit_code == 0

    # A single-strategy sync (not --all) must still refresh that strategy's family roster.
    result = runner.invoke(app, ["strategy", "doc", "kb_fam_strat"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["families"] == ["mom"]
    assert "backtested 1" in (tmp_path / "vault" / "families" / "mom.md").read_text()
