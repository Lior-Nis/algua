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
