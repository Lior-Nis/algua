import json
from pathlib import Path

import algua.strategies  # noqa: PLC0415
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


def test_strategy_new_requires_family(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["strategy", "new", "needs_family"])
    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "family is required" in payload["error"].lower()


def test_strategy_new_hyphen_family_maps_to_underscore_dir(tmp_path, monkeypatch, _cleanup_scaffolded):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["strategy", "new", "hyx", "--family", "mean-reversion"])
    assert result.exit_code == 0, result.stdout
    p = Path(json.loads(result.stdout)["path"])
    _cleanup_scaffolded.append(p)
    assert p.parent.name == "mean_reversion"  # dir uses underscores


def test_strategy_new_scaffolds_loadable_module(tmp_path, monkeypatch, _cleanup_scaffolded):
    # CWD does NOT matter — path must be package-relative, not CWD-relative (#74)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["strategy", "new", "my_strat", "--family", "momentum"])
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
    """Path must resolve to the package strategies/momentum dir regardless of CWD (#74)."""
    monkeypatch.chdir(tmp_path)  # CWD is a temp dir, NOT the project root
    result = runner.invoke(app, ["strategy", "new", "cwd_test_strat", "--family", "momentum"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    p = Path(payload["path"])
    _cleanup_scaffolded.append(p)
    # Must be absolute and NOT inside tmp_path
    assert p.is_absolute()
    assert not str(p).startswith(str(tmp_path)), (
        f"path {p} is inside tmp_path {tmp_path} — still CWD-relative"
    )
    # Must be inside the algua.strategies.momentum package dir
    expected_dir = Path(algua.strategies.__file__).parent / "momentum"
    assert str(p).startswith(str(expected_dir)), (
        f"path {p} is not under strategies/momentum {expected_dir}"
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
    # strategy new now validates derived_from against the registry; register the parent first.
    runner.invoke(app, ["registry", "add", "seed"])
    result = runner.invoke(
        app, ["strategy", "new", "kb_new_strat", "--family", "momentum", "--derived-from", "seed"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    _cleanup_scaffolded.append(Path(payload["path"]))
    assert (tmp_path / "vault" / "strategies" / "kb_new_strat.md").exists()
    assert (tmp_path / "vault" / "strategies" / "families" / "momentum.md").exists()
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

    new_result = runner.invoke(app, ["strategy", "new", "kb_sync_strat", "--family", "momentum"])
    assert new_result.exit_code == 0, new_result.stdout
    _cleanup_scaffolded.append(Path(json.loads(new_result.stdout)["path"]))
    # strategy new now registers; no separate registry add needed
    assert runner.invoke(
        app, ["registry", "transition", "kb_sync_strat", "--to", "backtested",
              "--actor", "agent", "--reason", "x"]
    ).exit_code == 0

    result = runner.invoke(app, ["strategy", "doc", "--all"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["ok"] is True
    assert "[[kb_sync_strat]]" in (tmp_path / "vault" / "strategies" / "_index.md").read_text()
    assert "stage: backtested" in (
        tmp_path / "vault" / "strategies" / "kb_sync_strat.md"
    ).read_text()


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
    assert runner.invoke(
        app, ["registry", "transition", "kb_fam_strat", "--to", "backtested",
              "--actor", "agent", "--reason", "x"]
    ).exit_code == 0

    # A single-strategy sync (not --all) must still refresh that strategy's family roster.
    result = runner.invoke(app, ["strategy", "doc", "kb_fam_strat"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["families"] == ["mom"]
    assert "backtested 1" in (
        tmp_path / "vault" / "strategies" / "families" / "mom.md"
    ).read_text()


# ---------------------------------------------------------------------------
# Task 14: strategy new registers with preflight + rollback
# ---------------------------------------------------------------------------


def test_strategy_new_registers(tmp_path, monkeypatch, _cleanup_scaffolded):
    """strategy new must insert a registry row and expose the metadata via registry show."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    out = json.loads(runner.invoke(app, [
        "strategy", "new", "newstrat", "--family", "mean-reversion",
        "--hypothesis-status", "untested",
    ]).stdout)
    assert out["ok"] is True, out
    _cleanup_scaffolded.append(Path(out["path"]))
    rec = json.loads(runner.invoke(app, ["registry", "show", "newstrat"]).stdout)
    assert rec["family"] == "mean-reversion"
    assert rec["stage"] == "idea"


def test_strategy_new_preflight_rejects_existing_registration(tmp_path, monkeypatch):
    """strategy new must fail if a strategy with the same name is already registered."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    # Register the name first via registry add.
    runner.invoke(app, ["registry", "add", "dup"])
    result = runner.invoke(app, ["strategy", "new", "dup"])
    out = json.loads(result.stdout)
    assert out["ok"] is False
    assert result.exit_code == 1


def test_strategy_new_rollback_on_scaffold_failure(tmp_path, monkeypatch):
    """If the kb scaffold fails after registry insert, the registry row must be rolled back."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))

    import algua.cli.strategy_cmd as sc
    import algua.strategies  # noqa: PLC0415

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(sc, "scaffold_strategy_doc", boom)
    result = runner.invoke(app, ["strategy", "new", "rollbackme", "--family", "momentum"])
    assert json.loads(result.stdout)["ok"] is False
    # The registry row must have been removed by the rollback.
    listed = json.loads(runner.invoke(app, ["registry", "list"]).stdout)
    assert "rollbackme" not in [r["name"] for r in listed]
    # The module .py file must also have been cleaned up by the rollback.
    module_path = Path(algua.strategies.__file__).parent / "momentum" / "rollbackme.py"
    assert not module_path.exists(), f"rollback left behind module file: {module_path}"


# ---------------------------------------------------------------------------
# Task 15: guard test — new doc frontmatter matches registry defaults
# ---------------------------------------------------------------------------


def test_new_doc_frontmatter_matches_registry(tmp_path, monkeypatch, _cleanup_scaffolded):
    """A freshly-created strategy's kb-doc frontmatter must match its registry record."""
    from algua.knowledge.frontmatter import parse_doc

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))

    result = runner.invoke(app, ["strategy", "new", "s1", "--family", "mean-reversion"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    _cleanup_scaffolded.append(Path(payload["path"]))

    rec = json.loads(runner.invoke(app, ["registry", "show", "s1"]).stdout)
    assert rec["family"] == "mean-reversion"
    assert rec["hypothesis_status"] == "untested"

    # Also verify the frontmatter in the doc itself matches.
    doc_path = Path(payload["doc"])
    fm, _ = parse_doc(doc_path.read_text())
    assert fm["hypothesis_status"] == "untested"
    assert fm.get("family") == "[[mean-reversion]]"


def test_strategy_new_rollback_removes_created_family_hub(tmp_path, monkeypatch):
    """If scaffold fails AFTER the family hub is written, rollback must remove the hub file."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))

    import algua.cli.strategy_cmd as sc

    # Force failure after the family hub has been written (sync_strategy_doc is called last).
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(sc, "sync_strategy_doc", boom)
    result = runner.invoke(
        app, ["strategy", "new", "rollback_fam", "--family", "new-family"]
    )
    assert json.loads(result.stdout)["ok"] is False

    # The family hub must have been removed by rollback (it was created this call).
    from algua.config.settings import get_settings
    from algua.knowledge.sync import family_doc_path
    fam_path = family_doc_path(get_settings(), "new-family")
    assert not fam_path.exists(), f"rollback left behind family hub: {fam_path}"


def test_strategy_new_rollback_keeps_preexisting_family_hub(tmp_path, monkeypatch,
                                                             _cleanup_scaffolded):
    """If the family hub already existed before this call, rollback must NOT remove it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))

    import algua.cli.strategy_cmd as sc
    from algua.config.settings import get_settings
    from algua.knowledge.sync import family_doc_path

    # Pre-create a strategy that establishes the family hub.
    result0 = runner.invoke(
        app, ["strategy", "new", "prior_strat", "--family", "existing-family"]
    )
    assert result0.exit_code == 0, result0.stdout
    _cleanup_scaffolded.append(Path(json.loads(result0.stdout)["path"]))

    fam_path = family_doc_path(get_settings(), "existing-family")
    assert fam_path.exists(), "pre-existing family hub must exist before test"

    # Force failure after family hub check (sync_strategy_doc is called last).
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(sc, "sync_strategy_doc", boom)
    result = runner.invoke(
        app, ["strategy", "new", "second_strat", "--family", "existing-family"]
    )
    assert json.loads(result.stdout)["ok"] is False

    # The pre-existing family hub must NOT have been removed.
    assert fam_path.exists(), "rollback must not remove a pre-existing family hub"


def test_strategy_new_metadata_roundtrips_to_doc(tmp_path, monkeypatch, _cleanup_scaffolded):
    """strategy new --author/--tag metadata must appear in the kb doc at creation time (Fix 3)."""
    from algua.knowledge.frontmatter import parse_doc

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))

    result = runner.invoke(
        app,
        ["strategy", "new", "mdoc", "--family", "momentum", "--author", "human", "--tag", "carry",
         "--hypothesis-status", "untested"],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    _cleanup_scaffolded.append(Path(payload["path"]))

    doc_path = Path(payload["doc"])
    fm, _ = parse_doc(doc_path.read_text())
    assert fm["author"] == "human", f"expected author='human', got {fm.get('author')!r}"
    assert fm["tags"] == ["carry"], f"expected tags=['carry'], got {fm.get('tags')!r}"
