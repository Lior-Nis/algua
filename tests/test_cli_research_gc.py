"""Smoke tests for the `research gc` CLI command (#510)."""
import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_env(monkeypatch, tmp_path):
    # Isolate the registry DB and point the report surface at a tmp knowledge dir so the real
    # repo's kb/strategies/*.md files never leak into the scan.
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "kb"))


def _json(result):
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def _write_ghost_report(tmp_path):
    reports = tmp_path / "kb" / "strategies"
    reports.mkdir(parents=True, exist_ok=True)
    ghost = reports / "ghost.md"
    ghost.write_text("# ghost report\n")
    return ghost


def test_gc_empty_registry_no_reports_is_clean_and_dry():
    """Empty registry + no kb reports: exit 0, dry-run advisory, nothing reapable.

    Real-repo strategy modules with no registry row classify as untracked_module (kept), so the
    reapable list stays deterministically empty."""
    result = runner.invoke(app, ["research", "gc"])
    payload = _json(result)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["reapable"] == []
    assert "by_reason" in payload
    assert "reclaimable_bytes" in payload
    assert "retention_days" in payload


def test_gc_reaps_orphaned_report(tmp_path):
    """A top-level report with no registry row is flagged reapable as orphaned_report."""
    _write_ghost_report(tmp_path)
    result = runner.invoke(app, ["research", "gc"])
    payload = _json(result)
    ghosts = [r for r in payload["reapable"] if r["strategy"] == "ghost"]
    assert len(ghosts) == 1
    assert ghosts[0]["surface"] == "report"
    assert ghosts[0]["reason"] == "orphaned_report"


def test_gc_archive_is_human_only(tmp_path):
    """--archive under the default (agent) actor fails closed, mentioning the human/actor gate."""
    _write_ghost_report(tmp_path)
    result = runner.invoke(app, ["research", "gc", "--archive"])
    assert result.exit_code != 0
    lowered = result.stdout.lower()
    assert "human" in lowered
    assert "actor" in lowered


def test_gc_archive_moves_files_and_is_idempotent(tmp_path):
    """--archive --actor human MOVES the report into the archive tree; a re-run is a no-op."""
    ghost = _write_ghost_report(tmp_path)
    arch = tmp_path / "arch"
    result = runner.invoke(
        app, ["research", "gc", "--archive", "--actor", "human", "--archive-dir", str(arch)])
    payload = _json(result)
    assert len(payload["archived"]) == 1
    assert payload["archived"][0]["strategy"] == "ghost"
    assert not ghost.exists()
    # The file now lives somewhere under the archive tree.
    assert any(p.name == "ghost.md" for p in arch.rglob("*.md"))

    # Idempotent re-run: nothing left to move.
    result2 = runner.invoke(
        app, ["research", "gc", "--archive", "--actor", "human", "--archive-dir", str(arch)])
    payload2 = _json(result2)
    assert payload2["archived"] == []
