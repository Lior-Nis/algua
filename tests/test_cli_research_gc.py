"""Smoke tests for the `research gc` CLI command (#510)."""
import json
import os
import time

import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.cli.research_cmd import _gc_archive
from algua.research import lifecycle_gc
from tests._human_actor_helpers import install_human_actor_anchor, promote_signed

runner = CliRunner()

# Report-experiments writes reports under <knowledge_dir>/strategies/<name>/reports/<stamp>/, so an
# orphaned report older than the retention window (default 90d) must be backdated on disk to reap.
_OLD = time.time() - 200 * 86400


@pytest.fixture(autouse=True)
def _tmp_env(monkeypatch, tmp_path):
    # Isolate the registry DB and point the report surface at a tmp knowledge dir so the real
    # repo's kb/strategies/ files never leak into the scan.
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "kb"))


def _json(result):
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def _write_ghost_report(tmp_path, *, name: str = "ghost", age_old: bool = True):
    """Write a report-experiments artifact at strategies/<name>/reports/<stamp>/report.md."""
    rdir = tmp_path / "kb" / "strategies" / name / "reports" / "20240101-000000"
    rdir.mkdir(parents=True, exist_ok=True)
    report = rdir / "report.md"
    report.write_text("# ghost report\n")
    if age_old:
        os.utime(report, (_OLD, _OLD))
    return report


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
    """A report with no registry row, older than the window, is flagged reapable as orphaned_report,
    keyed by its <name> directory component."""
    _write_ghost_report(tmp_path)
    result = runner.invoke(app, ["research", "gc"])
    payload = _json(result)
    ghosts = [r for r in payload["reapable"] if r["strategy"] == "ghost"]
    assert len(ghosts) == 1
    assert ghosts[0]["surface"] == "report"
    assert ghosts[0]["reason"] == "orphaned_report"


def test_gc_fresh_orphaned_report_is_kept(tmp_path):
    """A freshly-written orphaned report (within retention) is NOT reapable; never reap on sight."""
    _write_ghost_report(tmp_path, name="fresh", age_old=False)
    payload = _json(runner.invoke(app, ["research", "gc"]))
    assert [r for r in payload["reapable"] if r["strategy"] == "fresh"] == []


def test_gc_ignores_kb_sync_owned_top_level_files(tmp_path):
    """The kb-sync-owned router pages (_*.md) and per-strategy synced notes (<name>.md) live as
    TOP-LEVEL files under strategies/ and must NEVER be scanned/reaped."""
    sdir = tmp_path / "kb" / "strategies"
    sdir.mkdir(parents=True, exist_ok=True)
    for fname in ("_index.md", "_by-stage.md", "momentum.md"):
        f = sdir / fname
        f.write_text("# synced\n")
        os.utime(f, (_OLD, _OLD))  # even old, they must be ignored
    payload = _json(runner.invoke(app, ["research", "gc"]))
    reaped = {r["strategy"] for r in payload["reapable"]}
    assert "_index" not in reaped and "_by-stage" not in reaped and "momentum" not in reaped
    assert payload["reapable"] == []


def test_gc_archive_without_human_actor_fails_closed(tmp_path):
    """--archive under the default (agent) actor fails closed, mentioning the human/actor gate."""
    _write_ghost_report(tmp_path)
    result = runner.invoke(app, ["research", "gc", "--archive"])
    assert result.exit_code != 0
    lowered = result.stdout.lower()
    assert "human" in lowered
    assert "actor" in lowered


def test_gc_archive_human_without_signature_prints_challenge_and_mutates_nothing(tmp_path):
    """--archive --actor human with NO signature issues a #329 challenge and moves nothing."""
    ghost = _write_ghost_report(tmp_path)
    result = runner.invoke(app, ["research", "gc", "--archive", "--actor", "human"])
    payload = _json(result)
    assert payload["action"] == "human_actor_challenge"
    assert "challenge" in payload and "manifest_sha256" in payload
    assert ghost.exists()  # nothing moved


def test_gc_archive_bad_signature_fails_closed(tmp_path, monkeypatch):
    """A bogus --actor-signature is refused (fail closed) — a bare --actor human unlocks nothing."""
    install_human_actor_anchor(monkeypatch, tmp_path)
    _write_ghost_report(tmp_path)
    bogus = tmp_path / "bogus.sig"
    bogus.write_bytes(b"not a signature")
    result = runner.invoke(
        app, ["research", "gc", "--archive", "--actor", "human",
              "--actor-signature", str(bogus)])
    assert result.exit_code != 0


def test_gc_archive_signed_moves_files_and_is_idempotent(tmp_path, monkeypatch):
    """A genuinely-authenticated human (signed challenge) MOVES the report; a re-run is a no-op."""
    key = install_human_actor_anchor(monkeypatch, tmp_path)
    ghost = _write_ghost_report(tmp_path)
    arch = tmp_path / "arch"
    base = ["research", "gc", "--archive", "--actor", "human", "--archive-dir", str(arch)]

    result = promote_signed(runner, app, base, key, tmp_path)
    payload = _json(result)
    assert len(payload["archived"]) == 1
    assert payload["archived"][0]["strategy"] == "ghost"
    assert not ghost.exists()
    assert any(p.name == "report.md" for p in arch.rglob("*.md"))

    # Idempotent re-run: nothing left to move, so no challenge is even issued.
    result2 = promote_signed(runner, app, base, key, tmp_path)
    payload2 = _json(result2)
    assert payload2["archived"] == []


def _classified(path: str) -> lifecycle_gc.Classified:
    return lifecycle_gc.Classified(
        path=path, strategy="x", surface=lifecycle_gc.SURFACE_REPORT,
        size_bytes=1, reason=lifecycle_gc.REAP_ORPHANED_REPORT, reapable=True,
        stage=None, retired_at=None, age_days=100.0)


def _sha(path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gc_archive_refuses_symlink_source_and_never_follows_it(tmp_path):
    """A source swapped to a symlink is refused (not followed): the target is untouched, the link
    stays, and only the genuine regular file is archived."""
    secret = tmp_path / "secret.txt"
    secret.write_text("do-not-touch\n")
    link = tmp_path / "evil.md"
    link.symlink_to(secret)  # type-swap: a symlink where a regular report was classified
    real = tmp_path / "real.md"
    real.write_text("real report\n")
    arch = tmp_path / "arch"
    hashes = {str(link): _sha(secret), str(real): _sha(real)}

    run_dir, moved, skipped = _gc_archive(
        [_classified(str(link)), _classified(str(real))], arch, hashes)

    assert [m["src"] for m in moved] == [str(real)]
    assert [s["src"] for s in skipped] == [str(link)]
    assert {s["reason"] for s in skipped} == {"refused_non_regular_file"}
    # The symlink and its target survive untouched; the regular file moved out.
    assert link.is_symlink() and secret.read_text() == "do-not-touch\n"
    assert not real.exists()
    assert any(p.name == "real.md" for p in arch.rglob("*.md"))


def test_gc_archive_refuses_content_changed_since_authorization(tmp_path):
    """A regular file whose bytes differ from the SIGNED hash is refused at the point of use — the
    archived bytes are always the signed bytes, never a raced-in replacement."""
    f = tmp_path / "r.md"
    f.write_text("current bytes\n")
    arch = tmp_path / "arch"
    # expected_hashes carries the hash of DIFFERENT (signed) content than what is on disk now.
    stale = {str(f): "0" * 64}

    run_dir, moved, skipped = _gc_archive([_classified(str(f))], arch, stale)

    assert moved == []
    assert [s["reason"] for s in skipped] == ["content_changed_since_authorization"]
    assert f.exists()  # untouched — never deleted, never archived
