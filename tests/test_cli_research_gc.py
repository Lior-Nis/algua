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
        [_classified(str(link)), _classified(str(real))], arch, hashes, [tmp_path])

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

    run_dir, moved, skipped = _gc_archive([_classified(str(f))], arch, stale, [tmp_path])

    assert moved == []
    assert [s["reason"] for s in skipped] == ["content_changed_since_authorization"]
    assert f.exists()  # untouched — never deleted, never archived


def test_gc_archive_refuses_symlinked_intermediate_dir(tmp_path):
    """An INTERMEDIATE symlinked directory (which O_NOFOLLOW's final-component check would miss) is
    refused via the scan-root containment guard: the escape target is never read or moved."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "loot.md").write_text("loot\n")
    root = tmp_path / "root"
    root.mkdir()
    # root/reports -> ../outside ; a src under it resolves OUTSIDE the scan root.
    (root / "reports").symlink_to(outside)
    src = root / "reports" / "loot.md"
    arch = tmp_path / "arch"

    run_dir, moved, skipped = _gc_archive(
        [_classified(str(src))], arch, {str(src): _sha(outside / "loot.md")}, [root])

    assert moved == []
    assert [s["reason"] for s in skipped] == ["escaped_scan_root"]
    assert (outside / "loot.md").exists()  # escape target untouched


def test_gc_archive_uses_atomic_replace_no_both_paths_window(tmp_path, monkeypatch):
    """The move is a single atomic renameat — asserted by a spy that, at call time, sees the source
    present and the dest absent (never a copy+unlink window where BOTH exist). The move now targets
    a dst_dir_fd + bare name, so dest existence is probed relative to that fd."""
    f = tmp_path / "r.md"
    f.write_text("bytes\n")
    arch = tmp_path / "arch"
    hashes = {str(f): _sha(f)}

    real_replace = os.replace
    seen = {}

    def _spy(src, dst, *a, **k):
        dfd = k["dst_dir_fd"]  # renameat form: dst is a bare name under this dir fd
        # At the atomic instant: source still there, destination not yet created.
        seen["src_before"] = os.path.exists(src)
        seen["dst_before"] = dst in os.listdir(dfd)
        real_replace(src, dst, *a, **k)
        seen["src_after"] = os.path.exists(src)
        seen["dst_after"] = dst in os.listdir(dfd)

    monkeypatch.setattr(os, "replace", _spy)
    run_dir, moved, skipped = _gc_archive([_classified(str(f))], arch, hashes, [tmp_path])

    assert len(moved) == 1 and skipped == []
    assert seen == {"src_before": True, "dst_before": False,
                    "src_after": False, "dst_after": True}
    assert not f.exists()


def test_gc_archive_skips_cross_filesystem_rather_than_copy(tmp_path, monkeypatch):
    """A cross-filesystem destination (EXDEV) is SKIPPED and surfaced — never falls back to a
    non-atomic copy, so the source is left in place untouched."""
    import errno as _errno

    f = tmp_path / "r.md"
    f.write_text("bytes\n")
    arch = tmp_path / "arch"
    hashes = {str(f): _sha(f)}

    def _exdev(src, dst, *a, **k):
        raise OSError(_errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "replace", _exdev)
    run_dir, moved, skipped = _gc_archive([_classified(str(f))], arch, hashes, [tmp_path])

    assert moved == []
    assert [s["reason"] for s in skipped] == ["cross_filesystem"]
    assert f.exists() and f.read_text() == "bytes\n"  # left in place, never copied


def _classified_retired(path: str, strategy: str = "s") -> lifecycle_gc.Classified:
    return lifecycle_gc.Classified(
        path=path, strategy=strategy, surface=lifecycle_gc.SURFACE_MODULE,
        size_bytes=1, reason=lifecycle_gc.REAP_RETIRED_EXPIRED, reapable=True,
        stage=lifecycle_gc.RETIRED, retired_at="2020-01-01T00:00:00+00:00", age_days=999.0)


def test_gc_archive_refuses_symlinked_dest_component(tmp_path, monkeypatch):
    """The DESTINATION is hardened symmetrically with the source: a planted symlinked run-dir
    component under --archive-dir is refused (archive_dest_unsafe), so os.replace can never follow
    it OUT of the archive tree. Mirrors the source-side intermediate-symlink test."""
    f = tmp_path / "r.md"
    f.write_text("bytes\n")
    arch = tmp_path / "arch"
    arch.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    # Pin the run id so we can pre-plant the exact run dir as a symlink pointing outside the tree.
    monkeypatch.setattr("algua.cli.research_cmd._archive_run_id", lambda: "FIXEDRUN")
    (arch / "FIXEDRUN").symlink_to(outside)

    run_dir, moved, skipped = _gc_archive(
        [_classified(str(f))], arch, {str(f): _sha(f)}, [tmp_path])

    assert moved == []
    assert [s["reason"] for s in skipped] == ["archive_dest_unsafe"]
    assert f.exists() and f.read_text() == "bytes\n"  # source untouched
    assert not any(outside.rglob("r.md"))  # never written through the symlink


def test_archive_run_id_is_collision_resistant():
    """Two runs in the same UTC second must get distinct run dirs: the id carries a random suffix so
    os.replace can never silently overwrite a prior archived file (#510 GATE-2)."""
    from algua.cli.research_cmd import _archive_run_id
    ids = {_archive_run_id() for _ in range(200)}
    assert len(ids) == 200  # all unique despite same-second stamps
    for rid in ids:
        stamp, _, suffix = rid.partition("-")
        assert stamp.endswith("Z") and len(suffix) == 32 and suffix.isalnum()


def test_gc_archive_rechecks_registry_stage_before_move(tmp_path):
    """The point-of-use registry re-check closes the window between classify() and the move (which
    spans an out-of-band human signing delay): a retired-expired item whose strategy is no longer
    `retired`, an orphan whose name now has a row, and a strategy that un-retired and RE-retired
    (resetting its retention clock) are ALL skipped."""
    # (a) un-retired between classify and move -> skipped, source left in place.
    f1 = tmp_path / "unretired.py"
    f1.write_text("x=1\n")
    run_dir, moved, skipped = _gc_archive(
        [_classified_retired(str(f1))], tmp_path / "a1", {str(f1): _sha(f1)}, [tmp_path],
        current_entry=lambda name: lifecycle_gc.RegistryEntry(stage="live", retired_at=None))
    assert moved == []
    assert [s["reason"] for s in skipped] == ["registry_stage_changed"]
    assert f1.exists()

    # (b) orphan report that got `registry add`ed (now has a row) -> skipped.
    f2 = tmp_path / "nowtracked.md"
    f2.write_text("y\n")
    run_dir, moved, skipped = _gc_archive(
        [_classified(str(f2))], tmp_path / "a2", {str(f2): _sha(f2)}, [tmp_path],
        current_entry=lambda name: lifecycle_gc.RegistryEntry(stage="idea", retired_at=None))
    assert moved == []
    assert [s["reason"] for s in skipped] == ["registry_stage_changed"]
    assert f2.exists()

    # (c) control: stage still retired at the SAME old timestamp -> the move proceeds.
    f3 = tmp_path / "still_retired.py"
    f3.write_text("z=1\n")
    run_dir, moved, skipped = _gc_archive(
        [_classified_retired(str(f3))], tmp_path / "a3", {str(f3): _sha(f3)}, [tmp_path],
        current_entry=lambda name: lifecycle_gc.RegistryEntry(
            stage=lifecycle_gc.RETIRED, retired_at="2020-01-01T00:00:00+00:00"),
        retention_days=90.0)
    assert [m["src"] for m in moved] == [str(f3)]
    assert not f3.exists()

    # (d) RE-retired between classify and move: live stage reads "retired" again, but a fresh
    # retired_at means the retention clock reset -> skipped, NOT re-authorized by the stage match
    # alone (#510 GATE-2: still_reapable re-derives eligibility, not a bare stage-equality check).
    f4 = tmp_path / "re_retired.py"
    f4.write_text("w=1\n")
    from datetime import UTC, datetime
    fresh_retired_at = datetime.now(UTC).isoformat()
    run_dir, moved, skipped = _gc_archive(
        [_classified_retired(str(f4))], tmp_path / "a4", {str(f4): _sha(f4)}, [tmp_path],
        current_entry=lambda name: lifecycle_gc.RegistryEntry(
            stage=lifecycle_gc.RETIRED, retired_at=fresh_retired_at),
        retention_days=90.0)
    assert moved == []
    assert [s["reason"] for s in skipped] == ["registry_stage_changed"]
    assert f4.exists()


def test_gc_top_bounds_archived_set_to_shown_challenge(tmp_path, monkeypatch):
    """--top must bound what is AUTHORIZED, not just what is displayed: the challenge's reapable
    list and reclaimable_bytes reflect exactly the top-N, so the archived set never exceeds it."""
    # Two orphaned reports of different sizes so ranking (size DESC) is deterministic.
    big = tmp_path / "kb" / "strategies" / "big" / "reports" / "20240101-000000"
    small = tmp_path / "kb" / "strategies" / "small" / "reports" / "20240101-000000"
    for d, payload in ((big, "X" * 500), (small, "y")):
        d.mkdir(parents=True, exist_ok=True)
        report = d / "report.md"
        report.write_text(payload)
        os.utime(report, (_OLD, _OLD))

    result = runner.invoke(
        app, ["research", "gc", "--archive", "--actor", "human", "--top", "1"])
    payload = _json(result)
    assert payload["action"] == "human_actor_challenge"
    # Only the top-1 (the big report) is in the authorized manifest and counts.
    assert payload["reapable_count"] == 1
    shown = {r["strategy"] for r in payload["reapable"]}
    assert shown == {"big"}
    assert payload["reclaimable_bytes"] == 500
