"""Verification for the forage launcher (ideation engine spec 2026-09-08 §5) and the leap
launcher (§6): both are TRUSTED DRIVERS wrapping a sandboxed Codex agent, and what these tests
pin is the privilege story (what the agent may reach) plus the driver's own authority-side step
list — the two things a refactor can silently weaken. Also carries the systemd units and the
sources-registry seed check for both stages.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FORAGE = REPO / ".codex" / "scripts" / "forage.sh"
LEAP = REPO / ".codex" / "scripts" / "leap.sh"


def _dry(script, *args):
    return subprocess.run(["bash", str(script), "--dry-run", *args], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout


def test_forage_dry_run_is_sandboxed_web_search_only_no_registry():
    out = _dry(FORAGE, "--categories", "momentum,seasonality", "--max-notes", "4",
               "--timeout", "10m")
    assert "DRY RUN" in out and "codex exec" in out
    assert "-s workspace-write" in out
    assert "sandbox_workspace_write.network_access=false" in out
    assert "web_search=live" in out
    assert "--dangerously-bypass-approvals-and-sandbox" not in out
    assert "mcp_servers" not in out                       # MCP is opt-in (FORAGE_MCP=1)
    # No registry path VALUE reaches the agent (an `ALGUA_DB_PATH=...` assignment would be a leak;
    # the `-u ALGUA_DB_PATH` UNSET directive asserted below is the opposite of that).
    assert "ALGUA_DB_PATH=" not in out.split("would run:")[1]
    # The codex CHILD must have these four stripped even though the driver's own environment (and
    # its later accept/propose/digest calls) legitimately needs them.
    assert ("env -u ALGUA_DB_PATH -u ALGUA_KNOWLEDGE_DIR -u ALGUA_DATA_DIR "
            "-u ALGUA_MLFLOW_TRACKING_URI") in out
    assert "research inspirations accept" in out          # trusted driver lands the notes
    assert "categories: momentum,seasonality" in out and "max notes: 4" in out
    assert "forage/" in out and "timeout 10m" in out


def test_forage_mcp_opt_in_drops_the_sandbox_and_says_so():
    out = subprocess.run(["bash", str(FORAGE), "--dry-run"], cwd=REPO, capture_output=True,
                         text=True, check=True, env={**os.environ, "FORAGE_MCP": "1"}).stdout
    assert "--dangerously-bypass-approvals-and-sandbox" in out
    assert "NO OS WALL" in out and "paper-search-mcp==" in out  # pinned spec


def test_forage_rejects_unknown_argument():
    proc = subprocess.run(["bash", str(FORAGE), "--bogus"], cwd=REPO, capture_output=True,
                          text=True)
    assert proc.returncode == 2


def test_forage_units_shaped_and_daily():
    svc = (REPO / "deploy/systemd/algua-forage.service").read_text()
    assert "Type=oneshot" in svc and "forage.sh" in svc and "TimeoutStartSec=" in svc
    tmr = (REPO / "deploy/systemd/algua-forage.timer").read_text()
    assert "OnCalendar=*-*-* 03:00:00 UTC" in tmr and "Persistent=true" in tmr


def test_sources_registry_seed_has_venues_for_every_category():
    import yaml
    reg = yaml.safe_load((REPO / "kb/inspirations/_sources.yaml").read_text())
    cats = {line.split()[0] for line in (REPO / ".codex/categories.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")}
    covered = set()
    for v in reg["venues"]:
        covered |= set(v["categories"])
    assert cats <= covered


def test_forage_dry_run_leaves_the_rotation_cursor_untouched(tmp_path):
    cursor = tmp_path / "forage-cursor"
    cursor.write_text("3")
    subprocess.run(["bash", str(FORAGE), "--dry-run"], cwd=REPO, capture_output=True, text=True,
                   check=True, env={**os.environ, "ALGUA_DATA_DIR": str(tmp_path)})
    assert cursor.read_text() == "3"


def test_forage_lock_skip_leaves_the_rotation_cursor_untouched(tmp_path):
    # The cursor write-back happens only after the flock is held AND the worktree + uv sync
    # succeed — a lock-skip must exit 0 before ever touching CURSOR_FILE, so the next firing
    # retries the SAME rotation slice rather than silently skipping past it.
    cursor = tmp_path / "forage-cursor"
    cursor.write_text("3")
    lock = tmp_path / "forage.lock"
    holder = subprocess.Popen(["flock", "-n", str(lock), "sleep", "30"])
    try:
        deadline = time.monotonic() + 5
        while not lock.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.2)  # give flock a moment to actually acquire, not just create the file
        proc = subprocess.run(
            ["bash", str(FORAGE)], cwd=REPO, capture_output=True, text=True, timeout=30,
            env={**os.environ, "ALGUA_DATA_DIR": str(tmp_path)})
        assert proc.returncode == 0, proc.stderr
        assert cursor.read_text() == "3"
    finally:
        holder.terminate()
        holder.wait(timeout=5)


# --- Leap (spec §6) ------------------------------------------------------------------------------


def test_leap_dry_run_is_sandboxed_no_web_scratch_db_then_import():
    out = _dry(LEAP, "--max-ideas", "5", "--timeout", "10m", "--force")
    assert "-s workspace-write" in out
    assert "sandbox_workspace_write.network_access=false" in out
    assert "web_search=disabled" in out
    assert "mcp_servers" not in out and "--dangerously-bypass" not in out
    assert ".leap-scratch/data/algua.db" in out             # ALGUA_DB_PATH -> scratch copy
    assert "research idea import --from" in out            # trusted driver imports
    assert "research inspirations write-yield" in out       # scorecard -> _sources.yaml
    assert "max ideas: 5" in out and "leap/" in out


def test_leap_dry_run_prints_every_planned_driver_step():
    # The dry run is the ONLY cheap check that the driver still does its whole authority-side
    # step list (seed -> codex -> import -> yield -> digest) in the right order; each line here
    # is one step that would otherwise be droppable without any test noticing.
    out = _dry(LEAP, "--max-ideas", "5", "--timeout", "10m", "--force")
    assert "DRY RUN" in out
    assert "depth gate: skipped (--force)" in out
    assert "would create worktree: " in out and "on branch leap/" in out
    assert "max ideas: 5" in out
    assert "would seed scratch from: " in out
    assert "would pre-warm env: " in out
    assert "would run: " in out and "codex exec" in out and "timeout 10m" in out
    assert "would import via: uv run algua research idea import --from" in out
    assert "--seeded-max-id" in out and "--critic-file" in out
    assert ("would write yield via: uv run algua research inspirations write-yield "
            "--from-scorecard -") in out
    assert "would append digest to: " in out and "leap-runs.jsonl" in out
    # Order matters: nothing authoritative may be planned before the agent's run.
    assert out.index("would run: ") < out.index("would import via: ")
    assert out.index("would import via: ") < out.index("would write yield via: ")


def test_leap_dry_run_without_force_still_gates_on_pool_depth():
    out = _dry(LEAP, "--max-ideas", "5")
    assert "would check depth: " in out and "research idea depth" in out


def test_leap_rejects_unknown_argument():
    proc = subprocess.run(["bash", str(LEAP), "--bogus"], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 2


def test_leap_units_every_two_hours_at_half_past_and_disjoint_from_paper():
    from tests.test_operator_layer import _fire_minutes, _oncalendar
    leap = _fire_minutes(_oncalendar("algua-leap.timer"))
    paper = _fire_minutes(_oncalendar("algua-paper.timer"))
    assert leap == {30} and not (leap & paper)
    tmr = (REPO / "deploy/systemd/algua-leap.timer").read_text()
    assert "OnCalendar=*-*-* 00/2:30:00 UTC" in tmr


def test_leap_units_shaped_and_installed():
    svc = (REPO / "deploy/systemd/algua-leap.service").read_text()
    assert "Type=oneshot" in svc and "leap.sh" in svc
    assert "TimeoutStartSec=2100" in svc          # > LEAP_TIMEOUT (25m) + prewarm
    tmr = (REPO / "deploy/systemd/algua-leap.timer").read_text()
    assert "Persistent=true" in tmr and "WantedBy=timers.target" in tmr
    installer = (REPO / "deploy/systemd/install-user-units.sh").read_text()
    assert "algua-leap.service" in installer and "algua-leap.timer" in installer
