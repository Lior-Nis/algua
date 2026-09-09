"""Verification for the forage launcher (ideation engine spec 2026-09-08 §5) and, once Task 9
lands it, the leap launcher (§6). Task 8 wires forage only; `leap.sh` does not exist yet, so its
tests are added alongside it in Task 9 — this file carries the forage tests plus the units + the
sources-registry seed check.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FORAGE = REPO / ".codex" / "scripts" / "forage.sh"


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
