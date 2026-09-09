"""Verification for the forage launcher (ideation engine spec 2026-09-08 §5) and, once Task 9
lands it, the leap launcher (§6). Task 8 wires forage only; `leap.sh` does not exist yet, so its
tests are added alongside it in Task 9 — this file carries the forage tests plus the units + the
sources-registry seed check.
"""

from __future__ import annotations

import subprocess
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
    assert "ALGUA_DB_PATH" not in out.split("would run:")[1]  # no registry path to the agent
    assert "research inspirations accept" in out          # trusted driver lands the notes
    assert "categories: momentum,seasonality" and "max notes: 4" in out
    assert "forage/" in out and "timeout 10m" in out


def test_forage_mcp_opt_in_drops_the_sandbox_and_says_so(monkeypatch):
    out = subprocess.run(["bash", str(FORAGE), "--dry-run"], cwd=REPO, capture_output=True,
                         text=True, check=True, env={**__import__("os").environ,
                                                     "FORAGE_MCP": "1"}).stdout
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
