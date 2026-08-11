"""Verification for the skills-first agent operating layer (.codex/).

This slice is skills + config + one launcher script, so the tests verify wiring rather than
behavior: the launcher's dry-run emits the bounded, sandboxed codex command; every skill has the
required frontmatter; and the skills are reachable via the portable symlink paths.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILLS = REPO / ".codex" / "skills"
LAUNCHER = REPO / ".codex" / "scripts" / "run-research-loop.sh"
SOURCE_LAUNCHER = REPO / ".codex" / "scripts" / "source-ideas.sh"
SKILL_NAMES = [
    "operating-algua",
    "author-a-strategy",
    "run-the-research-loop",
    "interpret-results",
    "source-ideas",
]


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text()
    assert text.startswith("---\n"), f"{path} is missing YAML frontmatter"
    _, front, _body = text.split("---\n", 2)
    out: dict[str, str] = {}
    for line in front.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
    return out


def test_launcher_dry_run_emits_bounded_sandboxed_codex_command():
    proc = subprocess.run(
        ["bash", str(LAUNCHER), "--dry-run", "--hypotheses", "2", "--timeout", "10m"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    out = proc.stdout
    assert "DRY RUN" in out
    assert "codex exec" in out
    # REAL filesystem containment — the workspace-write sandbox confines writes to the worktree.
    # The old bypass-sandbox flag must NOT be used (env routing alone is not a wall).
    assert "-s workspace-write" in out
    assert "approval_policy=never" in out                         # headless, no prompts
    assert "--dangerously-bypass-approvals-and-sandbox" not in out
    assert "timeout 10m" in out                                  # OS-level hard bound
    assert "research-run/" in out                                # isolated branch
    assert ".funnel-scratch" in out                              # per-run scratch funnel
    assert "hypotheses: 2" in out                                # goal-level bound


def test_launcher_rejects_unknown_argument():
    proc = subprocess.run(
        ["bash", str(LAUNCHER), "--bogus"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert proc.returncode == 2


def test_source_ideas_dry_run_emits_web_tooled_pool_sourcing():
    proc = subprocess.run(
        ["bash", str(SOURCE_LAUNCHER), "--dry-run", "--thesis", "momentum", "--max-ideas", "3",
         "--timeout", "10m"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    out = proc.stdout
    assert "DRY RUN" in out
    assert "--dangerously-bypass-approvals-and-sandbox" in out  # MCP tools need full bypass (spike)
    assert "web_search=live" in out  # the web tooling this issue adds
    assert "paper_search_mcp" in out  # arXiv/SSRN MCP wired
    assert "research idea" in out  # sources into #126's pool, not files
    assert "ALGUA_DB_PATH=" in out  # persistent pool, not the throwaway worktree DB
    assert "timeout 10m" in out  # OS-level hard bound
    assert "source-ideas/" in out  # isolated branch


def test_source_ideas_rejects_unknown_argument():
    proc = subprocess.run(
        ["bash", str(SOURCE_LAUNCHER), "--bogus"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert proc.returncode == 2


def test_every_skill_has_name_and_description_frontmatter():
    for name in SKILL_NAMES:
        fm = _frontmatter(SKILLS / name / "SKILL.md")
        assert fm.get("name") == name, f"{name}: frontmatter name must equal the directory name"
        assert fm.get("description"), f"{name}: frontmatter description is required"


def test_mergeback_drain_systemd_units_present_and_shaped():
    # Factory slice 3: the auto merge-back drainer's own oneshot service + 30-minute timer pair,
    # following the exact conventions of the existing algua-research/algua-paper pairs.
    svc = (REPO / "deploy" / "systemd" / "algua-mergeback-drain.service").read_text()
    assert "Type=oneshot" in svc
    assert "drain-mergeback-queue.sh" in svc
    assert "TimeoutStartSec=" in svc

    tmr = (REPO / "deploy" / "systemd" / "algua-mergeback-drain.timer").read_text()
    assert "OnCalendar=" in tmr
    assert "Persistent=true" in tmr
    assert "WantedBy=timers.target" in tmr


def test_install_user_units_includes_mergeback_drain_pair():
    installer = (REPO / "deploy" / "systemd" / "install-user-units.sh").read_text()
    assert "algua-mergeback-drain.service" in installer
    assert "algua-mergeback-drain.timer" in installer


def test_skills_reachable_via_claude_skills_symlinks():
    # Canonical skills live in .codex/skills/ (Codex). Claude Code reads .claude/skills/,
    # so the same skills serve the co-dev harness too via symlink.
    for name in SKILL_NAMES:
        p = REPO / ".claude/skills" / name / "SKILL.md"
        assert p.exists(), f"{p} not reachable — .claude/skills symlink missing"
