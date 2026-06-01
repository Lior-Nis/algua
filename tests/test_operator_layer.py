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
SKILL_NAMES = [
    "operating-algua",
    "author-a-strategy",
    "run-the-research-loop",
    "interpret-results",
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
    assert "--dangerously-bypass-approvals-and-sandbox" in out  # yolo, but contained in a worktree
    assert "timeout 10m" in out                                  # OS-level hard bound
    assert "research-run/" in out                                # isolated branch
    assert "uv sync" in out                                       # worktree env pre-warm
    assert "2 strategy hypotheses" in out                        # goal-level bound


def test_launcher_rejects_unknown_argument():
    proc = subprocess.run(
        ["bash", str(LAUNCHER), "--bogus"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert proc.returncode == 2


def test_every_skill_has_name_and_description_frontmatter():
    for name in SKILL_NAMES:
        fm = _frontmatter(SKILLS / name / "SKILL.md")
        assert fm.get("name") == name, f"{name}: frontmatter name must equal the directory name"
        assert fm.get("description"), f"{name}: frontmatter description is required"


def test_skills_reachable_via_claude_skills_symlinks():
    # Canonical skills live in .codex/skills/ (Codex). Claude Code reads .claude/skills/,
    # so the same skills serve the co-dev harness too via symlink.
    for name in SKILL_NAMES:
        p = REPO / ".claude/skills" / name / "SKILL.md"
        assert p.exists(), f"{p} not reachable — .claude/skills symlink missing"
