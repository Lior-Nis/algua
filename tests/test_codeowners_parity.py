"""GitHub-visible CODEOWNERS coverage must equal the runtime merge-back denylist (#648).

GitHub resolves exactly ONE CODEOWNERS file (``.github/`` -> root -> ``docs/``, first found wins).
``operator/diff_policy.py`` is fed the root file. If a second file ever shadows the root one,
GitHub enforces code-owner review on only that file's paths while the runtime keeps denying the
root file's: the wall is silently one-legged. Single source of truth = root ``CODEOWNERS``.
"""

from __future__ import annotations

import pathlib
import subprocess

from algua.operator.diff_policy import parse_codeowners_denylist

REPO = pathlib.Path(__file__).resolve().parents[1]
# GitHub's documented lookup order; the first existing file wins.
GITHUB_LOOKUP = (".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS")


def _tracked_codeowners() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return sorted(p for p in out if p.rsplit("/", 1)[-1].upper() == "CODEOWNERS")


def _github_resolved(files: list[str]) -> str | None:
    return next((c for c in GITHUB_LOOKUP if c in files), None)


def test_root_codeowners_is_the_only_codeowners_file() -> None:
    assert _tracked_codeowners() == ["CODEOWNERS"], (
        "exactly one CODEOWNERS file may exist (the root one); a second shadows it on GitHub"
    )


def test_github_resolves_the_file_the_runtime_denylist_reads() -> None:
    resolved = _github_resolved(_tracked_codeowners())
    assert resolved == "CODEOWNERS", (
        f"GitHub would enforce {resolved!r} but the runtime denylist reads root CODEOWNERS"
    )
    # The resolver itself must detect a shadowing file.
    assert _github_resolved(["CODEOWNERS", ".github/CODEOWNERS"]) == ".github/CODEOWNERS"


def test_github_visible_rules_equal_runtime_denylist() -> None:
    resolved = _github_resolved(_tracked_codeowners())
    assert resolved is not None
    text = (REPO / resolved).read_text()
    github_paths = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].split()
        if not line:
            continue
        # A rule with no owner is treated by GitHub as "unowned" while the runtime still denies it.
        assert len(line) >= 2 and all(o.startswith("@") for o in line[1:]), raw
        github_paths.add(line[0].strip("/").casefold())
    runtime = set(parse_codeowners_denylist((REPO / "CODEOWNERS").read_text()))
    assert github_paths == runtime
    assert {".github", ".gitleaks.toml", ".pip-audit-ignore.txt", "codeowners"} <= runtime
