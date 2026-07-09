"""Structural repo-hygiene gate (issue #509).

The autonomous research loop churns runtime artifacts (scratchpads, `.sync.lock` files,
generated strategy modules). Without a gate, those leak into git as committed junk, the repo
root accretes stray files, and unplaced/un-stamped strategy modules slip past provenance
discipline. This module fails pytest the moment any of that is COMMITTED.

Scope is the git INDEX, not a filesystem walk (contrast the #277 data-wall scanner, which
rglobs the tree): CI cares only about what is tracked, and enumerating `git ls-files` avoids
false positives from local untracked scratch files a developer legitimately keeps in the tree.

Three properties:
  1. the repo root holds only an explicit whitelist of files;
  2. no tracked path is runtime junk (scratchpads, lock files, editor/OS cruft);
  3. every committed strategy module is correctly PLACED (in a family subdir, not stray at the
     strategies top level) and carries a module-level `GENERATED_BY` provenance stamp, checked
     AST-precisely (like #277) rather than by substring.
"""

import ast
import fnmatch
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]

# The ONLY files allowed to live at the repo root. Adding a root file is a deliberate act:
# extend this set on purpose, or (preferably) place the file in a subdirectory.
ROOT_WHITELIST = frozenset(
    {
        "AGENTS.md",
        "CLAUDE.md",
        "CODEOWNERS",
        "docker-compose.yml",
        "Dockerfile",
        ".dockerignore",
        ".env.example",
        ".gitignore",
        ".gitleaks.toml",
        ".pip-audit-ignore.txt",
        "pyproject.toml",
        "README.md",
        "uv.lock",
    }
)

# Runtime/editor/OS cruft that must never be committed. Matched against each path's basename.
JUNK_PATTERNS = [
    "*.sync.lock",
    "scratchpad-*",
    ".DS_Store",
    "Thumbs.db",
    "*.swp",
    "*.swo",
    "*~",
]

STRAT_PREFIX = "algua/strategies/"
# Modules allowed directly at the strategies top level: infrastructure, not strategies.
INFRA_TOP_LEVEL = frozenset({"__init__.py", "base.py", "loader.py"})


def _tracked_files() -> list[str]:
    """Every git-tracked path in the repo, as forward-slash relative strings (blanks dropped)."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _has_generated_by(tree: ast.Module) -> bool:
    """True iff the module has a MODULE-LEVEL assignment to a Name target `GENERATED_BY`.

    AST-precise (mirrors the #277 data-wall scanner's rigor): only top-level `tree.body`
    statements count, so a `GENERATED_BY` buried inside a function/class or appearing in a
    string/comment does NOT satisfy the stamp.
    """
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "GENERATED_BY":
                    return True
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            # A bare `GENERATED_BY: str` annotation assigns no marker (node.value is None).
            if (
                node.value is not None
                and isinstance(target, ast.Name)
                and target.id == "GENERATED_BY"
            ):
                return True
    return False


def test_has_generated_by_recognizes_marker_forms() -> None:
    """`_has_generated_by` accepts plain and value-carrying annotated assignments only.

    Regression guard (issue #509): a BARE `GENERATED_BY: str` annotation assigns no marker and
    must be REJECTED — mirroring algua/cli/app.py's AST check. Without the `node.value is not None`
    guard this silently regressed to accepting an un-stamped module.
    """
    assert _has_generated_by(ast.parse('GENERATED_BY = "author-a-strategy"'))
    assert _has_generated_by(ast.parse('GENERATED_BY: str = "author-a-strategy"'))
    # Bare annotation (no value) assigns nothing → not a valid stamp.
    assert not _has_generated_by(ast.parse("GENERATED_BY: str"))
    # No marker at all.
    assert not _has_generated_by(ast.parse("OTHER = 1"))
    # Nested (function-scoped) assignment does not count as module-level.
    assert not _has_generated_by(ast.parse("def f():\n    GENERATED_BY = 1\n"))


def test_repo_root_is_whitelisted() -> None:
    """Fail-closed: every root-level tracked file must be in ROOT_WHITELIST."""
    offenders = [
        path
        for path in _tracked_files()
        if "/" not in path and path not in ROOT_WHITELIST
    ]
    assert not offenders, (
        f"repo-root violation: {offenders} committed at the repo root but not in "
        f"ROOT_WHITELIST. Place each in a subdirectory, or add it to ROOT_WHITELIST "
        f"deliberately if it truly belongs at the root."
    )


def test_no_tracked_junk() -> None:
    """Fail-closed: no tracked path may be runtime/editor/OS junk."""
    offenders: list[str] = []
    for path in _tracked_files():
        base = path.rsplit("/", 1)[-1]
        if any(fnmatch.fnmatch(base, pattern) for pattern in JUNK_PATTERNS):
            offenders.append(path)
    assert not offenders, (
        f"tracked junk committed (matches {JUNK_PATTERNS}): {offenders}. Remove these from git "
        f"(git rm --cached) and ensure .gitignore covers them."
    )


def test_strategies_placement_and_provenance() -> None:
    """Fail-closed: strategy modules must be PLACED in a family subdir and carry GENERATED_BY.

    Scoped to `git ls-files` (like the other two tests and this module's docstring): only TRACKED
    strategy modules are checked, so a local untracked scratch module never trips the gate. A
    non-infra module directly at the strategies top level is an unplaced strategy. Every strategy
    module inside a family subdir must carry a module-level GENERATED_BY stamp.
    """
    strat_paths = [
        path
        for path in _tracked_files()
        if path.startswith(STRAT_PREFIX) and path.endswith(".py")
    ]

    for path in sorted(strat_paths):
        rel = path[len(STRAT_PREFIX) :]
        segments = rel.split("/")
        name = segments[-1]
        if name.startswith("_"):
            continue

        # No stray non-infra module directly at the strategies top level.
        if len(segments) == 1:
            assert name in INFRA_TOP_LEVEL, (
                f"unplaced strategy: {path} sits at the strategies top level. Strategy modules "
                f"must live in a family subdirectory (e.g. algua/strategies/<family>/)."
            )
            continue

        # Every strategy module inside a family subdir carries a GENERATED_BY provenance stamp.
        tree = ast.parse((REPO / path).read_text())
        assert _has_generated_by(tree), (
            f"missing provenance: {path} must carry a module-level GENERATED_BY provenance "
            f"stamp (additions-only discipline, issue #509)."
        )
