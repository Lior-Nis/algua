"""The ideation category vocabulary (`.codex/categories.txt`).

ONE reader for the file that is a human steering surface (spec 2026-09-08 §4, PRD §7): the forage
rotation, the leap prompt, `research inspirations propose/accept`, `research idea add` and the
trusted leap import all key on the SAME slug list. Duplicating the parser is how a slug becomes
legal in one lane and unknown in another.

Pure: stdlib only, no algua imports — the registry, the CLI and the research lane may all read it.
"""
from __future__ import annotations

from pathlib import Path

#: Repo-relative location of the category file (the human's steering surface).
CATEGORIES_REL = Path(".codex/categories.txt")


def categories_file_default() -> Path:
    """The repo's `.codex/categories.txt`, located by walking up from THIS file to the directory
    holding `pyproject.toml`. Never `Path.cwd()` — a driver may run from any working directory,
    but the category file lives at a fixed repo-relative location."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent / CATEGORIES_REL
    raise RuntimeError(f"could not locate pyproject.toml walking up from {here}")


def load_categories(path: Path | None = None) -> list[str]:
    """Category slugs in file order: the first whitespace-separated token of each non-comment,
    non-blank line (the rest of the line is an optional market=/horizon= hint)."""
    resolved = categories_file_default() if path is None else path
    if not resolved.exists():
        raise FileNotFoundError(f"categories file not found: {resolved}")
    slugs: list[str] = []
    for line in resolved.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        slug = stripped.split()[0]
        if slug not in slugs:
            slugs.append(slug)
    return slugs
