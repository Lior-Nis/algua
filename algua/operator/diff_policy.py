"""CODEOWNERS-aware diff allow/deny gate for the autonomous merge-back (#485, Task 2 — finding #3,
hardened for C5/R5).

The autonomous merge is a **security boundary the quality gate is not** (a branch can weaken the
very tests/config the gate runs), so before any merge the driver gates the branch's change set
*independently* of whether the branch's own gate passes. This module is pure (stdlib only) and takes
the raw diff entries + the CODEOWNERS text, returning a hard allow/deny verdict:

* **Reject-by-default allowlist** — only strategy artifacts
  (``algua/strategies/<family>/**.py``) and ``kb/**`` report/doc artifacts may be added/modified.
* **CODEOWNERS-derived denylist** — every protected path (parsed from CODEOWNERS at runtime so it
  cannot drift) plus static extras (``tests/**``, build/lint/type config, ``algua/operator/**``) is
  denied outright, even on an allowlisted-looking path.
* **Object-mode guard (C5)** — symlinks (``120000``), gitlinks (``160000``), and unknown modes are
  denied (a symlink under ``algua/strategies/`` could point at ``../registry/store.py``); an exec
  bit on a ``.py`` strategy file is denied (strategy artifacts are non-executable data).
* **Rename/copy dual-path guard (R5)** — a rename/copy is evaluated against BOTH its source and
  destination, so ``git mv store.py algua/strategies/foo/store.py`` (denied source dodged by a
  destination-only view) and ``cp store.py -> allowlisted dest`` are both rejected.
* **Path canonicalization (C5)** — NFC-normalize, reject ``..``/absolute/``.git`` segments, and
  match case-folded, so a Unicode/case trick cannot present a denied path as allowlisted.

``CODEOWNERS`` failing to parse fails closed. Imports nothing from ``algua``.
"""

from __future__ import annotations

import posixpath
import unicodedata
from dataclasses import dataclass

# Regular git blob modes. Anything else (symlink 120000, gitlink 160000, unknown) is denied.
_REGULAR_MODES = frozenset({"100644", "100755"})

# Static denylist extras NOT expressible from CODEOWNERS: build/lint/type config, the test suite the
# gate runs, and the driver's own package (it may never merge changes to itself). Matched as path
# prefixes on the canonical path.
_STATIC_DENY_PREFIXES = (
    "tests/",
    "algua/operator/",
    ".github/",
)
_STATIC_DENY_EXACT = frozenset({
    "pyproject.toml",
    "ruff.toml",
    "mypy.ini",
    "setup.cfg",
    ".importlinter",
    "codeowners",          # canonicalized (case-folded) "CODEOWNERS"
    "uv.lock",
})

# Allowlist: strategy artifacts + kb docs. Canonical (case-folded) prefixes/suffix rules.
_ALLOW_STRATEGY_PREFIX = "algua/strategies/"
_ALLOW_KB_PREFIX = "kb/"


@dataclass(frozen=True)
class DiffEntry:
    """One ``git diff --raw -M -C`` entry.

    ``mode`` is the POST-image (destination) git mode (e.g. ``100644``; ``000000`` for a deletion).
    ``change_type`` is the raw status letter (``A`` add, ``M`` modify, ``D`` delete, ``R`` rename,
    ``C`` copy, ``T`` type-change). ``old_path`` is the source for renames/copies (else None or the
    same as ``new_path``); ``new_path`` is the destination (the surviving path).
    """

    mode: str
    change_type: str
    old_path: str | None
    new_path: str


@dataclass(frozen=True)
class DiffPolicyResult:
    """Verdict over a full change set. ``ok`` is True iff EVERY entry passed; ``rejected`` lists the
    ``(path, reason)`` pairs that failed (empty iff ``ok``)."""

    ok: bool
    rejected: tuple[tuple[str, str], ...]


def _canonicalize(path: str) -> str | None:
    """NFC-normalize + case-fold ``path``, rejecting traversal/absolute/``.git`` tricks.

    Returns the canonical (case-folded, forward-slash) path, or None if the path is unsafe or does
    not round-trip through normalization (a homoglyph/encoding attack).
    """
    if not path:
        return None
    nfc = unicodedata.normalize("NFC", path)
    if nfc != path:
        # A non-NFC input (NFD homoglyph, etc.) — refuse rather than silently normalize a path that
        # presented differently to a human reviewer.
        return None
    if path.startswith("/") or path.startswith("\\"):
        return None
    norm = path.replace("\\", "/")
    parts = norm.split("/")
    if any(seg in ("..", ".git") for seg in parts):
        return None
    # posixpath.normpath collapses redundant separators; reject if it changes the segment structure
    # (a further traversal-smuggling guard).
    collapsed = posixpath.normpath(norm)
    if collapsed != norm or collapsed.startswith("/") or collapsed.startswith(".."):
        return None
    return collapsed.casefold()


def parse_codeowners_denylist(codeowners_text: str) -> tuple[str, ...]:
    """Derive canonical deny prefixes from the CODEOWNERS file text.

    Each non-comment CODEOWNERS line's first token is a path pattern (``/algua/registry/store.py``,
    ``/approvers/``). We strip the leading slash and canonicalize it into a deny prefix. Raises
    ``ValueError`` if the text yields no owned paths at all (a CODEOWNERS that failed to parse must
    fail closed, never silently allow protected writes).
    """
    prefixes: list[str] = []
    for raw in codeowners_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        pattern = token.lstrip("/").rstrip("/")
        if not pattern:
            continue
        canon = _canonicalize(pattern)
        if canon is None:
            raise ValueError(f"CODEOWNERS path {token!r} does not canonicalize; failing closed")
        prefixes.append(canon)
    if not prefixes:
        raise ValueError("CODEOWNERS yielded no owned paths; failing closed")
    return tuple(prefixes)


def _is_denied(canon: str, deny_prefixes: tuple[str, ...]) -> bool:
    if canon in _STATIC_DENY_EXACT:
        return True
    if any(canon == p or canon.startswith(p + "/") or canon.startswith(p)
           for p in _STATIC_DENY_PREFIXES):
        return True
    for p in deny_prefixes:
        # A CODEOWNERS entry is either a file (exact match) or a directory tree (prefix match).
        if canon == p or canon.startswith(p + "/"):
            return True
    return False


def _is_allowed(canon: str) -> bool:
    if canon.startswith(_ALLOW_KB_PREFIX):
        return True
    if canon.startswith(_ALLOW_STRATEGY_PREFIX) and canon.endswith(".py"):
        # algua/strategies/<family>/<file>.py — require a family segment (not a top-level file).
        rest = canon[len(_ALLOW_STRATEGY_PREFIX):]
        return "/" in rest
    return False


def _check_path(path: str, deny_prefixes: tuple[str, ...]) -> tuple[bool, str]:
    """Allow/deny a single canonicalized-or-raw path. Returns ``(ok, reason)``."""
    canon = _canonicalize(path)
    if canon is None:
        return False, f"non-canonical path: {path!r}"
    if _is_denied(canon, deny_prefixes):
        return False, f"denylisted path: {path!r}"
    if not _is_allowed(canon):
        return False, f"outside allowlist (algua/strategies/<family>/**.py, kb/**): {path!r}"
    return True, ""


def evaluate_diff(entries: list[DiffEntry], codeowners_text: str) -> DiffPolicyResult:
    """Hard allow/deny gate over a full change set. Rejects on the first problem PER entry but
    accumulates all rejected entries, so the caller sees every reason. Fail-closed on unparseable
    CODEOWNERS (raises ``ValueError``)."""
    deny_prefixes = parse_codeowners_denylist(codeowners_text)
    rejected: list[tuple[str, str]] = []

    for e in entries:
        label = e.new_path or e.old_path or "<unknown>"

        # Deletions are never permitted: a strategy branch ADDS on-main strategy code, it never
        # removes it, and deleting a denylisted path is obviously out too.
        if e.change_type.startswith("D"):
            rejected.append((label, "deletion is not permitted"))
            continue

        # Object-mode guard (C5): only regular blobs; no symlink/gitlink/unknown mode; no exec .py.
        if e.mode not in _REGULAR_MODES:
            rejected.append((label, f"non-regular git mode {e.mode!r} (symlink/gitlink/unknown)"))
            continue
        if e.mode == "100755" and (e.new_path or "").casefold().endswith(".py"):
            rejected.append((label, "executable bit on a .py strategy artifact"))
            continue

        # Rename/copy dual-path guard (R5): evaluate BOTH source and destination.
        paths_to_check = [e.new_path]
        if e.change_type.startswith(("R", "C")) and e.old_path is not None:
            paths_to_check.append(e.old_path)

        entry_ok = True
        for p in paths_to_check:
            if p is None:
                continue
            ok, reason = _check_path(p, deny_prefixes)
            if not ok:
                rejected.append((p, reason))
                entry_ok = False
        if not entry_ok:
            continue

    return DiffPolicyResult(ok=not rejected, rejected=tuple(rejected))
