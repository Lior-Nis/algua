"""Durable per-strategy merge-back recovery journal (#485, Task 1 — findings #1/#8, C2).

The merge-back driver is a **saga across two non-transactional systems** (git and the registry
SQLite DB). The lifecycle ``stage`` alone cannot prove *which* branch was merged, *which* commit was
tested, or whether a paper allocation corresponds to the intended code — so recovery is driven by
this durable journal, keyed on the **immutable branch-tip SHA** (NOT the mutable branch name), never
by re-deriving progress from ``stage`` + git ancestry.

The journal is **per-strategy** (``merge_back.<strategy>.journal`` beside the registry db): each
strategy's records live in their own file so the driver's per-strategy lock is that file's sole
writer, and two different-strategy attempts never race the same file. It is the driver's OWN
recovery state — NOT registry-domain state — so it needs no ``db.py`` schema bump and stays out of
CODEOWNERS-protected ``store.py``. This module imports **nothing** from ``algua`` (stdlib only), so
``algua.operator`` remains an import-linter leaf.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

# The exact strict-agent gate context the merge-back driver hard-wires (R4). Folded into the
# attempt-token derivation so a row minted under ANY other input set (a relaxed human run) can never
# collide with this attempt's token even by coincidence. The tuple is canonical + deterministic, so
# every recovery pass re-derives the identical fingerprint.
_STRICT_RELAXATION_CANONICAL = (
    "actor=agent;declared_combos=none;allow_holdout_reuse=false;"
    "allow_non_pit=false;assume_terminal_last_close=false;new_family=false"
)


def strict_relaxation_fingerprint() -> str:
    """SHA-256 of the canonical strict-agent relaxation tuple (R4). Deterministic — re-derivable on
    every recovery pass."""
    return hashlib.sha256(_STRICT_RELAXATION_CANONICAL.encode("utf-8")).hexdigest()


def derive_attempt_token(
    strategy: str, branch_tip: str, merge_sha: str, relaxation_fingerprint: str
) -> str:
    """The opaque per-attempt idempotency key stamped on this attempt's ``gate_evaluations`` row
    (C1/C2/R4). A deterministic function of durable fields only, so every recovery pass re-derives
    the *same* token — there is no fresh ``MAX(id)`` snapshot to drift. The relaxation fingerprint
    is part of the pre-image, so a row minted under a different input set cannot claim it."""
    payload = f"mergeback:v2:{strategy}:{branch_tip}:{merge_sha}:{relaxation_fingerprint}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MergeBackRecord:
    """One merge-back attempt, identified by ``(strategy, branch_tip)``.

    ``branch_tip`` (the resolved branch SHA at attempt start) is the immutable attempt identity; the
    branch *name* is informational only. The latest record for a ``(strategy, branch_tip)`` pair is
    authoritative — recovery resumes from the last durably-recorded step, never from ``stage`` +
    ancestry.
    """

    strategy: str
    branch: str
    branch_tip: str
    base_sha: str | None = None
    diff_policy: str = "pending"          # pending | passed | rejected
    gate_status: str = "pending"          # pending | green | failed
    merge_sha: str | None = None
    push_status: str = "pending"          # pending | pushed
    attempt_token: str | None = None
    promote_status: str = "pending"       # pending | passed | failed
    promote_gate_id: int | None = None
    intake_status: str = "pending"        # pending | allocated | queued | failed
    revert_sha: str | None = None
    terminal: str | None = None           # null while in-flight, else a terminal outcome


class Journal(Protocol):
    """The durable recovery-journal seam the orchestrator reads/writes, injected for testability."""

    def latest(self, strategy: str, branch_tip: str) -> MergeBackRecord | None:
        """The most-recently-appended record for ``(strategy, branch_tip)``, or None if none."""
        ...

    def append(self, record: MergeBackRecord) -> None:
        """Durably append ``record`` (fsync'd) as the latest for its ``(strategy, branch_tip)``."""
        ...


_SAFE = re.compile(r"[^A-Za-z0-9._-]")


class JsonlJournal:
    """Per-strategy append-only JSONL journal under a directory (atomic append + fsync).

    Each strategy gets its own ``merge_back.<sanitized-strategy>.journal`` file, so the driver's
    per-strategy lock is that file's sole writer. Appends are line-oriented and fsync'd; a crash
    that tears the final line leaves an unparseable trailing record, which the reader skips (keeping
    the last WELL-FORMED record), so a torn write never corrupts recovery.
    """

    def __init__(self, dir_path: Path) -> None:
        self._dir = dir_path

    def _path(self, strategy: str) -> Path:
        # Strategy names are already validated registry identifiers; sanitize defensively so a name
        # can never escape the journal directory or collide across the filesystem.
        safe = _SAFE.sub("_", strategy)
        return self._dir / f"merge_back.{safe}.journal"

    def append(self, record: MergeBackRecord) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(record.strategy)
        line = json.dumps(asdict(record), sort_keys=True) + "\n"
        # Append + fsync the file so the record survives a power loss; fsync the directory so the
        # (possibly new) file's directory entry is durable too.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        dir_fd = os.open(self._dir, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def latest(self, strategy: str, branch_tip: str) -> MergeBackRecord | None:
        path = self._path(strategy)
        if not path.exists():
            return None
        found: MergeBackRecord | None = None
        with path.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    # A torn final line from a crash mid-append — stop; the last well-formed record
                    # before it is authoritative.
                    break
                if data.get("strategy") == strategy and data.get("branch_tip") == branch_tip:
                    found = MergeBackRecord(**data)
        return found
