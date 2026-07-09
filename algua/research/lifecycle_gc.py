"""Pure lifecycle garbage-collection classifier (#510) — no I/O, no DB.

Classifies on-disk strategy artifacts (strategy modules + reports) against the
registry to decide which are safely reapable. Advisory/pure: the caller supplies
the scanned file list, the registry snapshot, the clock, and the retention
window; this module only decides. Mirrors the purity of ``family_audit.py`` so
``lint-imports`` stays green (import-free beyond stdlib ``dataclasses`` +
``datetime``).

FAIL-SAFE bias throughout: a file is reaped ONLY when we can positively prove it
is either an orphaned report or a retired strategy older than the retention
window. Any ambiguity (missing row for a module, retired-without-timestamp,
non-terminal stage) keeps the file.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime

# The only terminal, reapable stage; see algua/contracts/lifecycle.py.
RETIRED = "retired"

# Surface constants — what kind of on-disk artifact a FileItem is.
SURFACE_MODULE = "strategy_module"
SURFACE_REPORT = "report"

# Reason constants — the verdict rationale carried on every Classified.
REAP_RETIRED_EXPIRED = "retired_expired"
REAP_ORPHANED_REPORT = "orphaned_report"
KEEP_RETIRED_WITHIN_RETENTION = "retired_within_retention"
KEEP_ORPHANED_WITHIN_RETENTION = "orphaned_within_retention"
KEEP_PROTECTED_NON_TERMINAL = "protected_non_terminal"
KEEP_UNTRACKED_MODULE = "untracked_module"

# The #329 human-actor trust namespace, reused to gate `research gc --archive` (see
# build_gc_archive_challenge). Signing under this namespace requires a key enrolled in
# approvers/allowed_signers FOR THIS namespace — a bare `--actor human` string authorizes nothing.
GC_ARCHIVE_NAMESPACE = "algua-human-actor"


@dataclass(frozen=True)
class FileItem:
    """A scanned on-disk artifact."""

    path: str          # path as scanned — reporting + archive move-source (absolute when scanned)
    strategy: str      # strategy name: module filename stem OR the report's <name> directory
    surface: str       # SURFACE_MODULE | SURFACE_REPORT
    size_bytes: int
    mtime: float | None = None  # file mtime (unix epoch s); provable-age floor for orphaned reports


@dataclass(frozen=True)
class RegistryEntry:
    """A registry snapshot row for one strategy."""

    stage: str
    retired_at: str | None  # ISO ts of the latest transition INTO retired, else None


@dataclass(frozen=True)
class Classified:
    """A FileItem with its GC verdict."""

    path: str
    strategy: str
    surface: str
    size_bytes: int
    reason: str
    reapable: bool
    stage: str | None
    retired_at: str | None
    age_days: float | None


def _age_days(retired_at: str, now: datetime) -> float:
    """Days elapsed from ``retired_at`` to ``now``.

    tz-naive ISO timestamps are stamped UTC (fail-safe: never compare a naive
    and an aware datetime, which would raise).
    """
    parsed = datetime.fromisoformat(retired_at)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (now - parsed).total_seconds() / 86400.0


def _age_days_from_mtime(mtime: float, now: datetime) -> float:
    """Days elapsed from a unix-epoch file mtime to ``now`` (the orphaned-report age floor)."""
    return (now.timestamp() - mtime) / 86400.0


def classify(
    items: list[FileItem],
    registry: dict[str, RegistryEntry],
    *,
    now: datetime,
    retention_days: float,
) -> list[Classified]:
    """Classify each scanned item against the registry, preserving input order."""
    out: list[Classified] = []
    for item in items:
        row = registry.get(item.strategy)
        stage = row.stage if row is not None else None
        retired_at = row.retired_at if row is not None else None
        age_days: float | None = None

        if row is None:
            # No registry row.
            if item.surface == SURFACE_REPORT:
                # (a) orphaned report — reapable ONLY with a PROVABLE age past retention, the same
                # fail-safe floor the retired path applies (#510 GATE-2: never reap on sight). A
                # report with no provable timestamp (mtime is None) is kept.
                if item.mtime is None:
                    reason = KEEP_ORPHANED_WITHIN_RETENTION
                    reapable_flag = False
                else:
                    age_days = _age_days_from_mtime(item.mtime, now)
                    if age_days >= retention_days:
                        reason = REAP_ORPHANED_REPORT
                        reapable_flag = True
                    else:
                        reason = KEEP_ORPHANED_WITHIN_RETENTION
                        reapable_flag = False
            else:
                # (b) untracked module — never reap from absence.
                reason = KEEP_UNTRACKED_MODULE
                reapable_flag = False
        elif row.stage != RETIRED:
            # (c) non-terminal stage — protected.
            reason = KEEP_PROTECTED_NON_TERMINAL
            reapable_flag = False
        elif row.retired_at is None:
            # (d) retired but no provable age — fail-safe keep.
            reason = KEEP_RETIRED_WITHIN_RETENTION
            reapable_flag = False
        else:
            # (e) retired with a timestamp — reap iff past retention.
            age_days = _age_days(row.retired_at, now)
            if age_days >= retention_days:
                reason = REAP_RETIRED_EXPIRED
                reapable_flag = True
            else:
                reason = KEEP_RETIRED_WITHIN_RETENTION
                reapable_flag = False

        out.append(
            Classified(
                path=item.path,
                strategy=item.strategy,
                surface=item.surface,
                size_bytes=item.size_bytes,
                reason=reason,
                reapable=reapable_flag,
                stage=stage,
                retired_at=retired_at,
                age_days=age_days,
            )
        )
    return out


def reapable(classified: list[Classified]) -> list[Classified]:
    """Only the reapable entries, ranked by reclaimable space DESC, age DESC, path."""
    return sorted(
        (c for c in classified if c.reapable),
        key=lambda c: (-c.size_bytes, -(c.age_days or 0.0), c.path),
    )


def archive_manifest(reap: list[Classified], content_hashes: dict[str, str]) -> str:
    """Canonical, injective, CONTENT-ADDRESSED description of the EXACT file set an ``--archive``
    run would move. Sorted by path; compact JSON. Each row binds ``path``, ``size_bytes``,
    ``surface``, ``reason``, ``strategy`` AND a sha256 of the file's CURRENT bytes (supplied by the
    caller — the pure module does no I/O). This is hashed into the signed challenge
    (build_gc_archive_challenge) so a human signature authorizes exactly these files at exactly
    these contents.

    Replay is neutralized without a persisted nonce (``research gc`` is fleet-wide, so the
    strategy-scoped #329 actor_challenges table cannot key it): (1) archiving MOVES the sources, so
    a re-run re-derives a different (usually empty) manifest and the old signature cannot re-verify;
    (2) binding the content hash means a signature can NEVER be replayed onto a byte-different file
    that merely shares a path/size; (3) even a byte-identical file re-created at the same path only
    re-authorizes moving something ``research gc`` itself just re-classified as reapable — i.e. what
    a fresh signed run would authorize anyway — so replay grants no capability beyond a fresh run.
    """
    rows: list[list[object]] = [
        [c.path, c.size_bytes, c.surface, c.reason, c.strategy, content_hashes.get(c.path, "")]
        for c in sorted(reap, key=lambda c: c.path)
    ]
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def build_gc_archive_challenge(
    *, retention_days: float, archive_dir: str, manifest: str,
) -> str:
    """The exact bytes a human signs to authorize ``research gc --archive`` (#510 GATE-2).

    Reuses the #329 ``algua-human-actor`` namespace + the approvers/allowed_signers trust anchor
    (the same mechanism as ``registry transition --to live``); binds the retention window, the
    resolved archive root, and a sha256 of the exact reap manifest so a signature can neither be
    replayed onto a different file set nor forged by a bare ``--actor human`` string.
    """
    manifest_hash = hashlib.sha256(manifest.encode()).hexdigest()
    return (
        f"{GC_ARCHIVE_NAMESPACE}\ncommand=research gc --archive\n"
        f"retention_days={retention_days}\narchive_dir={archive_dir}\n"
        f"manifest_sha256={manifest_hash}"
    )
