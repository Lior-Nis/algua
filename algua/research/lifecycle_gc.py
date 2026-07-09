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
KEEP_PROTECTED_NON_TERMINAL = "protected_non_terminal"
KEEP_UNTRACKED_MODULE = "untracked_module"


@dataclass(frozen=True)
class FileItem:
    """A scanned on-disk artifact."""

    path: str          # path as scanned, relative to cwd — reporting + archive move-source
    strategy: str      # strategy name from the filename stem
    surface: str       # SURFACE_MODULE | SURFACE_REPORT
    size_bytes: int


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
                # (a) orphaned report — reapable.
                reason = REAP_ORPHANED_REPORT
                reapable_flag = True
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
