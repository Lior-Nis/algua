"""ISO date/datetime parsing stamped UTC — the stdlib-only parse shared by the CLI lane and the
evaluation lane.

Lives here rather than in ``cli/_common.py`` because both lanes need it and ``_common`` imports
``algua.registry.db``: importing it from ``algua/evaluation/`` would drag ``cli`` — and transitively
``registry`` — into a package whose whole point is being reachable from domain code. Duplicating the
two-line parse instead is what ``primitives`` exists to prevent (spec §4: the stdlib-only leaf is
"what lets true leaves stop hand-duplicating helpers").
"""

from __future__ import annotations

from datetime import UTC, datetime


def utc(date_str: str) -> datetime:
    """Parse an ISO date/datetime string and stamp it UTC."""
    return datetime.fromisoformat(date_str).replace(tzinfo=UTC)


def now_iso() -> str:
    """Current UTC instant as an ISO-8601 string — the shared 'now' for persisted timestamps."""
    return datetime.now(UTC).isoformat()
