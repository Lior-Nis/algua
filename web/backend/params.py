"""Param hygiene for user-controlled values that reach the algua CLI argv (slice C).

Every path/query parameter that is forwarded to the CLI is validated here BEFORE
it becomes an argv element, and option values are always passed in ``--flag=value``
form (one argv element) so a crafted value can never be parsed as a separate flag.
A violation raises :class:`InvalidParam`, rendered by the app as HTTP 422
``{"ok": false, "error": ..., "code": "invalid_param"}``.
"""

from __future__ import annotations

import re
from datetime import datetime

STRATEGY_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
ACTION_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
ACTORS = frozenset({"agent", "human", "system"})
LANES = frozenset({"paper", "live"})

# `algua runs series`'s own cap (algua/cli/runs_cmd.py MAX_SERIES_RUN_IDS) — mirrored here so
# an oversized id list is rejected at the HTTP layer, before a subprocess is ever spawned just
# to have the CLI reject it (the CLI would enforce the same cap, but only after paying for the
# exec).
MAX_SERIES_RUN_IDS = 16


class InvalidParam(Exception):
    """A user-supplied parameter failed validation (rendered as HTTP 422)."""

    def __init__(self, error: str) -> None:
        super().__init__(error)
        self.error = error


def validate_strategy_name(name: str) -> str:
    """Strategy name: ^[A-Za-z0-9._-]{1,64}$ AND must not start with '-'."""
    if name.startswith("-") or not STRATEGY_NAME_RE.fullmatch(name):
        raise InvalidParam(
            "strategy name must match ^[A-Za-z0-9._-]{1,64}$ and must not start with '-'"
        )
    return name


def validate_actor(actor: str) -> str:
    if actor not in ACTORS:
        raise InvalidParam("actor must be one of: agent, human, system")
    return actor


def validate_action(action: str) -> str:
    if not ACTION_RE.fullmatch(action):
        raise InvalidParam("action must match ^[A-Za-z0-9._:-]{1,64}$")
    return action


def validate_since(since: str) -> str:
    try:
        datetime.fromisoformat(since)
    except ValueError:
        raise InvalidParam("since must be an ISO-8601 timestamp") from None
    return since


def validate_lane(lane: str) -> str:
    if lane not in LANES:
        raise InvalidParam("lane must be one of: paper, live")
    return lane


def validate_run_ids(ids: str) -> list[int]:
    """`ids`: a comma-separated list of run ids. De-duplicated (order-preserving) and capped at
    `MAX_SERIES_RUN_IDS` — the same cap `algua runs series` itself enforces, checked here so an
    oversized or malformed list is rejected before a subprocess is ever spawned."""
    parsed: list[int] = []
    for raw_id in ids.split(","):
        raw_id = raw_id.strip()
        if not raw_id:
            raise InvalidParam("ids must be a comma-separated list of run ids")
        try:
            parsed.append(int(raw_id))
        except ValueError:
            raise InvalidParam(f"invalid run id: {raw_id!r}") from None
    unique_ids = list(dict.fromkeys(parsed))
    if not unique_ids:
        raise InvalidParam("ids must contain at least one run id")
    if len(unique_ids) > MAX_SERIES_RUN_IDS:
        raise InvalidParam(
            f"too many run ids: got {len(unique_ids)}, max {MAX_SERIES_RUN_IDS} per call"
        )
    return unique_ids
