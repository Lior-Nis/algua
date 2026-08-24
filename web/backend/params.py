"""Param hygiene for user-controlled values that reach the algua CLI argv (slice C).

Every path/query parameter that is forwarded to the CLI is validated here BEFORE
it becomes an argv element, and option values are always passed in ``--flag=value``
form (one argv element) so a crafted value can never be parsed as a separate flag.
A violation raises :class:`InvalidParam`, rendered by the app as HTTP 422
``{"ok": false, "error": ..., "code": "invalid_param"}``.

Every validator here is SYNTACTIC (shape only), not semantic. The one deliberate exception:
``sort`` (see ``validate_sort``) is NOT checked against the store's ``METRIC_COLUMNS`` vocabulary
(``algua/registry/store/runs.py``) — that list is long and churns, and ``web/`` deliberately cannot
import ``algua``, so the store stays the single semantic gate for it (a syntactically-valid but
unknown ``sort`` reaches the CLI and is rejected there, surfacing as a 502 ``CliError`` rather than
a 422). A path-typed ``int`` param (e.g. ``run_id`` on ``/api/runs/{run_id}``) is constrained by
Starlette's own route converter (digits only, so a negative id 404s before reaching a handler)
before this module is ever consulted.
"""

from __future__ import annotations

import re
from datetime import datetime

STRATEGY_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
ACTION_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
FAMILY_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
SORT_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
ACTORS = frozenset({"agent", "human", "system"})
LANES = frozenset({"paper", "live"})

# `algua/registry/store/runs.py` `_KINDS` — mirrored here beside ACTORS/LANES (which already
# duplicate algua enums, per precedent) so an unrecognized `kind` is a 422 before a subprocess
# is ever spawned, rather than a CLI-side ValueError surfacing as a 502.
RUN_KINDS = frozenset({"backtest", "walk_forward", "sweep", "sweep_trial", "gate"})

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


def validate_run_kind(kind: str) -> str:
    if kind not in RUN_KINDS:
        raise InvalidParam(
            "kind must be one of: backtest, walk_forward, sweep, sweep_trial, gate"
        )
    return kind


def validate_family(family: str) -> str:
    """Strategy family: same identifier shape as a strategy name (`STRATEGY_NAME_RE`) — syntactic
    only. Which families actually exist is a store-side question, not this module's."""
    if family.startswith("-") or not FAMILY_RE.fullmatch(family):
        raise InvalidParam(
            "family must match ^[A-Za-z0-9._-]{1,64}$ and must not start with '-'"
        )
    return family


def validate_sort(sort: str) -> str:
    """Sort metric name: syntactic guard only. Deliberately does NOT mirror `METRIC_COLUMNS`
    (see module docstring) — the store (`algua/registry/store/runs.py` `list_runs`) stays the one
    semantic gate. This only rejects a value that could not possibly be a metric column name, most
    importantly one starting with '-' (would otherwise become an argv flag)."""
    if sort.startswith("-") or not SORT_RE.fullmatch(sort):
        raise InvalidParam(
            "sort must match ^[A-Za-z0-9._-]{1,64}$ and must not start with '-'"
        )
    return sort


def validate_run_ids(ids: str) -> list[int]:
    """`ids`: a comma-separated list of run ids, each >= 1 (a run id is never zero or negative;
    a negative id would also become a bare LEADING argv token — see `validate_strategy_name`'s
    same no-leading-dash rationale for the first id, which `runs_series` passes positionally).
    De-duplicated and returned SORTED (not merely order-preserving): `algua_cli._cache`/`_locks`
    key on the exact argv tuple, so "1,2" and "2,1" would otherwise be two distinct cache entries
    for the same logical request. Capped at `MAX_SERIES_RUN_IDS` — the same cap `algua runs series`
    itself enforces, checked here so an oversized or malformed list is rejected before a subprocess
    is ever spawned."""
    parsed: list[int] = []
    for raw_id in ids.split(","):
        raw_id = raw_id.strip()
        if not raw_id:
            raise InvalidParam("ids must be a comma-separated list of run ids")
        try:
            run_id = int(raw_id)
        except ValueError:
            raise InvalidParam(f"invalid run id: {raw_id!r}") from None
        if run_id < 1:
            raise InvalidParam(f"invalid run id: {raw_id!r} (must be >= 1)")
        parsed.append(run_id)
    unique_ids = sorted(set(parsed))
    if not unique_ids:
        raise InvalidParam("ids must contain at least one run id")
    if len(unique_ids) > MAX_SERIES_RUN_IDS:
        raise InvalidParam(
            f"too many run ids: got {len(unique_ids)}, max {MAX_SERIES_RUN_IDS} per call"
        )
    return unique_ids
