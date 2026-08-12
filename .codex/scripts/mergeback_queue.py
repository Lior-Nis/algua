#!/usr/bin/env python3
"""The durable merge-back queue (factory slice 3): ``data/mergeback-queue.json``.

Two writers touch this ONE JSON file with NO natural ordering between them: the research driver
(``run-research-loop.sh``) enqueues a validated merge-back candidate once its trailer has been
parsed, and the drainer (``drain-mergeback-queue.sh``) updates an item's status after every
attempt. An atomic tmp+fsync+os.replace write (mirroring ``web/backend/push.py``'s
``_write_atomic`` idiom) prevents a TORN write, but not a LOST update between two concurrent
read-modify-writes — that needs a real mutual-exclusion lock around the whole
read-modify-write, not just the write. Every read-modify-write in this module therefore happens
under a dedicated, non-blocking ``flock`` on a SEPARATE lock file (``data/mergeback-queue.lock``),
with a short bounded retry (queue mutations are sub-second, so contention is rare and brief; a
caller that still can't get the lock after the retry budget gets a loud, explicit
:class:`QueueLockTimeout` rather than hanging or silently corrupting state).

This is a standalone script (stdlib only, no ``algua`` import) because it is invoked from BOTH a
bash driver (``run-research-loop.sh``, via ``python3 - <<'PY'`` heredocs that import it from a
known repo-relative path) and a bash drainer (``drain-mergeback-queue.sh``, via this file's own
CLI). Keeping the lock + atomic-write logic in ONE importable module — rather than duplicating it
in two heredocs — is the whole point: a single place to get the concurrency story right.

Queue schema (one JSON object per file):
    {"items": {"<strategy>@<branch>": {
        "strategy": str, "universe": str, "start": "YYYY-MM-DD", "end": "YYYY-MM-DD",
        "branch": str, "eval_context": dict (see validate_eval_context), "enqueued_at": ISO8601,
        "attempts": int,
        "status": "pending" | "in_progress" | "gate_failed" | "terminal_failed"
                 | "promoted_allocated" | "promoted_queued" | "already_done",
        "last_attempt_at": ISO8601 | None, "last_result": dict | None,
        "reserved_at": ISO8601 | None,
    }}}

``eval_context`` (mergeback authoritative intake) is the validated RECIPE the trusted drainer
replays authoritatively post-merge — data-context snapshot ids + the exact sweep grid the scratch
preview swept — never scratch evidence itself (a sandboxed agent could under-report breadth).
It is validated FAIL-CLOSED at enqueue by :func:`validate_eval_context` (an invalid candidate is
rejected with a raised ValueError the producer logs + drops; a malformed queue item is never
written). ``windows``/``holdout_frac``/thresholds/relaxations are deliberately NOT part of the
context: the authoritative run pins the strict-agent defaults, and the producer refuses to enqueue
a candidate whose preview deviated from them.

Status lifecycle: a fresh item starts ``pending``. The drainer reserves an item ATOMICALLY via
:func:`select_and_reserve` — in the SAME locked read-modify-write as selection, the item's status
flips to ``in_progress`` (with a ``reserved_at`` timestamp) so a second, overlapping drainer
invocation (a manual run overlapping the timer, a wedged prior invocation re-fired by systemd)
cannot ALSO select and act on the same item before the first attempt's result lands — closing the
TOCTOU window a read-only select-then-later-record two-step leaves open (``operator.lock`` only
prevents the two ``paper merge-back`` calls from running SIMULTANEOUSLY, not the second one from
starting, blocking, then running the same item after the first already completed it). A reservation
older than ``RESERVATION_STALE_SECONDS`` is treated as eligible again — sized generously above one
full merge-back attempt's realistic worst case, so a crashed/killed drainer can never permanently
wedge a queue item.

The drainer's ``record_attempt`` then classifies every REAL merge-back invocation (never a
lock-contention no-op, never an unparseable/hard failure — see ``record_attempt``'s docstring) from
``in_progress`` into:
  - ``promoted_allocated`` / ``promoted_queued`` / ``already_done`` — terminal SUCCESS, kept in
    the file for audit.
  - ``diff_policy_rejected`` / ``promote_failed`` -> ``terminal_failed`` — terminal FAILURE, never
    retried (the branch content itself is wrong; retrying wastes gate cycles for a guaranteed-
    identical outcome).
  - ``gate_failed`` -> stays ``gate_failed`` (retryable) while ``attempts <
    MAX_MERGEBACK_ATTEMPTS``, else ``terminal_failed`` (cap reached).
Lock contention and hard/unparseable failures never counted as an attempt (``attempts`` /
``last_attempt_at`` / ``last_result`` are left untouched) — see ``record_attempt`` — and RELEASE
the reservation immediately back to the item's pre-reservation status (never left wedged
``in_progress`` waiting out ``RESERVATION_STALE_SECONDS``, since no real attempt happened).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shlex
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

__all__ = [
    "MAX_MERGEBACK_ATTEMPTS",
    "MAX_SWEEP_COMBOS",
    "RESERVATION_STALE_SECONDS",
    "VALID_RANK_BY",
    "QueueLockTimeout",
    "enqueue",
    "read_locked",
    "record_attempt",
    "select_and_reserve",
    "select_eligible",
    "validate_eval_context",
]

# Terminal success statuses a completed cycle can land in (kept in the file for audit).
_SUCCESS_STATUSES = frozenset({"promoted_allocated", "promoted_queued", "already_done"})
# Terminal (non-retryable) failure statuses `paper merge-back` itself reports.
_TERMINAL_FAILURE_STATUSES = frozenset({"diff_policy_rejected", "promote_failed"})
# The one retryable-with-a-cap status.
_RETRYABLE_STATUS = "gate_failed"
# The reservation status `select_and_reserve` atomically stamps onto a selected item.
_RESERVED_STATUS = "in_progress"

MAX_MERGEBACK_ATTEMPTS = 3
_DEFAULT_BACKOFF_MINUTES_PER_ATTEMPT = 10
# Generously above one full merge-back attempt's realistic worst case. The cycle now stacks the
# FULL quality gate (~9 min observed) + a <=200-combo authoritative evidence sweep + a full-period
# backtest + the promote walk-forward/holdout (mergeback authoritative intake), so this rose
# 1800 -> 4200 IN LOCKSTEP with the drainer unit's TimeoutStartSec 1200 -> 3600 (stale > timeout
# ALWAYS: a systemd-killed attempt must already be reclaimable, never wedged). The flock +
# one-item-per-firing keep the 30-min timer safe — a long-running attempt just makes the next
# firing a no-op selection.
RESERVATION_STALE_SECONDS = 4200
_LOCK_RETRIES = 10
_LOCK_RETRY_SLEEP_S = 0.2  # ~2s total bounded retry budget


class QueueLockTimeout(Exception):
    """Could not acquire ``data/mergeback-queue.lock`` within the bounded retry budget.

    Queue mutations are sub-second, so real contention this long is exceedingly rare — a caller
    should log loudly and treat the attempt as a no-op (never lose or duplicate a queue write)."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _acquire_lock(lock_path: Path):  # noqa: ANN202 - returns an open file handle
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+")  # noqa: SIM115 - released by the caller's finally
    for attempt in range(_LOCK_RETRIES):
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except (BlockingIOError, OSError):
            if attempt == _LOCK_RETRIES - 1:
                handle.close()
                raise QueueLockTimeout(
                    f"could not acquire {lock_path} after {_LOCK_RETRIES} tries "
                    f"(~{_LOCK_RETRIES * _LOCK_RETRY_SLEEP_S:.1f}s)"
                ) from None
            time.sleep(_LOCK_RETRY_SLEEP_S)
    raise AssertionError("unreachable")  # pragma: no cover


def _release_lock(handle: Any) -> None:
    try:
        fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


def _write_atomic(path: Path, data: dict) -> None:
    """tmp file (same dir) + fsync + ``os.replace`` + parent-dir fsync — mirrors
    ``web/backend/push.py``'s ``_write_atomic`` idiom, ported to this bash-driven Python side of
    the system (that module is FastAPI-process-local; this one is invoked as a one-shot script from
    bash, so it is reimplemented here rather than imported across that boundary)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read(path: Path) -> dict:
    """The current ``{"items": {...}}`` document, or a fresh empty one if the file is absent.

    A PRESENT-but-corrupt file (invalid JSON / wrong shape) raises loudly — the atomic-replace
    write path means this can only happen from external tampering or a genuine bug, never a torn
    write, so silently treating it as empty would risk losing every in-flight item."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"items": {}}
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("items"), dict):
        raise ValueError(f"{path}: queue file is not the expected {{'items': {{...}}}} shape")
    return data


def read_locked(queue_path: Path, lock_path: Path) -> dict:
    """Read-only snapshot of the queue, taken under the lock (no write)."""
    handle = _acquire_lock(lock_path)
    try:
        return _read(queue_path)
    finally:
        _release_lock(handle)


def _with_queue_lock(
    queue_path: Path, lock_path: Path, mutate: Callable[[dict], tuple[dict, Any]]
) -> Any:
    """Read-modify-write the queue file under the lock. ``mutate(data) -> (new_data, result)``;
    ``new_data`` is always written back (even if unchanged — the write is cheap and keeps the
    read-modify-write atomic as one operation), ``result`` is returned to the caller."""
    handle = _acquire_lock(lock_path)
    try:
        data = _read(queue_path)
        new_data, result = mutate(data)
        _write_atomic(queue_path, new_data)
        return result
    finally:
        _release_lock(handle)


# --- eval_context validation (mergeback authoritative intake) ------------------------------------

# Mirrors the sweep engine's `_MAX_COMBOS` (algua/backtest/sweep.py). Duplicated as a literal, with
# a test cross-checking the two, because this module is deliberately stdlib-only / no `algua`
# import (see the module docstring).
MAX_SWEEP_COMBOS = 200
# Mirrors the sweep engine's `_RANK_KEYS` (same stdlib-only rationale).
VALID_RANK_BY = frozenset({"mean_sharpe", "min_sharpe"})

_ALLOWED_CONTEXT_KEYS = frozenset({
    "demo", "snapshot", "fundamentals_snapshot", "news_snapshot", "delistings",
    "sweep_grid", "rank_by",
})
_SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
# `construction.<key>` names are permitted (they tune the construction policy in a sweep).
_GRID_KEY_RE = re.compile(r"^(construction\.)?[A-Za-z_][A-Za-z0-9_]{0,63}$")
# A STRING grid value must survive the argv round-trip (KEY=v1,v2 -> parse_grid): no comma, no
# whitespace, no shell-hostile chars — and it must not LOOK numeric (parse_grid would coerce it to
# int/float drainer-side, silently changing the grid the preview declared).
_GRID_STR_VALUE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9._-]{0,63}$")


def _validated_grid_values(key: str, values: object) -> list:
    if not isinstance(values, list) or not values:
        raise ValueError(f"sweep_grid[{key!r}] must be a non-empty list")
    out: list = []
    for v in values:
        if isinstance(v, bool):  # bool is an int subclass — never a legitimate grid value
            raise ValueError(f"sweep_grid[{key!r}] contains a bool {v!r}")
        if isinstance(v, float):
            if not math.isfinite(v):
                raise ValueError(f"sweep_grid[{key!r}] contains a non-finite float {v!r}")
            out.append(v)
        elif isinstance(v, int):
            out.append(v)
        elif isinstance(v, str):
            if not _GRID_STR_VALUE_RE.match(v):
                raise ValueError(
                    f"sweep_grid[{key!r}] string value {v!r} fails the transport-safe format "
                    f"check (must not look numeric; no commas/whitespace)")
            out.append(v)
        else:
            raise ValueError(f"sweep_grid[{key!r}] value {v!r} is not int/float/str")
    # Mirror the sweep parser's widening rule (`_coerce_values`): a homogeneous-numeric list with
    # any float widens to all-float, so the canonical grid equals what parse_grid re-derives.
    if any(type(v) is float for v in out) and all(isinstance(v, (int, float)) for v in out):
        out = [float(v) for v in out]
    return out


def validate_eval_context(ctx: object) -> dict:
    """Validate + canonicalize one candidate's ``eval_context`` recipe; raise ValueError if bad.

    FAIL-CLOSED shape validation at enqueue (the drainer-side, strategy-dependent half — grid keys
    vs the strategy module — runs post-merge via ``validate_sweep_grid``, because the module isn't
    on ``main`` at enqueue time): required dict; ``snapshot`` XOR ``demo: true``; optional
    ``fundamentals_snapshot``/``news_snapshot``/``delistings`` snapshot handles; a required
    non-empty ``sweep_grid`` of JSON-canonical parsed values (sorted keys, no NaN/inf, no bools,
    combos <= ``MAX_SWEEP_COMBOS``, ``construction.<key>`` names permitted); ``rank_by``
    allowlisted against ``VALID_RANK_BY`` (default ``mean_sharpe``). Unknown keys — including
    ``windows``/``holdout_frac``: the strict-agent defaults are pinned, never transported — are
    rejected. Returns the canonicalized context dict to persist.
    """
    if not isinstance(ctx, dict):
        raise ValueError("eval_context must be a JSON object")
    unknown = set(ctx) - _ALLOWED_CONTEXT_KEYS
    if unknown:
        raise ValueError(f"eval_context has unknown keys {sorted(unknown)} "
                         f"(windows/holdout_frac/thresholds are never transported)")
    demo = ctx.get("demo", False)
    snapshot = ctx.get("snapshot")
    if demo is not False and demo is not True:
        raise ValueError("eval_context.demo must be the JSON literal true (or absent)")
    if snapshot is not None and (
            not isinstance(snapshot, str) or not _SNAPSHOT_ID_RE.match(snapshot)):
        raise ValueError(f"eval_context.snapshot {snapshot!r} fails the snapshot-id format check")
    if (demo is True) == (snapshot is not None):
        raise ValueError("eval_context requires exactly one of demo: true | snapshot: <id>")
    out: dict = {"demo": True} if demo is True else {"snapshot": snapshot}
    for opt in ("fundamentals_snapshot", "news_snapshot", "delistings"):
        val = ctx.get(opt)
        if val is None:
            continue
        if not isinstance(val, str) or not _SNAPSHOT_ID_RE.match(val):
            raise ValueError(f"eval_context.{opt} {val!r} fails the snapshot-id format check")
        out[opt] = val
    rank_by = ctx.get("rank_by", "mean_sharpe")
    if rank_by not in VALID_RANK_BY:
        raise ValueError(f"eval_context.rank_by {rank_by!r} not in {sorted(VALID_RANK_BY)}")
    out["rank_by"] = rank_by
    grid = ctx.get("sweep_grid")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("eval_context.sweep_grid must be a non-empty object of param: [values]")
    canonical_grid: dict = {}
    combos = 1
    for key in sorted(grid):
        if not isinstance(key, str) or not _GRID_KEY_RE.match(key):
            raise ValueError(f"sweep_grid key {key!r} fails the param-name format check")
        canonical_grid[key] = _validated_grid_values(key, grid[key])
        combos *= len(canonical_grid[key])
        if combos > MAX_SWEEP_COMBOS:
            raise ValueError(
                f"sweep_grid too large: > {MAX_SWEEP_COMBOS} combos (the sweep engine cap)")
    out["sweep_grid"] = canonical_grid
    return out


def _grid_value_str(v: object) -> str:
    # repr for floats (round-trips through parse_grid's float()); plain str otherwise.
    return repr(v) if isinstance(v, float) else str(v)


def _sweep_param_tokens(grid: dict) -> list[str]:
    """The ``KEY=v1,v2`` tokens the drainer passes as repeatable ``--sweep-param`` argv."""
    return [f"{key}={','.join(_grid_value_str(v) for v in values)}"
            for key, values in sorted(grid.items())]


def _shell_context_fields(item: dict) -> dict[str, str]:
    """The eval_context transport vars for ``--format shell`` (empty-string = absent)."""
    ctx = item.get("eval_context") or {}
    grid = ctx.get("sweep_grid") or {}
    return {
        "MERGEBACK_DEMO": "1" if ctx.get("demo") is True else "0",
        "MERGEBACK_SNAPSHOT": ctx.get("snapshot") or "",
        "MERGEBACK_FUNDAMENTALS_SNAPSHOT": ctx.get("fundamentals_snapshot") or "",
        "MERGEBACK_NEWS_SNAPSHOT": ctx.get("news_snapshot") or "",
        "MERGEBACK_DELISTINGS": ctx.get("delistings") or "",
        "MERGEBACK_RANK_BY": ctx.get("rank_by") or "",
        # Space-joined KEY=v1,v2 tokens: every charset is validated at enqueue to be whitespace-
        # free, so bash `read -r -a` splits them back losslessly.
        "MERGEBACK_SWEEP_PARAMS": " ".join(_sweep_param_tokens(grid)),
    }


def enqueue(
    queue_path: Path, lock_path: Path, *, strategy: str, universe: str, start: str, end: str,
    branch: str, eval_context: dict,
) -> dict:
    """Idempotently enqueue one validated merge-back candidate, keyed on ``"<strategy>@<branch>"``.

    If the key ALREADY EXISTS — regardless of its current status, terminal or not — this is a
    no-op that returns the EXISTING item untouched. Reasoning: a research branch is produced
    exactly once per run stamp, so the only realistic way to re-enqueue the same
    (strategy, branch) key is a retried driver step for the SAME cycle (e.g. the digest-append
    retried after a transient failure) with IDENTICAL inputs — never a legitimate reason to reset
    an in-flight item's ``attempts``/``status``, and never a legitimate reason to resurrect a
    TERMINAL item (a resurrected ``terminal_failed`` would silently re-attempt a branch already
    proven bad; a resurrected success would be pure noise). This is a defensive no-op path, not
    the common case.

    ``eval_context`` is REQUIRED and validated fail-closed via :func:`validate_eval_context`
    BEFORE the lock is even taken — an invalid candidate raises ValueError and never touches the
    queue file (the producer logs the warning and drops the candidacy).
    """
    context = validate_eval_context(eval_context)
    key = f"{strategy}@{branch}"

    def _mutate(data: dict) -> tuple[dict, dict]:
        if key in data["items"]:
            return data, {"key": key, "created": False, "item": data["items"][key]}
        item = {
            "strategy": strategy, "universe": universe, "start": start, "end": end,
            "branch": branch, "eval_context": context, "enqueued_at": _now_iso(), "attempts": 0,
            "status": "pending", "last_attempt_at": None, "last_result": None,
        }
        data["items"][key] = item
        return data, {"key": key, "created": True, "item": item}

    return _with_queue_lock(queue_path, lock_path, _mutate)


def _is_eligible(
    item: dict, *, now: datetime, max_attempts: int, backoff_minutes_per_attempt: float
) -> bool:
    status = item.get("status")
    if status == "pending":
        return True
    if status != _RETRYABLE_STATUS:
        return False
    if item.get("attempts", 0) >= max_attempts:
        return False
    last = item.get("last_attempt_at")
    if not isinstance(last, str):
        return True  # no recorded attempt timestamp — nothing to back off from
    try:
        last_dt = datetime.fromisoformat(last)
    except ValueError:
        return True  # unparseable timestamp — fail OPEN to eligible rather than wedge the item
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=UTC)
    backoff = timedelta(minutes=backoff_minutes_per_attempt * item.get("attempts", 0))
    return now - last_dt >= backoff


def select_eligible(
    queue_path: Path, lock_path: Path, *, max_attempts: int = MAX_MERGEBACK_ATTEMPTS,
    backoff_minutes_per_attempt: float = _DEFAULT_BACKOFF_MINUTES_PER_ATTEMPT,
) -> dict:
    """The single eligible item to drain THIS cycle (FIFO by enqueue order), or ``{"key": None,
    "item": None}`` if none is eligible right now.

    Eligible = ``status == "pending"``, OR ``status == "gate_failed"`` with
    ``attempts < max_attempts`` AND the simple linear backoff window (``attempts * per-attempt
    minutes`` since ``last_attempt_at``) has elapsed. Read-only (no write) — a read-only snapshot
    for inspection/tooling.

    NOTE: the drainer itself does NOT use this function — a read-only select here followed by a
    separate, later, out-of-lock :func:`record_attempt` leaves a TOCTOU window open (a second,
    overlapping drainer invocation could select and act on the same item twice before the first
    one's result lands). The drainer uses :func:`select_and_reserve`, which selects AND reserves
    atomically under one locked operation."""
    now = datetime.now(UTC)
    data = read_locked(queue_path, lock_path)
    for key, item in data["items"].items():
        if _is_eligible(
            item, now=now, max_attempts=max_attempts,
            backoff_minutes_per_attempt=backoff_minutes_per_attempt,
        ):
            return {"key": key, "item": dict(item)}
    return {"key": None, "item": None}


def _is_reservable(
    item: dict, *, now: datetime, max_attempts: int, backoff_minutes_per_attempt: float,
    stale_reservation_seconds: float,
) -> bool:
    """Same predicate as :func:`_is_eligible`, PLUS a currently-``in_progress`` item whose
    ``reserved_at`` is older than ``stale_reservation_seconds`` (a crashed/killed drainer's
    reservation is reclaimed rather than wedging the item forever). A non-stale ``in_progress``
    item is never eligible — that's the whole point of the reservation."""
    if item.get("status") == _RESERVED_STATUS:
        reserved_at = item.get("reserved_at")
        if not isinstance(reserved_at, str):
            return True  # no recorded reservation timestamp — fail OPEN rather than wedge
        try:
            reserved_dt = datetime.fromisoformat(reserved_at)
        except ValueError:
            return True  # unparseable timestamp — fail OPEN to eligible rather than wedge
        if reserved_dt.tzinfo is None:
            reserved_dt = reserved_dt.replace(tzinfo=UTC)
        return (now - reserved_dt).total_seconds() >= stale_reservation_seconds
    return _is_eligible(
        item, now=now, max_attempts=max_attempts,
        backoff_minutes_per_attempt=backoff_minutes_per_attempt,
    )


def select_and_reserve(
    queue_path: Path, lock_path: Path, *, max_attempts: int = MAX_MERGEBACK_ATTEMPTS,
    backoff_minutes_per_attempt: float = _DEFAULT_BACKOFF_MINUTES_PER_ATTEMPT,
    stale_reservation_seconds: float = RESERVATION_STALE_SECONDS,
) -> dict:
    """Atomically select AND reserve the single eligible item for this drain cycle, in the SAME
    locked read-modify-write (closes the TOCTOU window a read-only :func:`select_eligible` plus a
    later, out-of-lock :func:`record_attempt` leaves open: ``operator.lock`` prevents two
    ``paper merge-back`` calls from running SIMULTANEOUSLY, but not a second drainer invocation from
    starting, blocking on that lock, then ALSO running the same item after the first one already
    completed it, since the item was never marked in-progress).

    On selecting an item, immediately flips its ``status`` to ``"in_progress"`` and stamps
    ``reserved_at`` = now, before releasing the lock — so a second, concurrently-racing call sees
    the reservation and skips it. Eligibility is the same predicate as :func:`select_eligible`
    (``pending``, or backed-off ``gate_failed`` under the attempt cap) PLUS a STALE
    ``in_progress`` item (``reserved_at`` older than ``stale_reservation_seconds`` — see
    :func:`_is_reservable`); a non-stale ``in_progress`` item is simply not eligible, so a `pending`
    item elsewhere in the file is picked instead (no special-cased "prefer pending" logic needed —
    it falls out of the eligibility predicate itself). FIFO by enqueue order, same as
    :func:`select_eligible`.

    :func:`record_attempt` is the counterpart: it transitions the reserved item FROM
    ``"in_progress"`` to the appropriate terminal/retry status once the (possibly multi-minute)
    merge-back attempt completes.

    Returns ``{"key": None, "item": None}`` if nothing is eligible; otherwise
    ``{"key": ..., "item": <item dict AFTER the reservation was stamped>}``.
    """
    now = datetime.now(UTC)

    def _mutate(data: dict) -> tuple[dict, dict]:
        for key, item in data["items"].items():
            if _is_reservable(
                item, now=now, max_attempts=max_attempts,
                backoff_minutes_per_attempt=backoff_minutes_per_attempt,
                stale_reservation_seconds=stale_reservation_seconds,
            ):
                # Remember the status the item actually had BEFORE this reservation (never
                # "in_progress" itself — re-reserving an already-stale "in_progress" item keeps
                # whatever was stashed the FIRST time it was reserved) so record_attempt can
                # release the lease back to it on a lock_contention/transient_failure outcome
                # (no real attempt happened — the item must be immediately eligible again, not
                # wait out stale_reservation_seconds).
                pre_status = item.get("pre_reservation_status", item.get("status"))
                item["pre_reservation_status"] = pre_status
                item["status"] = _RESERVED_STATUS
                item["reserved_at"] = now.isoformat()
                return data, {"key": key, "item": dict(item)}
        return data, {"key": None, "item": None}

    return _with_queue_lock(queue_path, lock_path, _mutate)


def _last_top_level_object(text: str) -> str | None:
    """Locate the last balanced top-level ``{...}`` in ``text`` via brace-depth counting.

    Scans from the END: finds the final ``}``, then walks backwards tracking brace depth (ignoring
    braces inside JSON string literals) until depth returns to zero, yielding the matching ``{``.
    Returns the substring, or ``None`` if no balanced object is found.

    This is the SAME algorithm as ``algua/cli/operator_cmd.py``'s ``_last_top_level_object``
    (ported here, not imported, because this module is stdlib-only / no ``algua`` import — see the
    module docstring), reused deliberately rather than inventing a second way to solve the same
    "extract the JSON envelope from a subprocess's mixed stdout" problem."""
    end = text.rfind("}")
    if end == -1:
        return None
    depth = 0
    in_string = False
    i = end
    while i >= 0:
        ch = text[i]
        if in_string:
            if ch == '"':
                backslashes = 0
                j = i - 1
                while j >= 0 and text[j] == "\\":
                    backslashes += 1
                    j -= 1
                if backslashes % 2 == 0:
                    in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                return text[i : end + 1]
        i -= 1
    return None


def _parse_result_json(stdout_text: str) -> dict | None:
    """Best-effort parse of ``algua operator lock-run``'s stdout as a JSON object — tolerating
    arbitrary non-JSON noise BEFORE the final JSON line.

    ``lock-run`` is a transparent passthrough (see ``algua/cli/operator_cmd.py``): on a benign
    lock-contention no-op it prints its OWN single-line envelope; on an actual run it prints
    NOTHING of its own — the wrapped ``paper merge-back`` command's single ``emit()`` call is its
    stdout's final line. But ``paper merge-back`` itself first runs the FULL quality gate
    (pytest/ruff/mypy/lint-imports) with INHERITED stdout (see ``algua/cli/paper_cmd.py``'s
    ``_run_quality_gate`` — each check is a plain ``subprocess.run(cmd, cwd=repo_root)``, no
    ``capture_output``), so on every REAL invocation the captured stdout is
    ``<gate tool noise>\\n<final JSON line>``. This is the NORMAL case for a real gate run, not a
    hypothetical — a parser that requires the ENTIRE stdout to be exactly one JSON object would
    return ``None`` on every real merge-back attempt.

    Tries the whole trimmed stdout as JSON first (the common case for a benign lock-contention
    envelope, and for tests that stub a bare JSON payload); if that fails, falls back to the LAST
    balanced top-level ``{...}`` object in the text (see :func:`_last_top_level_object`) — the SAME
    algorithm ``algua/cli/operator_cmd.py``'s ``_parse_driver_payload`` already uses to solve this
    exact problem for the ``paper`` operator job path, reused here rather than reinvented.

    Returns ``None`` (not ``{}``) when nothing parses, so the caller can tell an unparseable/absent
    result from a real-but-atypical envelope."""
    text = stdout_text.strip()
    if not text:
        return None
    for candidate in (text, _last_top_level_object(text)):
        if candidate is None:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def classify_attempt(payload: dict | None, *, attempts: int, max_attempts: int) -> dict:
    """Pure classification of one merge-back attempt's parsed result. Returns
    ``{"action": ..., "status": str | None, "attempts": int}`` where ``action`` is one of:

    - ``"lock_contention"`` — ``operator.lock`` was held; merge-back never ran. NOT an attempt.
    - ``"transient_failure"`` — unparseable stdout, OR an ``ok: false`` envelope (a raised
      fail-closed exception from ``paper merge-back`` itself, or from ``lock-run``'s own setup
      failure) that carries no recognized merge-back ``status``. Treated as an environment
      problem, not a branch-content problem — NOT an attempt; the item is left exactly as it was.
    - ``"terminal"`` — a recognized terminal ``status`` (success or non-retryable failure).
    - ``"retry"`` — ``gate_failed`` with headroom left under ``max_attempts``.
    - ``"exhausted"`` — ``gate_failed`` with no headroom left; terminal (mapped to
      ``terminal_failed`` by the caller).

    ``attempts`` in the returned dict is the NEW attempts count to persist (unchanged for
    ``lock_contention``/``transient_failure``, incremented for every other action — "attempts"
    counts REAL invocations, i.e. every time merge-back actually ran, regardless of outcome).
    """
    if payload is not None and payload.get("ran") is False and payload.get("reason") == "locked":
        return {"action": "lock_contention", "status": None, "attempts": attempts}
    if payload is None or payload.get("ok") is not True or "status" not in payload:
        # Either genuinely unparseable, or an ok:false envelope (a fail-closed exception out of
        # `paper merge-back` — moved remote, stage drift — or a `lock-run`-level setup failure
        # like git_dir_unresolved/lock_unavailable), or an ok:true envelope we don't recognize the
        # shape of. All three are "we cannot confidently classify this as a branch-content
        # outcome" — transient/environmental, never counted as a burned attempt.
        return {"action": "transient_failure", "status": None, "attempts": attempts}
    status = payload["status"]
    new_attempts = attempts + 1
    if status in _SUCCESS_STATUSES or status in _TERMINAL_FAILURE_STATUSES:
        return {"action": "terminal", "status": status, "attempts": new_attempts}
    if status == _RETRYABLE_STATUS:
        if new_attempts < max_attempts:
            return {"action": "retry", "status": status, "attempts": new_attempts}
        return {"action": "exhausted", "status": status, "attempts": new_attempts}
    # An unrecognized status string (a future merge-back status this module doesn't know about
    # yet) — fail closed the same as an unparseable payload rather than silently mis-terminaling.
    return {"action": "transient_failure", "status": None, "attempts": attempts}


def record_attempt(
    queue_path: Path, lock_path: Path, *, key: str, stdout_text: str,
    max_attempts: int = MAX_MERGEBACK_ATTEMPTS,
) -> dict:
    """Update ``key``'s queue entry after one drain attempt, under the lock.

    Reads the CURRENT item fresh (under the lock) so ``attempts``/``status`` are always advanced
    from authoritative state, never from a value the drainer cached before the (multi-minute)
    merge-back attempt ran. This is the counterpart to :func:`select_and_reserve`: it transitions
    the item FROM ``"in_progress"`` to the appropriate terminal/retry status.

    On ``lock_contention``/``transient_failure`` — no real merge-back attempt happened — the item's
    ``attempts``/``last_attempt_at``/``last_result`` are left COMPLETELY untouched. If the item was
    reserved (``status == "in_progress"``), the reservation is RELEASED back to whatever status it
    had before the reservation (not left wedged ``in_progress`` until
    ``RESERVATION_STALE_SECONDS`` elapses) so it is immediately eligible again next cycle. A caller
    that never went through :func:`select_and_reserve` (e.g. a direct call against a plain
    ``pending``/``gate_failed`` item) sees no change at all, exactly as before reservations existed.
    """
    payload = _parse_result_json(stdout_text)

    def _mutate(data: dict) -> tuple[dict, dict]:
        item = data["items"].get(key)
        if item is None:
            return data, {"action": "missing_key", "key": key}
        verdict = classify_attempt(
            payload, attempts=item.get("attempts", 0), max_attempts=max_attempts)
        if verdict["action"] in ("lock_contention", "transient_failure"):
            if item.get("status") == _RESERVED_STATUS:
                item["status"] = item.pop("pre_reservation_status", "pending")
                item.pop("reserved_at", None)
            return data, {"action": verdict["action"], "key": key, "item": dict(item)}
        if verdict["action"] == "exhausted" or verdict["status"] in _TERMINAL_FAILURE_STATUSES:
            # gate_failed-at-cap, diff_policy_rejected, and promote_failed all land on the SAME
            # non-retryable terminal status — only the (kept, unmodified) `last_result` payload
            # distinguishes WHY, for audit.
            new_status = "terminal_failed"
        else:
            new_status = verdict["status"]  # a success status, or retryable gate_failed
        item["status"] = new_status
        item["attempts"] = verdict["attempts"]
        item["last_attempt_at"] = _now_iso()
        item["last_result"] = payload
        item.pop("reserved_at", None)
        item.pop("pre_reservation_status", None)
        return data, {"action": verdict["action"], "key": key, "item": dict(item)}

    return _with_queue_lock(queue_path, lock_path, _mutate)


# --- CLI (invoked by drain-mergeback-queue.sh; the research driver imports this module directly) --


def _shell_line(**fields: str) -> str:
    return "\n".join(f"{k}={shlex.quote(v)}" for k, v in fields.items())


def _cmd_enqueue(args: argparse.Namespace) -> int:
    result = enqueue(
        Path(args.queue), Path(args.lock), strategy=args.strategy, universe=args.universe,
        start=args.start, end=args.end, branch=args.branch,
        eval_context=json.loads(args.eval_context),
    )
    print(json.dumps(result))
    return 0


def _print_selection(result: dict, fmt: str) -> None:
    if fmt != "shell":
        print(json.dumps(result))
        return
    item = result["item"]
    if item is None:
        print(_shell_line(MERGEBACK_SELECTED="0"))
        return
    print(_shell_line(
        MERGEBACK_SELECTED="1", MERGEBACK_KEY=result["key"],
        MERGEBACK_STRATEGY=item["strategy"], MERGEBACK_UNIVERSE=item["universe"],
        MERGEBACK_START=item["start"], MERGEBACK_END=item["end"],
        MERGEBACK_BRANCH=item["branch"],
        **_shell_context_fields(item),
    ))


def _cmd_select(args: argparse.Namespace) -> int:
    result = select_eligible(
        Path(args.queue), Path(args.lock), max_attempts=args.max_attempts,
        backoff_minutes_per_attempt=args.backoff_minutes,
    )
    _print_selection(result, args.format)
    return 0


def _cmd_select_and_reserve(args: argparse.Namespace) -> int:
    result = select_and_reserve(
        Path(args.queue), Path(args.lock), max_attempts=args.max_attempts,
        backoff_minutes_per_attempt=args.backoff_minutes,
        stale_reservation_seconds=args.stale_reservation_seconds,
    )
    _print_selection(result, args.format)
    return 0


def _cmd_record_attempt(args: argparse.Namespace) -> int:
    stdout_text = sys.stdin.read() if args.stdin else args.stdout_text
    result = record_attempt(
        Path(args.queue), Path(args.lock), key=args.key, stdout_text=stdout_text or "",
        max_attempts=args.max_attempts,
    )
    print(json.dumps(result))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enqueue = sub.add_parser("enqueue")
    p_enqueue.add_argument("--queue", required=True)
    p_enqueue.add_argument("--lock", required=True)
    p_enqueue.add_argument("--strategy", required=True)
    p_enqueue.add_argument("--universe", required=True)
    p_enqueue.add_argument("--start", required=True)
    p_enqueue.add_argument("--end", required=True)
    p_enqueue.add_argument("--branch", required=True)
    p_enqueue.add_argument(
        "--eval-context", required=True, dest="eval_context",
        help="the eval_context recipe as a JSON object (validated fail-closed)")
    p_enqueue.set_defaults(fn=_cmd_enqueue)

    p_select = sub.add_parser("select")
    p_select.add_argument("--queue", required=True)
    p_select.add_argument("--lock", required=True)
    p_select.add_argument("--max-attempts", type=int, default=MAX_MERGEBACK_ATTEMPTS)
    p_select.add_argument(
        "--backoff-minutes", type=float, default=_DEFAULT_BACKOFF_MINUTES_PER_ATTEMPT)
    p_select.add_argument("--format", choices=("json", "shell"), default="json")
    p_select.set_defaults(fn=_cmd_select)

    p_select_reserve = sub.add_parser("select-and-reserve")
    p_select_reserve.add_argument("--queue", required=True)
    p_select_reserve.add_argument("--lock", required=True)
    p_select_reserve.add_argument("--max-attempts", type=int, default=MAX_MERGEBACK_ATTEMPTS)
    p_select_reserve.add_argument(
        "--backoff-minutes", type=float, default=_DEFAULT_BACKOFF_MINUTES_PER_ATTEMPT)
    p_select_reserve.add_argument(
        "--stale-reservation-seconds", type=float, default=RESERVATION_STALE_SECONDS)
    p_select_reserve.add_argument("--format", choices=("json", "shell"), default="json")
    p_select_reserve.set_defaults(fn=_cmd_select_and_reserve)

    p_record = sub.add_parser("record-attempt")
    p_record.add_argument("--queue", required=True)
    p_record.add_argument("--lock", required=True)
    p_record.add_argument("--key", required=True)
    p_record.add_argument("--max-attempts", type=int, default=MAX_MERGEBACK_ATTEMPTS)
    p_record.add_argument("--stdout-text", default=None, help="the wrapped command's stdout")
    p_record.add_argument(
        "--stdin", action="store_true", help="read the wrapped command's stdout from stdin instead")
    p_record.set_defaults(fn=_cmd_record_attempt)

    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
