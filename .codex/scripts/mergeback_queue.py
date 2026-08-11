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
        "branch": str, "enqueued_at": ISO8601, "attempts": int,
        "status": "pending" | "gate_failed" | "terminal_failed"
                 | "promoted_allocated" | "promoted_queued" | "already_done",
        "last_attempt_at": ISO8601 | None, "last_result": dict | None,
    }}}

Status lifecycle: a fresh item starts ``pending``. The drainer's ``record_attempt`` classifies
every REAL merge-back invocation (never a lock-contention no-op, never an unparseable/hard
failure — see ``record_attempt``'s docstring) into:
  - ``promoted_allocated`` / ``promoted_queued`` / ``already_done`` — terminal SUCCESS, kept in
    the file for audit.
  - ``diff_policy_rejected`` / ``promote_failed`` -> ``terminal_failed`` — terminal FAILURE, never
    retried (the branch content itself is wrong; retrying wastes gate cycles for a guaranteed-
    identical outcome).
  - ``gate_failed`` -> stays ``gate_failed`` (retryable) while ``attempts <
    MAX_MERGEBACK_ATTEMPTS``, else ``terminal_failed`` (cap reached).
Lock contention and hard/unparseable failures leave the item COMPLETELY untouched (not even
``last_attempt_at`` moves) — see ``record_attempt``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
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
    "QueueLockTimeout",
    "enqueue",
    "read_locked",
    "record_attempt",
    "select_eligible",
]

# Terminal success statuses a completed cycle can land in (kept in the file for audit).
_SUCCESS_STATUSES = frozenset({"promoted_allocated", "promoted_queued", "already_done"})
# Terminal (non-retryable) failure statuses `paper merge-back` itself reports.
_TERMINAL_FAILURE_STATUSES = frozenset({"diff_policy_rejected", "promote_failed"})
# The one retryable-with-a-cap status.
_RETRYABLE_STATUS = "gate_failed"

MAX_MERGEBACK_ATTEMPTS = 3
_DEFAULT_BACKOFF_MINUTES_PER_ATTEMPT = 10
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


def enqueue(
    queue_path: Path, lock_path: Path, *, strategy: str, universe: str, start: str, end: str,
    branch: str,
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
    """
    key = f"{strategy}@{branch}"

    def _mutate(data: dict) -> tuple[dict, dict]:
        if key in data["items"]:
            return data, {"key": key, "created": False, "item": data["items"][key]}
        item = {
            "strategy": strategy, "universe": universe, "start": start, "end": end,
            "branch": branch, "enqueued_at": _now_iso(), "attempts": 0,
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
    minutes`` since ``last_attempt_at``) has elapsed. Read-only (no write) — the drainer calls
    :func:`record_attempt` separately, AFTER the merge-back attempt itself, to avoid holding the
    queue lock across a multi-minute quality-gate run."""
    now = datetime.now(UTC)
    data = read_locked(queue_path, lock_path)
    for key, item in data["items"].items():
        if _is_eligible(
            item, now=now, max_attempts=max_attempts,
            backoff_minutes_per_attempt=backoff_minutes_per_attempt,
        ):
            return {"key": key, "item": dict(item)}
    return {"key": None, "item": None}


def _parse_result_json(stdout_text: str) -> dict | None:
    """Best-effort parse of ``algua operator lock-run``'s stdout as ONE JSON object.

    ``lock-run`` is a transparent passthrough (see ``algua/cli/operator_cmd.py``): on a benign
    lock-contention no-op it prints its OWN single-line envelope; on an actual run it prints
    NOTHING of its own — the wrapped ``paper merge-back`` command's single ``emit()`` call is the
    entire stdout. Either way, the common case is "the whole trimmed stdout is one JSON object".
    Returns ``None`` (not ``{}``) when nothing parses, so the caller can tell an unparseable/absent
    result from a real-but-atypical envelope."""
    text = stdout_text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


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
    merge-back attempt ran. On ``lock_contention``/``transient_failure`` the item is left
    COMPLETELY untouched (not even ``last_attempt_at`` moves) and no write happens at all.
    """
    payload = _parse_result_json(stdout_text)

    def _mutate(data: dict) -> tuple[dict, dict]:
        item = data["items"].get(key)
        if item is None:
            return data, {"action": "missing_key", "key": key}
        verdict = classify_attempt(
            payload, attempts=item.get("attempts", 0), max_attempts=max_attempts)
        if verdict["action"] in ("lock_contention", "transient_failure"):
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
        return data, {"action": verdict["action"], "key": key, "item": dict(item)}

    return _with_queue_lock(queue_path, lock_path, _mutate)


# --- CLI (invoked by drain-mergeback-queue.sh; the research driver imports this module directly) --


def _shell_line(**fields: str) -> str:
    return "\n".join(f"{k}={shlex.quote(v)}" for k, v in fields.items())


def _cmd_enqueue(args: argparse.Namespace) -> int:
    result = enqueue(
        Path(args.queue), Path(args.lock), strategy=args.strategy, universe=args.universe,
        start=args.start, end=args.end, branch=args.branch,
    )
    print(json.dumps(result))
    return 0


def _cmd_select(args: argparse.Namespace) -> int:
    result = select_eligible(
        Path(args.queue), Path(args.lock), max_attempts=args.max_attempts,
        backoff_minutes_per_attempt=args.backoff_minutes,
    )
    if args.format == "shell":
        item = result["item"]
        if item is None:
            print(_shell_line(MERGEBACK_SELECTED="0"))
        else:
            print(_shell_line(
                MERGEBACK_SELECTED="1", MERGEBACK_KEY=result["key"],
                MERGEBACK_STRATEGY=item["strategy"], MERGEBACK_UNIVERSE=item["universe"],
                MERGEBACK_START=item["start"], MERGEBACK_END=item["end"],
                MERGEBACK_BRANCH=item["branch"],
            ))
    else:
        print(json.dumps(result))
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
    p_enqueue.set_defaults(fn=_cmd_enqueue)

    p_select = sub.add_parser("select")
    p_select.add_argument("--queue", required=True)
    p_select.add_argument("--lock", required=True)
    p_select.add_argument("--max-attempts", type=int, default=MAX_MERGEBACK_ATTEMPTS)
    p_select.add_argument(
        "--backoff-minutes", type=float, default=_DEFAULT_BACKOFF_MINUTES_PER_ATTEMPT)
    p_select.add_argument("--format", choices=("json", "shell"), default="json")
    p_select.set_defaults(fn=_cmd_select)

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
