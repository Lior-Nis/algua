"""The seam between the web backend and the algua CLI.

Every read the monitor serves goes through :func:`run_cli`: an async subprocess
runner with a per-args-tuple TTL cache, singleflight deduplication, a global
concurrency cap, and loud stale-serving when the CLI fails but a cached value
exists. The CLI's JSON-on-stdout contract is classified by envelope SHAPE, not
by the ``ok`` flag — ``fleet health`` exits 1 with ``ok: false`` while alerting
BY DESIGN, and that payload is data, not an error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# web/backend/algua_cli.py -> web/backend -> web -> repo root (contains .venv and algua/).
REPO = Path(__file__).resolve().parents[2]

_MAX_CONCURRENT_SUBPROCESSES = 2
_KILL_GRACE_S = 2.0
_BAD_OUTPUT_PREVIEW_CHARS = 200


class CliError(Exception):
    """A CLI invocation failed: error envelope, timeout, or unparseable output."""

    def __init__(self, code: str, error: str) -> None:
        super().__init__(f"{code}: {error}")
        self.code = code
        self.error = error


# Cache entry per args tuple: {"value", "fetched_monotonic", "fetched_at",
# "last_error_code", "last_error_at"}.
_cache: dict[tuple[str, ...], dict[str, Any]] = {}
_locks: dict[tuple[str, ...], asyncio.Lock] = {}
_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SUBPROCESSES)
_exec_path_logged = False


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _cli_command(args: tuple[str, ...]) -> list[str]:
    """Resolve the CLI executable: direct venv exec (prod) or uv run (dev fallback)."""
    global _exec_path_logged
    direct = REPO / ".venv" / "bin" / "algua"
    if direct.exists():
        cmd = [str(direct), *args]
        exec_path = f"direct exec ({direct})"
    else:
        cmd = ["uv", "run", "--no-sync", "algua", *args]
        exec_path = "dev fallback (uv run --no-sync algua)"
    if not _exec_path_logged:
        logger.info("algua CLI exec path: %s", exec_path)
        _exec_path_logged = True
    return cmd


def _classify(payload: Any) -> Any:
    """Classify parsed CLI stdout by envelope SHAPE.

    A dict with ``ok == False`` AND both ``error`` and ``code`` keys is the CLI
    error envelope -> CliError. ANY other dict (including fleet health's
    ``ok: false`` alerting payload) is success data. A bare list is wrapped as
    ``{"data": [...]}``. Anything else is not a known CLI payload -> bad_output.
    """
    if isinstance(payload, dict):
        if payload.get("ok") is False and "error" in payload and "code" in payload:
            raise CliError(str(payload["code"]), str(payload["error"]))
        return payload
    if isinstance(payload, list):
        return {"data": payload}
    raise CliError("bad_output", f"unexpected JSON payload of type {type(payload).__name__}")


async def _execute(args: tuple[str, ...], timeout_s: float) -> Any:
    """Run the CLI once and return the classified payload. Raises CliError."""
    cmd = _cli_command(args)
    async with _semaphore:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=REPO,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, _stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=_KILL_GRACE_S)
            except TimeoutError:
                proc.kill()
                await proc.wait()
            raise CliError(
                "cli_timeout", f"algua {' '.join(args)} exceeded {timeout_s}s wall timeout"
            ) from None
    text = stdout.decode("utf-8", errors="replace")
    # Parse REGARDLESS of exit code: fleet health exits 1 while alerting by design,
    # and typer usage errors (exit 2) still emit the JSON error envelope.
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        raise CliError("bad_output", text[:_BAD_OUTPUT_PREVIEW_CHARS]) from None
    return _classify(payload)


async def run_cli(*args: str, ttl_s: float, timeout_s: float = 60.0) -> dict[str, Any]:
    """Run an algua CLI command with caching, singleflight, and stale-serving.

    Returns ``{"ok": True, "data": ..., "fetched_at": iso, "stale": bool, ...}``.
    On failure with a cached value, serves the cache with LOUD stale metadata;
    on failure without one, raises :class:`CliError`.
    """
    key = tuple(args)
    lock = _locks.setdefault(key, asyncio.Lock())
    # Singleflight: a concurrent second caller awaits the first fetch, then hits
    # the fresh cache instead of spawning a duplicate subprocess.
    async with lock:
        entry = _cache.get(key)
        if entry is not None and (time.monotonic() - entry["fetched_monotonic"]) < ttl_s:
            return {
                "ok": True,
                "data": entry["value"],
                "fetched_at": entry["fetched_at"],
                "stale": False,
            }
        try:
            payload = await _execute(key, timeout_s)
        except CliError as exc:
            if entry is None:
                raise
            entry["last_error_code"] = exc.code
            entry["last_error_at"] = _utc_now_iso()
            return {
                "ok": True,
                "data": entry["value"],
                "fetched_at": entry["fetched_at"],
                "stale": True,
                "cache_age_s": time.monotonic() - entry["fetched_monotonic"],
                "last_success_at": entry["fetched_at"],
                "last_error_code": entry["last_error_code"],
                "last_error_at": entry["last_error_at"],
            }
        fetched_at = _utc_now_iso()
        _cache[key] = {
            "value": payload,
            "fetched_monotonic": time.monotonic(),
            "fetched_at": fetched_at,
            "last_error_code": None,
            "last_error_at": None,
        }
        return {"ok": True, "data": payload, "fetched_at": fetched_at, "stale": False}
