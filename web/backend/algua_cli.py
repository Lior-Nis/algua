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
import os
import signal
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
# force_refresh's "just fetched" guard: an entry younger than this is served as fresh even
# under force_refresh. A poller queued behind a user request on the same per-key lock would
# otherwise re-run the CLI back-to-back the moment the lock is released.
_FORCE_REFRESH_SKEW_S = 2.0


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


def _hit(entry: dict[str, Any], *, stale: bool) -> dict[str, Any]:
    """Serve a cache entry. A stale hit carries the LOUD metadata the UI banners read."""
    served: dict[str, Any] = {
        "ok": True,
        "data": entry["value"],
        "fetched_at": entry["fetched_at"],
        "stale": stale,
    }
    if stale:
        served.update(
            cache_age_s=time.monotonic() - entry["fetched_monotonic"],
            last_success_at=entry["fetched_at"],
            last_error_code=entry["last_error_code"],
            last_error_at=entry["last_error_at"],
        )
    return served


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
        # "code" may be absent on a malformed/regressed error envelope — ok:false + error is
        # already unambiguous (fleet health's alerting payload has NO top-level "error" key),
        # and letting it through as data would hand the frontend a shape it can't render.
        if payload.get("ok") is False and "error" in payload:
            raise CliError(str(payload.get("code") or "cli_error"), str(payload["error"]))
        return payload
    if isinstance(payload, list):
        return {"data": payload}
    raise CliError("bad_output", f"unexpected JSON payload of type {type(payload).__name__}")


def _signal_group(proc: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    """Signal the subprocess's WHOLE process group (start_new_session makes it the leader) —
    signalling only the direct child would orphan the uv-run fallback's real CLI process."""
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


async def _reap_group(proc: asyncio.subprocess.Process) -> None:
    _signal_group(proc, signal.SIGTERM)
    try:
        await asyncio.wait_for(proc.wait(), timeout=_KILL_GRACE_S)
    except TimeoutError:
        _signal_group(proc, signal.SIGKILL)
        await proc.wait()


async def _execute(args: tuple[str, ...], timeout_s: float) -> Any:
    """Run the CLI once and return the classified payload. Raises CliError."""
    cmd = _cli_command(args)
    async with _semaphore:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=REPO,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            # A spawn failure (missing/non-executable binary, uv absent) must be a CliError,
            # not an uncaught 500 — stale-serving only catches CliError.
            raise CliError("cli_spawn_failed", f"{cmd[0]}: {exc}") from None
        try:
            stdout, _stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            await _reap_group(proc)
            raise CliError(
                "cli_timeout", f"algua {' '.join(args)} exceeded {timeout_s}s wall timeout"
            ) from None
        except asyncio.CancelledError:
            # Request/server cancellation mid-communicate would otherwise leak the subprocess.
            await _reap_group(proc)
            raise
    text = stdout.decode("utf-8", errors="replace")
    # Parse REGARDLESS of exit code: fleet health exits 1 while alerting by design,
    # and typer usage errors (exit 2) still emit the JSON error envelope.
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        raise CliError("bad_output", text[:_BAD_OUTPUT_PREVIEW_CHARS]) from None
    return _classify(payload)


async def run_cli(
    *args: str, ttl_s: float, timeout_s: float = 60.0, force_refresh: bool = False
) -> dict[str, Any]:
    """Run an algua CLI command with caching, singleflight, and stale-serving.

    Returns ``{"ok": True, "data": ..., "fetched_at": iso, "stale": bool, ...}``.
    On failure with a cached value, serves the cache with LOUD stale metadata;
    on failure without one, raises :class:`CliError`. An entry whose last refresh
    failed keeps serving stale — including on a TTL-fresh hit — until a refresh
    succeeds, so staleness never flickers off between two failing refreshes.

    ``force_refresh=True`` (the background poller's mode) skips the TTL freshness
    check inside the same per-key lock — the CLI re-runs even on a fresh cache,
    updating the same entry on success and stale-serving on CliError like any
    other call. One guard remains: an entry fetched less than
    ``_FORCE_REFRESH_SKEW_S`` (2s) ago — i.e. a refresh that just completed while
    we waited on the lock — is served as fresh, so a poller queued behind a user
    request never runs a duplicate refresh back-to-back.
    """
    key = tuple(args)
    lock = _locks.setdefault(key, asyncio.Lock())
    # Singleflight: a concurrent second caller awaits the first fetch, then hits
    # the fresh cache instead of spawning a duplicate subprocess.
    async with lock:
        entry = _cache.get(key)
        if entry is not None:
            age_s = time.monotonic() - entry["fetched_monotonic"]
            freshness_window_s = _FORCE_REFRESH_SKEW_S if force_refresh else ttl_s
            if age_s < freshness_window_s:
                # A refresh that FAILED since this value was fetched leaves the entry's
                # error marker set. Serving it as `stale: False` would make the UI's
                # stale banner (and the poller's R1 fresh-only diff gate) flap: loud for
                # the failing request, silent for every TTL-fresh one behind it. The
                # marker is cleared the moment a refresh succeeds, so staleness sticks
                # exactly as long as the CLI is actually failing.
                return _hit(entry, stale=entry["last_error_code"] is not None)
        try:
            payload = await _execute(key, timeout_s)
        except CliError as exc:
            if entry is None:
                raise
            entry["last_error_code"] = exc.code
            entry["last_error_at"] = _utc_now_iso()
            return _hit(entry, stale=True)
        fetched_at = _utc_now_iso()
        _cache[key] = {
            "value": payload,
            "fetched_monotonic": time.monotonic(),
            "fetched_at": fetched_at,
            "last_error_code": None,
            "last_error_at": None,
        }
        return {"ok": True, "data": payload, "fetched_at": fetched_at, "stale": False}
