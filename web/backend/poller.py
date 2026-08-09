"""Background cache prewarmer (slice D): keep the hot endpoints' cache warm.

No diffing and no push here — that's slice E. The loop's only job is that a
user opening the monitor after hours of idleness hits a warm cache instead of
waiting on a cold CLI run. Each pass force-refreshes the prewarm targets
through the SAME :func:`backend.algua_cli.run_cli` seam the endpoints use
(same cache entries, same TTLs), so endpoint behavior never changes — the
poller just keeps the entries the endpoints read recently written.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os

from backend.algua_cli import CliError, run_cli

logger = logging.getLogger(__name__)

_DEFAULT_POLL_SECONDS = 600.0

# (argv, ttl_s) — ttl_s mirrors the SERVING endpoint's TTL for the same argv
# (fleet health -> /api/fleet 10s; registry list -> /api/strategies 60s) so a
# poller-written cache entry is exactly what the endpoint would have written.
PREWARM_TARGETS: tuple[tuple[tuple[str, ...], float], ...] = (
    (("fleet", "health"), 10.0),
    (("registry", "list"), 60.0),
)


def poll_seconds() -> float:
    """ALGUA_WEB_POLL_SECONDS, parsed defensively: a bad value -> default + warning."""
    raw = os.environ.get("ALGUA_WEB_POLL_SECONDS")
    if raw is None:
        return _DEFAULT_POLL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value <= 0.0:
        logger.warning(
            "ALGUA_WEB_POLL_SECONDS=%r is not a positive number; using default %ss",
            raw,
            _DEFAULT_POLL_SECONDS,
        )
        return _DEFAULT_POLL_SECONDS
    return value


async def poll_loop() -> None:
    """Prewarm the targets forever, sleeping POLL_SECONDS between passes.

    A failed target NEVER kills the loop (CliError -> warning; any other
    exception -> log.exception) and never stops the remaining targets in the
    pass. Cancellation (server shutdown) propagates cleanly.
    """
    interval_s = poll_seconds()
    while True:
        for target_argv, ttl_s in PREWARM_TARGETS:
            try:
                # Return value ignored on purpose: success warms the cache, and a
                # CliError-with-cache stale-serve already updated the entry's
                # last_error_* metadata for the endpoints to surface.
                await run_cli(*target_argv, ttl_s=ttl_s, force_refresh=True)
            except asyncio.CancelledError:
                raise
            except CliError as exc:
                logger.warning("poller: prewarm of %s failed: %s", " ".join(target_argv), exc)
            except Exception:
                logger.exception(
                    "poller: unexpected error prewarming %s", " ".join(target_argv)
                )
        await asyncio.sleep(interval_s)
