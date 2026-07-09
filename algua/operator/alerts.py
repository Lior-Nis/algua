"""Operator alert hook: one structured log record plus an optional external command.

Every alert ALWAYS lands as a structured ``operator_alert`` log record. Delivery via an
operator-supplied external command is best-effort and HARDENED: the command is split with
``shlex`` and run with ``shell=False`` (no shell interpolation of the payload), under a bounded
timeout, with the JSON payload on stdin and captured output truncated before logging. Any delivery
failure (timeout, non-zero exit, OSError) is logged and swallowed so an alert never crashes the
operator loop.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable

from algua.observability import get_logger

__all__ = ["emit_alert"]

log = get_logger(__name__)

_ALERT_TIMEOUT_SECONDS = 10.0
_OUTPUT_CAP = 500


def _default_runner(cmd: str, payload: str) -> int:
    """Run the alert command with ``shell=False``, feeding ``payload`` on stdin; return its rc.

    ``cmd`` is split with :func:`shlex.split` and invoked as an argv vector — the payload is NEVER
    interpolated into a shell string, so operational specifics in the alert detail can never be
    interpreted by a shell. A bounded ``timeout`` prevents a hung alert command from wedging the
    operator; captured stdout/stderr are truncated before logging (they may echo the payload).
    """
    argv = shlex.split(cmd)
    if not argv:
        return 1
    result = subprocess.run(  # noqa: S603 - shell=False, fixed argv from operator config
        argv,
        shell=False,
        input=payload,
        text=True,
        capture_output=True,
        timeout=_ALERT_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        log.warning(
            "operator_alert_command_nonzero",
            extra={
                "fields": {
                    "rc": result.returncode,
                    "stdout_head": (result.stdout or "")[:_OUTPUT_CAP],
                    "stderr_head": (result.stderr or "")[:_OUTPUT_CAP],
                }
            },
        )
    return result.returncode


def emit_alert(
    kind: str,
    detail: dict,
    *,
    alert_cmd: str | None = None,
    runner: Callable[[str, str], int] = _default_runner,
) -> bool:
    """Log a structured alert and optionally deliver it via an external command.

    Always emits exactly one ``operator_alert`` ERROR record. When ``alert_cmd`` is set, the JSON
    payload is piped to ``runner``; delivery is best-effort and never propagates (a
    ``TimeoutExpired`` / non-zero rc / ``OSError`` is logged at warning and swallowed). Returns True
    only when the command was invoked and exited 0.
    """
    log.error("operator_alert", extra={"fields": {"kind": kind, **detail}})
    if not alert_cmd:
        return False
    payload = json.dumps({"kind": kind, **detail}, sort_keys=True, default=str)
    try:
        rc = runner(alert_cmd, payload)
    except Exception:
        log.warning(
            "operator_alert_delivery_failed",
            extra={"fields": {"kind": kind}},
            exc_info=True,
        )
        return False
    return rc == 0
