"""The operator session decision tree (#486, stage 6d task 3): gate -> (run -> record).

Pulled out of :mod:`algua.cli.operator_cmd` so the decision tree is pure domain logic. The CLI
output surface (``emit`` — ``algua.cli.app.emit`` — and the JSON success-envelope stamp ``ok`` —
``algua.cli._common.ok``) and the driver-subprocess seam (``run_driver`` —
``operator_cmd._run_driver``) are all INJECTED rather than imported: ``algua.operator`` is
contract-walled off ``algua.cli`` (``operator lane stays off the cli layer``, pyproject.toml) —
importing any of them here would be a gate failure, not a style choice. ``emit_alert``
(``algua.operator.alerts``) is already a domain module, so it travels as an ordinary import.

``typer.Exit`` stays a CLI concern (Decision 3): this module returns a plain ``bool`` instead of
raising — ``True`` means the CLI wrapper should exit 0, ``False`` means it must
``raise typer.Exit(1)``. All five original ``raise typer.Exit(1)`` call sites carried the identical
exit code, so one boolean is enough to preserve the exact exit-code contract. Two of the five
originally used ``... from None`` to suppress exception chaining onto an already-emitted JSON
envelope; converting each into a plain ``return False`` from inside its ``except`` block achieves
the same suppression for free — the function returns normally, so no exception context is live by
the time the CLI wrapper (``algua/cli/operator_cmd.py::run``) does its own
``raise typer.Exit(1) from None``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from datetime import datetime
from typing import Any

from algua.calendar.factory import get_calendar
from algua.config.settings import get_settings
from algua.operator.alerts import emit_alert
from algua.operator.driver_payload import classify_failure, parse_driver_payload
from algua.operator.jobs import OperatorJob
from algua.operator.schedule import SessionMarker, session_gate

__all__ = ["run_session"]

_STDOUT_HEAD_CAP = 500
# Hard wall-clock cap on a single driver subprocess, as a multiple of the job's stuck-lock grace
# (`expected_duration_seconds`). A driver hung on a broker/network stall is KILLED at this cap so it
# can never hold `operator.lock` indefinitely and silently stop the fleet from ever trading again.
# The kill leaves the session marker unwritten, so the next timer fire re-attempts (run-all
# reconciles-before-trading, so a retry never blind-double-trades). systemd `TimeoutStartSec` is a
# further backstop set ABOVE this app-level cap.
_DRIVER_TIMEOUT_FACTOR = 2.0


def run_session(
    job: str,
    op_job: OperatorJob,
    command: list[str],
    now_dt: datetime,
    alert_cmd: str | None,
    host: str,
    pid: int,
    *,
    emit: Callable[[Any], None],
    ok: Callable[[dict], dict],
    run_driver: Callable[[list[str], float], subprocess.CompletedProcess],
) -> bool:
    """Gate → (run → record), all inside the held run lock.

    Returns ``True`` if the CLI wrapper should exit 0, ``False`` if it must
    ``raise typer.Exit(1)`` (see the module docstring — all five failure branches share one exit
    code, so this single boolean carries the full contract).
    """
    marker = SessionMarker(get_settings().db_path.parent)
    decision = session_gate(job, now_dt, get_calendar(), marker)
    sess_iso = decision.session.isoformat() if decision.session else None

    if decision.reason == "calendar_out_of_bounds":
        emit_alert(
            "calendar_out_of_bounds", {"job": job, "now": now_dt.isoformat()}, alert_cmd=alert_cmd
        )
        emit(
            {
                "ok": False,
                "job": job,
                "ran": False,
                "reason": "calendar_out_of_bounds",
                "alerted": True,
            }
        )
        return False

    if decision.reason == "marker_corrupt":
        emit_alert("marker_corrupt", {"job": job, "session": sess_iso}, alert_cmd=alert_cmd)
        emit(
            {
                "ok": False,
                "job": job,
                "ran": False,
                "reason": "marker_corrupt",
                "session": sess_iso,
                "alerted": True,
            }
        )
        return False

    if not decision.due:
        emit(ok({"job": job, "ran": False, "reason": decision.reason, "session": sess_iso}))
        return True

    assert decision.session is not None

    if decision.skipped_sessions > 0:
        last_recorded = marker.last_session(job)  # not corrupt on the due path
        emit_alert(
            "session_gap",
            {
                "job": job,
                "last_recorded": last_recorded.isoformat() if last_recorded else None,
                "target": sess_iso,
                "skipped_sessions": decision.skipped_sessions,
            },
            alert_cmd=alert_cmd,
        )

    driver_timeout = op_job.expected_duration_seconds * _DRIVER_TIMEOUT_FACTOR
    try:
        proc = run_driver(command, driver_timeout)
    except subprocess.TimeoutExpired:
        # A hung driver (broker/network stall) would otherwise hold operator.lock until a human
        # intervenes and silently stop the fleet from trading. It is KILLED at the wall-clock cap;
        # the marker is left unwritten so the next fire re-attempts (run-all reconciles-before-
        # trading, so a retry never blind-double-trades), and the timeout is alerted.
        emit_alert(
            "driver_timeout",
            {"job": job, "session": sess_iso, "timeout_seconds": driver_timeout},
            alert_cmd=alert_cmd,
        )
        emit(
            {
                "ok": False,
                "job": job,
                "ran": True,
                "recorded": False,
                "reason": "driver_timeout",
                "session": sess_iso,
                "timeout_seconds": driver_timeout,
                "alerted": True,
            }
        )
        return False
    except OSError as exc:
        # The driver could not even be SPAWNED (binary not on PATH, permission denied, …) — this is
        # not a driver failure, it is an operator-config failure. Without this catch it would
        # propagate past the run lock's `finally` (releasing the lock correctly) straight to the
        # generic `@json_errors` catch-all, which renders a JSON error envelope but — critically —
        # never calls `emit_alert`: the operator would then fail EVERY fire, forever, with zero
        # paging (GATE-2 finding, #486). Alert explicitly and leave the marker unwritten.
        emit_alert(
            "driver_spawn_failed",
            {"job": job, "session": sess_iso, "error": str(exc)},
            alert_cmd=alert_cmd,
        )
        emit(
            {
                "ok": False,
                "job": job,
                "ran": False,
                "recorded": False,
                "reason": "driver_spawn_failed",
                "session": sess_iso,
                "error": str(exc),
                "alerted": True,
            }
        )
        return False
    payload = parse_driver_payload(proc.stdout)
    rc = proc.returncode
    stdout_head = (proc.stdout or "")[:_STDOUT_HEAD_CAP]

    if rc == 0:
        # rc==0 does NOT by itself prove the session completed. Check the anomaly cases FIRST — an
        # unparseable envelope is a completion we cannot confirm, and a `deferred` cycle chose not
        # to trade — before applying the job's positive-completion predicate (which, for `paper`,
        # would otherwise treat a bare rc0 with no `deferred` flag as complete, §D4).
        if payload is None:
            # The drivers always emit JSON; unparseable stdout is an anomaly. Refuse to assert a
            # completion we can't verify: do NOT record, alert, and let the next fire retry.
            emit_alert(
                "completion_unconfirmed",
                {"job": job, "rc": rc, "stdout_head": stdout_head},
                alert_cmd=alert_cmd,
            )
            emit(
                ok(
                    {
                        "job": job,
                        "ran": True,
                        "recorded": False,
                        "reason": "completion_unconfirmed",
                        "session": sess_iso,
                        "rc": 0,
                    }
                )
            )
            return True
        if not op_job.is_completed(rc, payload):
            if payload.get("deferred") is True:
                # A benign deferral (the driver deliberately chose NOT to trade this cycle — a
                # transient reconcile condition): NOT completed, so the marker is left unwritten and
                # the next fire retries. Expected operation, not a failure — no alert.
                emit(
                    ok(
                        {
                            "job": job,
                            "ran": True,
                            "recorded": False,
                            "reason": "deferred",
                            "session": sess_iso,
                            "rc": 0,
                        }
                    )
                )
                return True
            # rc==0 but the driver neither asserted success (`ok:true`) NOR deferred — e.g.
            # `ok:false`, or an `ok`-less envelope at rc0. We cannot confirm the session completed,
            # so — exactly like the unparseable case above — refuse to record, ALERT, and let the
            # next fire retry. Without this, a broken-but-rc0 driver would be silently misfiled as a
            # benign deferral and retried FOREVER with zero paging (GATE-2 finding, #486).
            emit_alert(
                "completion_unconfirmed",
                {"job": job, "rc": rc, "stdout_head": stdout_head},
                alert_cmd=alert_cmd,
            )
            emit(
                ok(
                    {
                        "job": job,
                        "ran": True,
                        "recorded": False,
                        "reason": "completion_unconfirmed",
                        "session": sess_iso,
                        "rc": 0,
                    }
                )
            )
            return True
        marker.record(job, decision.session, command=list(command), rc=rc, host=host, pid=pid)
        emit(ok({"job": job, "ran": True, "recorded": True, "session": sess_iso, "rc": rc}))
        return True

    # rc != 0 — a failure. The alert ALWAYS fires and ALWAYS carries rc + stdout_head;
    # classification is a best-effort label only. Marker NOT recorded — the next fire re-attempts.
    kind = classify_failure(payload)
    emit_alert(
        kind,
        {"job": job, "session": sess_iso, "rc": rc, "stdout_head": stdout_head},
        alert_cmd=alert_cmd,
    )
    emit(
        {
            "ok": False,
            "job": job,
            "ran": True,
            "recorded": False,
            "session": sess_iso,
            "rc": rc,
            "alerted": True,
            "alert_kind": kind,
        }
    )
    return False
