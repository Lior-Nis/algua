"""Autonomous-operator CLI wrapper: the single-shot driver the systemd timers fire (#486).

The timers fire this command on a wall-clock cadence; the XNYS *calendar gate* + per-session
*idempotency marker* + a git-dir-anchored *run lock* (all from :mod:`algua.operator.schedule`)
decide whether the wrapped driver command actually runs. A weekend/holiday firing, a re-fire of a
session, or an overlap with a still-running sibling all no-op cleanly.

The ``--job`` key is resolved against the :data:`~algua.operator.jobs.OPERATOR_JOBS` manifest and
the trailing ``-- <command…>`` is bound to that job's FULL canonical argv (exact-arity) BEFORE the
gate — an unknown job or a command that does not structurally match fails closed with an alert, so a
mistyped / rogue / extended command can never mark a session done.

The session marker is recorded ONLY on a run the job's completion predicate accepts (NOT bare
``rc==0``): a non-zero run — OR an rc-0 run the driver ``deferred`` — leaves the marker untouched so
the next timer fire re-attempts. Failures / anomalies are routed to the operator alert hook and the
wrapper exits non-zero (a failed unit is visible to systemd); a benign no-op exits 0.

This module MUST NOT import any sibling ``algua.cli.*_cmd`` module (import-linter independence
contract): it reaches only the shared CLI infra (``app``, ``_common``, ``errors``) and the pure
``algua.operator`` core.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import typer

from algua.calendar.market_calendar import MarketCalendar
from algua.cli._common import ok
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.observability import configure_logging, correlation_context
from algua.operator.alerts import emit_alert
from algua.operator.jobs import OPERATOR_JOBS, CommandMismatch, OperatorJob
from algua.operator.schedule import (
    OperatorLockHeld,
    SessionMarker,
    operator_run_lock,
    session_gate,
)

operator_app = typer.Typer(
    help="Autonomous operator clock: session-gated single-shot driver runs",
    no_args_is_help=True,
)
app.add_typer(operator_app, name="operator")

_STDOUT_HEAD_CAP = 500


def _run_driver(command: list[str]) -> subprocess.CompletedProcess:
    """Subprocess seam (monkeypatched in tests): run the driver, capturing stdout/stderr."""
    return subprocess.run(command, capture_output=True, text=True)  # noqa: S603


def _resolve_git_dir() -> Path:
    """The per-worktree git dir (``git rev-parse --absolute-git-dir``) that anchors the run lock.

    A module-level seam so tests can point the ``operator.lock`` at an isolated tmp dir. Mirrors the
    #485 merge-back lock: the run lock protects the shared per-worktree mutation surface (the paper
    account + the git checkout), which is shared per WORKING TREE, not per db.
    """
    here = Path(__file__).resolve().parent
    out = subprocess.run(  # noqa: S603,S607 — fixed argv, no shell
        ["git", "rev-parse", "--absolute-git-dir"],
        cwd=here,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def _last_top_level_object(text: str) -> str | None:
    """Locate the last balanced top-level ``{...}`` in ``text`` via brace-depth counting.

    Scans from the END: finds the final ``}``, then walks backwards tracking brace depth (ignoring
    braces inside JSON string literals) until depth returns to zero, yielding the matching ``{``.
    Returns the substring, or ``None`` if no balanced object is found.
    """
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


def _parse_driver_payload(stdout: str) -> dict | None:
    """Best-effort recover the driver's JSON envelope from its stdout, or ``None`` if none parses.

    A driver's single ``emit()`` call round-trips as one ``json.dumps(..., indent=2)`` document, so
    the FULL stdout parses cleanly in the common case. If the driver interleaved extra output, fall
    back to the last balanced top-level ``{...}``. Returns ``None`` (NOT ``{}``) when nothing parses
    to a dict, so the caller can tell "the driver did not emit parseable JSON" (a completion it
    cannot confirm) from "the driver emitted a valid but non-deferred envelope".
    """
    text = stdout.strip()
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


def _classify_failure(payload: dict | None) -> str:
    """Best-effort alert label for an rc!=0 outcome (never load-bearing — the alert fires and
    carries rc+stdout_head regardless). ``halted`` → global_halt, ``ok:false`` → breach, else
    job_failed (also the parse-failure fallback)."""
    if payload is not None and payload.get("halted"):
        return "global_halt"
    if payload is not None and payload.get("ok") is False:
        return "breach"
    return "job_failed"


@operator_app.command("run")
@json_errors
def run(
    job: str = typer.Option(..., "--job", help="job key resolved against OPERATOR_JOBS"),
    command: list[str] = typer.Argument(
        None, help="driver command to run when the session is due (must match the job's argv)"
    ),
    now: str | None = typer.Option(
        None,
        "--now",
        help="ISO instant override (default utcnow); for manual bring-up/tests",
    ),
) -> None:
    """Session-gate ``job`` and run its wrapped driver command exactly once per XNYS session.

    Emits a JSON envelope describing the decision. On a due, positively-completed run the session
    marker is recorded; on a due, failed run the marker is NOT recorded (a re-fire re-attempts), the
    failure is classified and alerted, and the wrapper exits non-zero.
    """
    configure_logging()
    if not command:
        raise ValueError("operator run requires a driver command after --")

    alert_cmd = get_settings().alert_cmd

    # Resolve + bind the job identity BEFORE the lock/gate — fail closed on an unknown job or any
    # deviation from the canonical argv, so no subprocess ever runs and no marker is ever poisoned.
    op_job = OPERATOR_JOBS.get(job)
    if op_job is None:
        emit_alert("unknown_job", {"job": job}, alert_cmd=alert_cmd)
        emit({"ok": False, "job": job, "ran": False, "reason": "unknown_job", "alerted": True})
        raise typer.Exit(1)
    try:
        op_job.bind(tuple(command))
    except CommandMismatch:
        emit_alert(
            "command_mismatch",
            {"job": job, "expected": list(op_job.argv_template), "got": list(command)},
            alert_cmd=alert_cmd,
        )
        emit(
            {
                "ok": False,
                "job": job,
                "ran": False,
                "reason": "command_mismatch",
                "expected": list(op_job.argv_template),
                "got": list(command),
                "alerted": True,
            }
        )
        raise typer.Exit(1) from None

    now_dt = datetime.fromisoformat(now) if now else datetime.now(UTC)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=UTC)

    host, pid = socket.gethostname(), os.getpid()
    lock_path = _resolve_git_dir() / "operator.lock"

    with correlation_context():
        try:
            with operator_run_lock(lock_path, job=job, host=host, pid=pid):
                _run_session(job, op_job, command, now_dt, alert_cmd, host, pid)
        except OperatorLockHeld as held:
            _emit_lock_held(job, op_job, held.holder, now_dt, alert_cmd)


def _held_seconds(holder: dict | None, now_dt: datetime) -> float | None:
    """Seconds the lock has been held, from the holder's ``started_at``; ``None`` if unknown."""
    if not holder:
        return None
    started = holder.get("started_at")
    if not isinstance(started, str):
        return None
    try:
        started_dt = datetime.fromisoformat(started)
    except ValueError:
        return None
    if started_dt.tzinfo is None:
        started_dt = started_dt.replace(tzinfo=UTC)
    return (now_dt - started_dt).total_seconds()


def _emit_lock_held(
    job: str,
    op_job: OperatorJob,
    holder: dict | None,
    now_dt: datetime,
    alert_cmd: str | None,
) -> None:
    """A held run lock: an ordinary overlap within grace is a benign no-op; a wedged holder past
    grace (or an unknown holder) is surfaced via an ``operator_lock_stuck`` alert. Either way the
    fire itself no-ops at exit 0 (flooding systemd with failures would blame the wrong process)."""
    held = _held_seconds(holder, now_dt)
    if held is None or held > op_job.expected_duration_seconds:
        emit_alert(
            "operator_lock_stuck",
            {
                "job": job,
                "holder_pid": (holder or {}).get("pid"),
                "holder_job": (holder or {}).get("job"),
                "started_at": (holder or {}).get("started_at"),
                "held_seconds": held,
                "grace_seconds": op_job.expected_duration_seconds,
            },
            alert_cmd=alert_cmd,
        )
    emit(
        ok(
            {
                "job": job,
                "ran": False,
                "reason": "locked",
                "session": None,
                "holder": holder,
            }
        )
    )


def _run_session(
    job: str,
    op_job: OperatorJob,
    command: list[str],
    now_dt: datetime,
    alert_cmd: str | None,
    host: str,
    pid: int,
) -> None:
    """Gate → (run → record), all inside the held run lock."""
    marker = SessionMarker(get_settings().db_path.parent)
    decision = session_gate(job, now_dt, MarketCalendar(), marker)
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
        raise typer.Exit(1)

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
        raise typer.Exit(1)

    if not decision.due:
        emit(ok({"job": job, "ran": False, "reason": decision.reason, "session": sess_iso}))
        return

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

    proc = _run_driver(command)
    payload = _parse_driver_payload(proc.stdout)
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
            return
        if not op_job.is_completed(rc, payload):
            # A benign deferral (driver chose not to trade): NOT completed, so the marker is left
            # unwritten and the next fire retries. Not a failure — no alert.
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
            return
        marker.record(job, decision.session, command=list(command), rc=rc, host=host, pid=pid)
        emit(ok({"job": job, "ran": True, "recorded": True, "session": sess_iso, "rc": rc}))
        return

    # rc != 0 — a failure. The alert ALWAYS fires and ALWAYS carries rc + stdout_head;
    # classification is a best-effort label only. Marker NOT recorded — the next fire re-attempts.
    kind = _classify_failure(payload)
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
    raise typer.Exit(1)
