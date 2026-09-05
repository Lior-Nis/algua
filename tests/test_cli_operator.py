"""In-process CLI tests for `algua operator run` (#486).

Drives the mounted Typer app with CliRunner. The driver subprocess is stubbed via the
``_run_driver`` seam; the marker dir is isolated by pointing ``ALGUA_DB_PATH`` at a tmp file; the
git-dir run lock is isolated by monkeypatching ``_resolve_git_dir`` to the tmp dir (so
``operator.lock`` sits at ``tmp/operator.lock``, never the real repo's git dir).
"""

from __future__ import annotations

import fcntl
import json
import subprocess
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

import algua.cli.operator_cmd as operator_cmd
import algua.operator.session_runner as session_runner
from algua.cli.main import app
from algua.operator.schedule import SessionMarker

runner = CliRunner()

_DUE = "2023-06-01T21:30:00+00:00"  # after the 2023-06-01 XNYS close -> session 2023-06-01
_SESSION = date(2023, 6, 1)
_CMD = ["algua", "paper", "run-all", "--refresh"]  # the paper job's canonical argv


@pytest.fixture
def db_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the marker dir AND the run lock at tmp_path; drop any ambient alert cmd."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "algua.db"))
    monkeypatch.delenv("ALGUA_ALERT_CMD", raising=False)
    monkeypatch.setattr(operator_cmd, "_resolve_git_dir", lambda: tmp_path)
    return tmp_path


def _record(m: SessionMarker, session: date) -> None:
    m.record("paper", session, command=_CMD, rc=0, host="t", pid=1)


def _fake_driver(returncode: int, stdout: str):
    def _run(command: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(command, returncode=returncode, stdout=stdout, stderr="")

    return _run


def _spy_alerts(monkeypatch) -> list[tuple[str, dict]]:
    alerts: list[tuple[str, dict]] = []
    def _rec(kind, detail, **kw):
        alerts.append((kind, detail))
        return False

    # `operator_cmd.emit_alert` covers alerts fired by `run()` itself (unknown_job,
    # command_mismatch) and by `_emit_lock_held`/`_emit_setup_failed`; `session_runner.emit_alert`
    # covers alerts fired from inside the moved decision tree (stage 6d task 3) — it is a distinct
    # module-level binding there (an ordinary import, not the injected `emit`/`ok` seam), so both
    # must be patched to catch every alert regardless of which layer raised it.
    monkeypatch.setattr(operator_cmd, "emit_alert", _rec)
    monkeypatch.setattr(session_runner, "emit_alert", _rec)
    return alerts


def _invoke(job: str = "paper", cmd: list[str] | None = None, now: str = _DUE):
    return runner.invoke(app, ["operator", "run", "--job", job, "--now", now, "--", *(cmd or _CMD)])


# --- (a) already-ran: pre-seeded marker suppresses the run --------------------------------------


def test_already_ran_suppresses_and_skips_driver(db_dir, monkeypatch):
    _record(SessionMarker(db_dir), _SESSION)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        operator_cmd, "_run_driver", lambda c, *_: calls.append(c) or _fake_driver(0, "{}")(c)
    )

    result = _invoke()

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["ran"] is False
    assert payload["reason"] == "already_ran"
    assert calls == []


# --- (b) weekend firing no-ops: Friday already recorded -> already_ran, no run ------------------


def test_weekend_after_friday_ran_is_no_op(db_dir, monkeypatch):
    # Sunday maps to Friday's completed session (June 2). With Friday recorded, the weekend firing
    # is idempotent — already_ran, driver NOT called.
    _record(SessionMarker(db_dir), date(2023, 6, 2))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        operator_cmd, "_run_driver", lambda c, *_: calls.append(c) or _fake_driver(0, "{}")(c)
    )

    result = _invoke(now="2023-06-04T07:00:00+00:00")  # Sunday

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ran"] is False
    assert payload["reason"] == "already_ran"
    assert calls == []


# --- (c) due, clean completed run: marker recorded (full argv) ----------------------------------


def test_due_clean_run_records_full_argv(db_dir, monkeypatch):
    calls: list[list[str]] = []

    def _spy(command, timeout=None):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout='{"ok":true}', stderr="")

    monkeypatch.setattr(operator_cmd, "_run_driver", _spy)

    result = _invoke()

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ran"] is True
    assert payload["recorded"] is True
    assert payload["rc"] == 0
    assert calls == [_CMD]
    assert SessionMarker(db_dir).last_session("paper") == _SESSION
    entry = json.loads((db_dir / "operator_sessions.json").read_text())["paper"]
    assert entry["command"] == _CMD  # the full canonical argv, not a head prefix


# --- (d) due, rc!=0 global-halt: alert + NO marker + ok:false + exit 1 --------------------------


def test_due_failed_global_halt(db_dir, monkeypatch):
    monkeypatch.setattr(operator_cmd, "_run_driver", _fake_driver(1, '{"ok":false,"halted":true}'))
    alerts = _spy_alerts(monkeypatch)

    result = _invoke()

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is False  # the rc!=0 envelope is ok:false, not ok:true
    assert payload["ran"] is True
    assert payload["rc"] == 1
    assert payload["alert_kind"] == "global_halt"
    assert alerts[0][0] == "global_halt"
    assert alerts[0][1]["rc"] == 1 and "stdout_head" in alerts[0][1]
    assert SessionMarker(db_dir).last_session("paper") is None


def test_due_failed_non_json_is_job_failed(db_dir, monkeypatch):
    monkeypatch.setattr(operator_cmd, "_run_driver", _fake_driver(2, "not json"))
    alerts = _spy_alerts(monkeypatch)

    result = _invoke()

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["alert_kind"] == "job_failed"
    assert alerts[0][0] == "job_failed"


def test_due_failed_ok_false_is_breach(db_dir, monkeypatch):
    monkeypatch.setattr(operator_cmd, "_run_driver", _fake_driver(1, '{"ok":false}'))
    alerts = _spy_alerts(monkeypatch)

    result = _invoke()

    assert result.exit_code == 1, result.output
    assert json.loads(result.output)["alert_kind"] == "breach"
    assert alerts[0][0] == "breach"


# --- (d') due, driver HANGS: killed at the timeout cap, alert + NO marker + exit 1 --------------


def test_due_driver_timeout_kills_and_leaves_marker_for_retry(db_dir, monkeypatch):
    seen: dict[str, float] = {}

    def _hang(command, timeout=None):
        seen["timeout"] = timeout
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

    monkeypatch.setattr(operator_cmd, "_run_driver", _hang)
    alerts = _spy_alerts(monkeypatch)

    result = _invoke()

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["ran"] is True
    assert payload["recorded"] is False
    assert payload["reason"] == "driver_timeout"
    # the wrapper hard-caps the driver at 2x the paper job's 900s grace
    assert payload["timeout_seconds"] == 1800.0
    assert seen["timeout"] == 1800.0
    assert alerts[0][0] == "driver_timeout"
    # marker left unwritten so the NEXT fire re-attempts the session
    assert SessionMarker(db_dir).last_session("paper") is None


# --- (i') due, driver CANNOT BE SPAWNED (OSError): alert + NO marker + exit 1 -------------------


def test_due_driver_spawn_failure_alerts_and_leaves_marker_for_retry(db_dir, monkeypatch):
    # GATE-2 (#486): before this fix, a spawn failure (binary not on PATH, permission denied, …)
    # propagated past the run lock straight to the generic `@json_errors` catch-all, which renders
    # an error envelope but never alerts — every fire would fail forever with zero paging.
    def _unspawnable(command, timeout=None):
        raise FileNotFoundError(2, "No such file or directory", command[0])

    monkeypatch.setattr(operator_cmd, "_run_driver", _unspawnable)
    alerts = _spy_alerts(monkeypatch)

    result = _invoke()

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["ran"] is False
    assert payload["recorded"] is False
    assert payload["reason"] == "driver_spawn_failed"
    assert alerts[0][0] == "driver_spawn_failed"
    # marker left unwritten so the NEXT fire re-attempts the session
    assert SessionMarker(db_dir).last_session("paper") is None


# --- (j) due, rc==0 deferred: NO marker, NO alert, exit 0 ---------------------------------------


def test_due_rc0_deferred_no_marker_no_alert(db_dir, monkeypatch):
    stdout = json.dumps({"ok": True, "deferred": True, "reason": "no signal"}, indent=2)
    monkeypatch.setattr(operator_cmd, "_run_driver", _fake_driver(0, stdout))
    alerts = _spy_alerts(monkeypatch)

    result = _invoke()

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ran"] is True
    assert payload["recorded"] is False
    assert payload["reason"] == "deferred"
    assert alerts == []  # a deferral is not a failure — no alert
    assert SessionMarker(db_dir).last_session("paper") is None


# --- (k) due, rc==0 non-JSON: completion_unconfirmed alert, NO marker, exit 0 -------------------


def test_due_rc0_non_json_completion_unconfirmed(db_dir, monkeypatch):
    monkeypatch.setattr(operator_cmd, "_run_driver", _fake_driver(0, "not json at all"))
    alerts = _spy_alerts(monkeypatch)

    result = _invoke()

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["recorded"] is False
    assert payload["reason"] == "completion_unconfirmed"
    assert alerts[0][0] == "completion_unconfirmed"
    assert SessionMarker(db_dir).last_session("paper") is None


# --- (k') due, rc==0 non-success payload (NOT deferred): completion_unconfirmed alert, NO marker --


@pytest.mark.parametrize(
    "body",
    [
        {"ok": False, "reason": "boom"},  # explicit failure at rc0
        {"reason": "no ok field"},  # ok-less envelope
        {},  # empty object
    ],
)
def test_due_rc0_non_success_payload_alerts_completion_unconfirmed(db_dir, monkeypatch, body):
    # GATE-2 (#486): rc==0 with a parsed payload that is NOT `ok:true` and NOT `deferred:true` must
    # NOT be silently misfiled as a benign `deferred` — that would retry a broken driver forever
    # with zero paging. It fails closed like the unparseable case: alert + no marker + exit 0.
    stdout = json.dumps(body, indent=2)
    monkeypatch.setattr(operator_cmd, "_run_driver", _fake_driver(0, stdout))
    alerts = _spy_alerts(monkeypatch)

    result = _invoke()

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["recorded"] is False
    assert payload["reason"] == "completion_unconfirmed"
    assert alerts[0][0] == "completion_unconfirmed"
    assert SessionMarker(db_dir).last_session("paper") is None


# --- (g) corrupt marker: alert + fail closed + exit 1 ------------------------------------------


def test_corrupt_marker_fails_closed(db_dir, monkeypatch):
    (db_dir / "operator_sessions.json").write_text("{corrupt json")
    calls: list = []
    monkeypatch.setattr(operator_cmd, "_run_driver", lambda c, *_: calls.append(c))
    alerts = _spy_alerts(monkeypatch)

    result = _invoke()

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["reason"] == "marker_corrupt"
    assert alerts[0][0] == "marker_corrupt"
    assert calls == []


# --- (l) calendar out of bounds: alert + fail closed + exit 1 -----------------------------------


def test_calendar_out_of_bounds_fails_closed(db_dir, monkeypatch):
    calls: list = []
    monkeypatch.setattr(operator_cmd, "_run_driver", lambda c, *_: calls.append(c))
    alerts = _spy_alerts(monkeypatch)

    result = _invoke(now="2099-01-01T21:30:00+00:00")

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["reason"] == "calendar_out_of_bounds"
    assert alerts[0][0] == "calendar_out_of_bounds"
    assert calls == []


# --- (m) command mismatch (wrong head AND trailing junk) ----------------------------------------


@pytest.mark.parametrize(
    "cmd",
    [
        ["algua", "data", "inspect"],  # wrong head
        ["algua", "paper", "run-all", "--refresh", "--evil"],  # trailing junk
    ],
)
def test_command_mismatch_fails_closed(db_dir, monkeypatch, cmd):
    calls: list = []
    monkeypatch.setattr(operator_cmd, "_run_driver", lambda c, *_: calls.append(c))
    alerts = _spy_alerts(monkeypatch)

    result = _invoke(cmd=cmd)

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["reason"] == "command_mismatch"
    assert alerts[0][0] == "command_mismatch"
    assert calls == []


# --- (n) unknown job ----------------------------------------------------------------------------


def test_unknown_job_fails_closed(db_dir, monkeypatch):
    calls: list = []
    monkeypatch.setattr(operator_cmd, "_run_driver", lambda c, *_: calls.append(c))
    alerts = _spy_alerts(monkeypatch)

    result = _invoke(job="frobnicate")

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["reason"] == "unknown_job"
    assert alerts[0][0] == "unknown_job"
    assert calls == []


# --- (o) git-dir unresolvable: alert + fail closed + exit 1, driver NOT called ------------------


def test_git_dir_unresolvable_fails_closed(db_dir, monkeypatch):
    # GATE-2 (#486): the run lock is anchored at the per-worktree git dir; if it cannot be resolved
    # (not a git worktree, `git` missing, permission denied) the fire must alert + fail closed, not
    # escape to the generic `@json_errors` catch-all that pages no one and fails every fire forever.
    def _boom() -> Path:
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "--absolute-git-dir"])

    monkeypatch.setattr(operator_cmd, "_resolve_git_dir", _boom)
    calls: list = []
    monkeypatch.setattr(operator_cmd, "_run_driver", lambda c, *_: calls.append(c))
    alerts = _spy_alerts(monkeypatch)

    result = _invoke()

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["ran"] is False
    assert payload["reason"] == "git_dir_unresolved"
    assert alerts[0][0] == "git_dir_unresolved"
    assert calls == []
    assert SessionMarker(db_dir).last_session("paper") is None


# --- (p) lock file unopenable: alert + fail closed + exit 1, driver NOT called ------------------


def test_lock_file_unopenable_fails_closed(db_dir, monkeypatch):
    # GATE-2 (#486): `operator_run_lock` raises a RAW OSError (not OperatorLockHeld) when it cannot
    # open/create the lock file — permission denied, read-only fs, disk full. Force the shared
    # primitive's `os.open` (algua.primitives.flock, #8) to fail: `operator_run_lock` now goes
    # through `flock.acquire`/`file_lock`, which auto-creates any missing parent dir and would
    # otherwise mask a bare "missing dir" simulation; it must alert + fail closed the same way,
    # never silently at the un-paging catch-all.
    def _open_boom(*_a, **_k):
        raise PermissionError("EACCES")

    monkeypatch.setattr("algua.primitives.flock.os.open", _open_boom)
    calls: list = []
    monkeypatch.setattr(operator_cmd, "_run_driver", lambda c, *_: calls.append(c))
    alerts = _spy_alerts(monkeypatch)

    result = _invoke()

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["ran"] is False
    assert payload["reason"] == "lock_unavailable"
    assert alerts[0][0] == "lock_unavailable"
    assert calls == []
    assert SessionMarker(db_dir).last_session("paper") is None


# --- (h)/(i) run-lock contention: benign within grace, stuck past grace -------------------------


def _hold_lock(db_dir: Path, started_at: str):
    """Hold tmp/operator.lock with the given holder metadata (a separate fd => the CLI's LOCK_NB
    fails even in-process). Returns the open handle; caller must release it."""
    lock = db_dir / "operator.lock"
    handle = open(lock, "a+")
    fcntl.flock(handle, fcntl.LOCK_EX)
    handle.seek(0)
    handle.truncate()
    holder = {"pid": 999, "job": "paper", "started_at": started_at, "host": "other"}
    handle.write(json.dumps(holder))
    handle.flush()
    return handle


def test_run_lock_within_grace_is_benign_no_op(db_dir, monkeypatch):
    calls: list = []
    monkeypatch.setattr(operator_cmd, "_run_driver", lambda c, *_: calls.append(c))
    alerts = _spy_alerts(monkeypatch)
    handle = _hold_lock(db_dir, "2023-06-01T21:29:30+00:00")  # 30s before _DUE -> within grace
    try:
        result = _invoke()
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ran"] is False
    assert payload["reason"] == "locked"
    assert calls == []
    assert alerts == []  # within grace -> no stuck alert


def test_run_lock_past_grace_alerts_stuck(db_dir, monkeypatch):
    calls: list = []
    monkeypatch.setattr(operator_cmd, "_run_driver", lambda c, *_: calls.append(c))
    alerts = _spy_alerts(monkeypatch)
    handle = _hold_lock(db_dir, "2023-06-01T18:00:00+00:00")  # 3.5h before _DUE -> past 900s grace
    try:
        result = _invoke()
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["reason"] == "locked"
    assert alerts[0][0] == "operator_lock_stuck"
    assert calls == []


# --- (o) session gap: due after a one-session gap fires session_gap before running --------------


def test_session_gap_alerts_before_running(db_dir, monkeypatch):
    _record(SessionMarker(db_dir), date(2023, 5, 30))  # two sessions back -> one skipped (5/31)
    calls: list = []
    monkeypatch.setattr(
        operator_cmd,
        "_run_driver",
        lambda c, *_: calls.append(c) or _fake_driver(0, '{"ok":true}')(c),
    )
    alerts = _spy_alerts(monkeypatch)

    result = _invoke()

    assert result.exit_code == 0, result.output
    assert calls == [_CMD]  # the run proceeded
    gaps = [a for a in alerts if a[0] == "session_gap"]
    assert len(gaps) == 1
    assert gaps[0][1]["skipped_sessions"] == 1  # proves > 0, not > 1


# --- (e) empty command -> error envelope --------------------------------------------------------


def test_empty_command_errors(db_dir):
    result = runner.invoke(app, ["operator", "run", "--job", "paper", "--now", _DUE])
    assert result.exit_code == 1
    assert json.loads(result.output)["ok"] is False


# --- (f) systemd packaging shape (paper only; research deferred) --------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SYSTEMD = _REPO_ROOT / "deploy" / "systemd"


def test_paper_systemd_units_present_and_shaped():
    svc = (_SYSTEMD / "algua-paper.service").read_text()
    assert "Type=oneshot" in svc
    assert "operator run --job paper" in svc
    # a hung firing must not wedge forever — systemd caps it as the last-resort backstop
    assert "TimeoutStartSec=" in svc
    assert "paper run-all --refresh" in svc
    assert "ALGUA_PAPER_SNAPSHOT" not in svc
    env = (_SYSTEMD / "algua.env.example").read_text()
    assert "ALGUA_PAPER_SNAPSHOT=" not in env
    installer = (_SYSTEMD / "install-user-units.sh").read_text()
    assert "ALGUA_PAPER_SNAPSHOT" not in installer

    tmr = (_SYSTEMD / "algua-paper.timer").read_text()
    assert "OnCalendar" in tmr
    assert "Persistent=true" in tmr


def test_research_systemd_units_shipped_and_sandboxed():
    # The research producer now ships (candidate producer). Verify the units exist and encode the
    # key safety properties: the service runs the launcher and scrubs broker creds (the sandboxed
    # agent has network but must carry no secrets); the timer is a weekly non-session-gated cycle.
    service = (_SYSTEMD / "algua-research.service").read_text()
    timer = (_SYSTEMD / "algua-research.timer").read_text()
    assert "run-research-loop.sh" in service            # drives the launcher
    assert "UnsetEnvironment=" in service               # broker creds scrubbed from the unit env
    assert "ALGUA_ALPACA_API_KEY" in service
    assert "ALGUA_ALPACA_API_SECRET" in service
    assert "OnCalendar=" in timer                        # scheduled cadence
    assert "WantedBy=timers.target" in timer


def test_due_clean_run_records_snapshot_id_from_driver_payload(db_dir, monkeypatch):
    monkeypatch.setattr(operator_cmd, "_run_driver", _fake_driver(
        0, json.dumps({"ok": True, "strategies": [{"strategy": "s", "ok": True}],
                       "snapshot": {"id": "snap-9", "refreshed": True}})))
    r = _invoke()
    assert r.exit_code == 0, r.stdout
    assert json.loads(r.stdout)["snapshot_id"] == "snap-9"
    entry = json.loads((db_dir / "operator_sessions.json").read_text())["paper"]
    assert entry["snapshot_id"] == "snap-9" and entry["command"] == _CMD


def test_due_ticked_without_snapshot_is_completion_unconfirmed(db_dir, monkeypatch):
    alerts = _spy_alerts(monkeypatch)
    monkeypatch.setattr(operator_cmd, "_run_driver", _fake_driver(
        0, json.dumps({"ok": True, "strategies": [{"strategy": "s", "ok": True}]})))
    r = _invoke()
    assert r.exit_code == 0, r.stdout
    assert json.loads(r.stdout)["recorded"] is False
    assert not (db_dir / "operator_sessions.json").exists() or \
        "paper" not in json.loads((db_dir / "operator_sessions.json").read_text())
    assert any(kind == "completion_unconfirmed" for kind, _ in alerts)


def test_resolve_driver_argv_uses_venv_sibling(monkeypatch, tmp_path):
    # Under a systemd unit the canonical "algua" argv head is NOT on PATH; the spawn must resolve
    # it to the sibling of the running interpreter (the venv's own bin/algua) — #548's bug class,
    # one layer deeper (masked for months by command_mismatch firing first).
    bin_dir = tmp_path / "venv-bin"
    bin_dir.mkdir()
    (bin_dir / "algua").write_text("#!/bin/sh\n")
    monkeypatch.setattr(operator_cmd.sys, "executable", str(bin_dir / "python3"))
    resolved = operator_cmd._resolve_driver_argv(["algua", "paper", "run-all"])
    assert resolved == [str(bin_dir / "algua"), "paper", "run-all"]


def test_resolve_driver_argv_passthrough_without_sibling(monkeypatch, tmp_path):
    # A dev shell whose interpreter has no algua sibling keeps the bare argv (PATH resolution).
    monkeypatch.setattr(operator_cmd.sys, "executable", str(tmp_path / "nowhere" / "python3"))
    assert operator_cmd._resolve_driver_argv(["algua", "x"]) == ["algua", "x"]


def test_resolve_driver_argv_ignores_non_algua_head(monkeypatch, tmp_path):
    bin_dir = tmp_path / "venv-bin"
    bin_dir.mkdir()
    (bin_dir / "algua").write_text("#!/bin/sh\n")
    monkeypatch.setattr(operator_cmd.sys, "executable", str(bin_dir / "python3"))
    assert operator_cmd._resolve_driver_argv(["/abs/algua", "x"]) == ["/abs/algua", "x"]
    assert operator_cmd._resolve_driver_argv([]) == []
