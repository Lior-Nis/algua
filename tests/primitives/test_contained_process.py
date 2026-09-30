from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from algua.primitives import contained_process
from algua.primitives.contained_process import ContainedResult, run_contained

# The frozen-child environment shape (contract §5); LC_ALL also stops PEP 538 locale coercion from
# adding LC_CTYPE to the child's environment, so the env the child sees is exactly this mapping.
_ENV = {
    "PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
    "TZ": "UTC",
}


def _run(code: str, tmp_path: Path, **overrides: Any) -> ContainedResult:
    options: dict[str, Any] = {
        "env": _ENV, "timeout": 10.0, "max_stdout": 4096, "stderr_capture": 4096, "grace": 1.0}
    options.update(overrides)
    return run_contained([sys.executable, "-I", "-S", "-c", code], cwd=tmp_path, **options)


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _wait_gone(pid: int) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if _gone(pid):
            return True
        time.sleep(0.05)
    return False


def _open_fds() -> set[str]:
    return set(os.listdir("/proc/self/fd"))


def test_a_normal_exit_captures_both_streams(tmp_path: Path) -> None:
    result = _run("import sys; sys.stdout.write('out'); sys.stderr.write('err')", tmp_path)

    assert result == ContainedResult(
        returncode=0, signal=None, timed_out=False, stdout=b"out", stdout_exceeded=False,
        stderr=b"err", stderr_truncated=False)


def test_a_nonzero_exit_status_is_reported(tmp_path: Path) -> None:
    result = _run("import sys; sys.stdout.write('partial'); sys.exit(3)", tmp_path)

    assert (result.returncode, result.signal, result.timed_out) == (3, None, False)
    assert result.stdout == b"partial"


def test_death_by_a_signal_reports_the_signal_not_a_returncode(tmp_path: Path) -> None:
    result = _run("import os, signal; os.kill(os.getpid(), signal.SIGTERM)", tmp_path)

    assert (result.returncode, result.signal, result.timed_out) == (None, signal.SIGTERM, False)


def test_a_timeout_sigkills_a_child_that_ignores_sigterm_after_the_grace(tmp_path: Path) -> None:
    code = (
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "sys.stdout.write('armed'); sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )
    started = time.monotonic()

    result = _run(code, tmp_path, timeout=0.5, grace=0.3)

    elapsed = time.monotonic() - started
    assert result.stdout == b"armed"  # SIGTERM arrived only after the child ignored it
    assert (result.timed_out, result.returncode, result.signal) == (True, None, signal.SIGKILL)
    assert 0.75 <= elapsed < 5


def test_a_timeout_needs_no_sigkill_when_the_child_exits_on_sigterm(tmp_path: Path) -> None:
    started = time.monotonic()

    result = _run("import time; time.sleep(60)", tmp_path, timeout=0.5, grace=5.0)

    assert (result.timed_out, result.returncode, result.signal) == (True, None, signal.SIGTERM)
    assert time.monotonic() - started < 4  # the 5 s grace was not waited out


def test_stdout_exactly_at_the_bound_is_accepted(tmp_path: Path) -> None:
    result = _run("import sys; sys.stdout.write('x' * 1000)", tmp_path, max_stdout=1000)

    assert (result.stdout, result.stdout_exceeded, result.returncode) == (b"x" * 1000, False, 0)


def test_stdout_overflow_is_cut_at_the_bound_and_the_child_terminated(tmp_path: Path) -> None:
    code = "import sys\nwhile True: sys.stdout.write('x' * 65536); sys.stdout.flush()"
    started = time.monotonic()

    result = _run(code, tmp_path, timeout=30.0, max_stdout=1000)

    assert result.stdout_exceeded and not result.timed_out
    assert result.stdout == b"x" * 1000
    assert result.returncode != 0
    assert time.monotonic() - started < 10


def test_stdout_reads_never_request_more_than_one_byte_past_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[int] = []
    real_read = contained_process._read

    def recording(fd: int, size: int) -> bytes:
        chunk = real_read(fd, size)
        if chunk:  # the child writes only stdout; stderr reads see just EOF
            requested.append(size)
        return chunk

    monkeypatch.setattr(contained_process, "_read", recording)

    result = _run("import sys; sys.stdout.write('x' * 100000)", tmp_path, max_stdout=10)

    assert result.stdout_exceeded
    assert requested and max(requested) <= 11


def test_stderr_beyond_the_capture_is_drained_and_is_not_a_failure(tmp_path: Path) -> None:
    # 4 MiB is far past any pipe buffer: a runner that stopped reading stderr at the capture
    # would leave the child blocked on write and the run would end in a timeout instead.
    code = "import sys; sys.stderr.write('HEAD' + 'x' * (4 << 20)); sys.stdout.write('ok')"
    started = time.monotonic()

    result = _run(code, tmp_path, timeout=5.0, stderr_capture=100)

    assert (result.returncode, result.timed_out, result.stdout) == (0, False, b"ok")
    assert result.stderr == b"HEAD" + b"x" * 96
    assert result.stderr_truncated and not result.stdout_exceeded
    assert time.monotonic() - started < 4


def test_stderr_exactly_at_the_capture_is_not_truncated(tmp_path: Path) -> None:
    result = _run("import sys; sys.stderr.write('e' * 100)", tmp_path, stderr_capture=100)

    assert (result.stderr, result.stderr_truncated) == (b"e" * 100, False)


def test_a_stray_grandchild_in_the_process_group_is_killed(tmp_path: Path) -> None:
    # The grandchild stays in the child's process group and inherits both pipes, so a runner
    # that did not kill the group after the child's exit would also hang until the deadline.
    code = (
        "import subprocess, sys\n"
        f"child = subprocess.Popen([{sys.executable!r}, '-I', '-S', '-c', "
        "'import time; time.sleep(60)'])\n"
        "sys.stdout.write(str(child.pid))\n"
    )
    started = time.monotonic()

    result = _run(code, tmp_path, timeout=5.0)

    assert (result.returncode, result.timed_out) == (0, False)
    assert time.monotonic() - started < 4
    assert _wait_gone(int(result.stdout))


def test_a_pipe_holder_that_escaped_the_group_cannot_outlast_the_deadline(tmp_path: Path) -> None:
    # A new-session descendant is outside the group kill; it only must not wedge the caller.
    code = (
        "import subprocess, sys\n"
        f"child = subprocess.Popen([{sys.executable!r}, '-I', '-S', '-c', "
        "'import time; time.sleep(60)'], start_new_session=True)\n"
        "sys.stdout.write(str(child.pid))\n"
    )
    started = time.monotonic()

    result = _run(code, tmp_path, timeout=0.5, grace=0.2)

    escaped = int(result.stdout)
    os.kill(escaped, signal.SIGKILL)
    assert (result.returncode, result.timed_out) == (0, True)
    assert time.monotonic() - started < 4


def test_the_post_exit_group_kill_happens_while_the_zombie_holds_the_pgid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int]] = []
    real_waitid, real_killpg = contained_process._waitid, contained_process._killpg

    def waitid(idtype: int, ident: int, options: int) -> Any:
        outcome = real_waitid(idtype, ident, options)
        if options & os.WNOWAIT and outcome is not None:
            events.append(("observed", ident))
        return outcome

    def killpg(pgid: int, sig: int) -> None:
        try:  # is the group leader still an unreaped zombie at this signal?
            state = real_waitid(os.P_PID, pgid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
            events.append(("kill-zombie" if state is not None else "kill-running", sig))
        except ChildProcessError:
            events.append(("kill-reaped", sig))
        real_killpg(pgid, sig)

    monkeypatch.setattr(contained_process, "_waitid", waitid)
    monkeypatch.setattr(contained_process, "_killpg", killpg)

    result = _run("import sys; sys.exit(0)", tmp_path)

    assert result.returncode == 0
    kills = [event for event in events if event[0].startswith("kill")]
    assert kills == [("kill-zombie", signal.SIGKILL)]
    assert events.index(("kill-zombie", signal.SIGKILL)) > 0  # observed before the kill
    pid = events[0][1]
    with pytest.raises(ChildProcessError):  # and reaped afterwards: no zombie is left behind
        real_waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)


def test_a_timeout_signals_the_group_term_then_kill_before_the_post_exit_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []
    real_killpg = contained_process._killpg

    def killpg(pgid: int, sig: int) -> None:
        signals.append(sig)
        real_killpg(pgid, sig)

    monkeypatch.setattr(contained_process, "_killpg", killpg)
    code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"

    _run(code, tmp_path, timeout=0.3, grace=0.2)

    assert signals == [signal.SIGTERM, signal.SIGKILL, signal.SIGKILL]


def test_the_child_sees_exactly_the_given_env_and_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALGUA_CONTAINED_LEAK_PROBE", "1")
    workdir = tmp_path / "work"
    workdir.mkdir()
    env = {**_ENV, "ALGUA_PROBE": "value"}

    result = run_contained(
        [sys.executable, "-I", "-S", "-c",
         "import json, os, sys; json.dump([dict(os.environ), os.getcwd()], sys.stdout)"],
        cwd=workdir, env=env, timeout=10.0, max_stdout=65536, stderr_capture=4096, grace=1.0)

    seen_env, seen_cwd = json.loads(result.stdout)
    assert seen_env == env
    assert seen_cwd == str(workdir.resolve())


def test_stdin_is_dev_null(tmp_path: Path) -> None:
    code = (
        "import os, sys\n"
        "print(os.path.samestat(os.fstat(0), os.stat('/dev/null')), sys.stdin.buffer.read())\n"
    )

    result = _run(code, tmp_path, timeout=5.0)

    assert result.stdout == b"True b''\n"


def test_an_inheritable_parent_descriptor_is_not_visible_in_the_child(tmp_path: Path) -> None:
    fd = os.open(tmp_path / "secret", os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        os.set_inheritable(fd, True)
        code = (
            "import os\n"
            f"try: os.fstat({fd})\n"
            "except OSError: print('closed')\n"
            "else: print('open')\n"
        )
        result = _run(code, tmp_path)
    finally:
        os.close(fd)

    assert result.stdout == b"closed\n"


@pytest.mark.parametrize("code, overrides", [
    ("import sys; sys.stdout.write('ok')", {}),
    ("import sys\nwhile True: sys.stdout.write('x' * 65536)", {"max_stdout": 10}),
    ("import time; time.sleep(60)", {"timeout": 0.3, "grace": 0.3}),
])
def test_no_descriptor_is_leaked(tmp_path: Path, code: str, overrides: dict[str, Any]) -> None:
    before = _open_fds()

    _run(code, tmp_path, **overrides)

    assert _open_fds() == before


def test_launch_failure_propagates_as_an_os_error(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        run_contained(
            [str(tmp_path / "missing")], cwd=tmp_path, env=_ENV, timeout=5.0, max_stdout=1,
            stderr_capture=1, grace=0.0)


@pytest.mark.parametrize("overrides", [
    {"timeout": 0.0}, {"timeout": -1.0}, {"timeout": float("nan")}, {"timeout": float("inf")},
    {"max_stdout": 0}, {"stderr_capture": 0}, {"grace": -0.1}, {"grace": float("nan")},
    {"grace": float("inf")},
])
def test_invalid_bounds_are_refused_before_launch(
    tmp_path: Path, overrides: dict[str, Any],
) -> None:
    marker = tmp_path / "launched"

    with pytest.raises(ValueError):
        _run(f"open({str(marker)!r}, 'w')", tmp_path, **overrides)

    assert not marker.exists()
