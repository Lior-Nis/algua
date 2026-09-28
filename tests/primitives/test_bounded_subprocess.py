from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from algua.primitives import bounded_subprocess
from algua.primitives.bounded_subprocess import OutputLimitExceeded, run_bounded


def _run(code: str, tmp_path: Path, **kwargs):
    options = {"timeout": 10.0, "max_stdout": 1024, "max_stderr": 1024}
    options.update(kwargs)
    return run_bounded(
        [sys.executable, "-I", "-c", code], cwd=tmp_path, env={"PATH": "/usr/bin"}, **options)


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


def test_captures_both_streams_and_the_exit_status(tmp_path: Path) -> None:
    result = _run(
        "import sys; sys.stdout.write('out'); sys.stderr.write('err'); sys.exit(3)", tmp_path)

    assert (result.returncode, result.stdout, result.stderr) == (3, b"out", b"err")


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_output_exactly_at_the_bound_is_accepted(tmp_path: Path, stream: str) -> None:
    result = _run(f"import sys; sys.{stream}.write('x' * 1024)", tmp_path)

    assert getattr(result, stream) == b"x" * 1024


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_output_one_byte_over_the_bound_is_refused(tmp_path: Path, stream: str) -> None:
    with pytest.raises(OutputLimitExceeded) as caught:
        _run(f"import sys; sys.{stream}.write('x' * 1025)", tmp_path)

    assert caught.value.stream == stream


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_unbounded_output_is_stopped_promptly_and_the_process_killed(
    tmp_path: Path, stream: str,
) -> None:
    pid_file = tmp_path / "pid"
    code = (
        f"import os, sys; open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        f"while True: sys.{stream}.write('x' * 65536); sys.{stream}.flush()"
    )
    started = time.monotonic()

    with pytest.raises(OutputLimitExceeded):
        _run(code, tmp_path, timeout=30.0)

    assert time.monotonic() - started < 10
    assert _wait_gone(int(pid_file.read_text()))


def test_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild"
    code = (
        "import subprocess, sys\n"
        "child = subprocess.Popen(['sleep', '60'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "sys.stdout.write('started'); sys.stdout.flush()\n"
    )
    started = time.monotonic()

    with pytest.raises(subprocess.TimeoutExpired):
        _run(code, tmp_path, timeout=1.5)

    assert time.monotonic() - started < 10
    assert _wait_gone(int(pid_file.read_text()))


def test_a_hung_process_times_out(tmp_path: Path) -> None:
    started = time.monotonic()

    with pytest.raises(subprocess.TimeoutExpired):
        _run("import time; time.sleep(60)", tmp_path, timeout=0.5)

    assert time.monotonic() - started < 10


def test_pipe_reads_never_request_more_than_one_byte_past_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[int] = []
    real_read = bounded_subprocess._read

    def recording(fd: int, size: int) -> bytes:
        requested.append(size)
        return real_read(fd, size)

    monkeypatch.setattr(bounded_subprocess, "_read", recording)

    with pytest.raises(OutputLimitExceeded):
        _run("import sys; sys.stdout.write('x' * 100000)", tmp_path, max_stdout=10)

    assert requested and max(requested) <= 11


def test_launch_failure_is_an_os_error(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        run_bounded(
            [str(tmp_path / "missing")], cwd=tmp_path, env={}, timeout=5, max_stdout=1,
            max_stderr=1)
