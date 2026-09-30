"""Run a child in its own process group with bounded output, a deadline and a full teardown.

Unlike `run_bounded`, which raises, this runner reports every outcome in a `ContainedResult` so a
caller can classify it. Both pipes are read incrementally: stdout up to its bound (one byte past it
means overflow), stderr keeping only its head while the rest is drained and discarded so the child
never blocks on a full pipe. On a timeout or a stdout overflow the child's process group gets
SIGTERM, then SIGKILL once the grace has passed.

However the child ends, its exit is first observed with ``waitid(WNOWAIT)``: the unreaped zombie
keeps its PID, and so its process group ID, reserved. Only then is the group SIGKILLed, so every
descendant still in it dies and a recycled PGID can never be signalled; the child is reaped last.
"""
from __future__ import annotations

import math
import os
import selectors
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

_CHUNK = 64 * 1024
_POLL = 0.05  # longest quiet interval between checks for the child's exit
_read = os.read
_killpg = os.killpg
_waitid = os.waitid


@dataclass(frozen=True)
class ContainedResult:
    returncode: int | None  # exit status if the child exited normally, else None
    signal: int | None  # terminating signal number, else None
    timed_out: bool
    stdout: bytes  # at most max_stdout bytes
    stdout_exceeded: bool
    stderr: bytes  # at most stderr_capture bytes (the head)
    stderr_truncated: bool


class _Capture:
    """Both pipes of one child: bounded stdout, a head-only stderr drained until EOF."""

    def __init__(
        self, selector: selectors.BaseSelector, stdout: IO[bytes], stderr: IO[bytes],
        max_stdout: int, stderr_capture: int,
    ) -> None:
        self._selector = selector
        self._stdout = stdout
        self._limits = {"stdout": max_stdout, "stderr": stderr_capture}
        self.buffers = {"stdout": bytearray(), "stderr": bytearray()}
        self.stdout_exceeded = False
        self.stderr_truncated = False
        selector.register(stdout, selectors.EVENT_READ, "stdout")
        selector.register(stderr, selectors.EVENT_READ, "stderr")

    def done(self) -> bool:
        """Every pipe reached EOF, or was abandoned after an overflow."""
        return not self._selector.get_map()

    def pump(self, timeout: float) -> None:
        """Read whatever becomes ready within ``timeout`` seconds (a plain wait once done)."""
        for key, _events in self._selector.select(timeout):
            stream = key.data
            buffer = self.buffers[stream]
            room = self._limits[stream] - len(buffer)
            # stdout never requests more than one byte past its bound; stderr always drains.
            chunk = _read(key.fd, min(_CHUNK, room + 1) if stream == "stdout" else _CHUNK)
            if not chunk:
                self._selector.unregister(key.fileobj)
                continue
            buffer.extend(chunk[:room])
            if len(chunk) <= room:
                continue
            if stream == "stderr":
                self.stderr_truncated = True
            else:  # stop accepting: a still-writing child now meets a closed pipe
                self.stdout_exceeded = True
                self._selector.unregister(self._stdout)
                self._stdout.close()


def _signal_group(pgid: int, sig: int) -> None:
    try:
        _killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _has_exited(pid: int) -> bool:
    return _waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT | os.WNOHANG) is not None


def _contain_and_reap(process: subprocess.Popen[bytes]) -> None:
    """Observe the exit without reaping, SIGKILL the group while the PGID is held, then reap."""
    _waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
    _signal_group(process.pid, signal.SIGKILL)
    process.wait()


def _terminate(process: subprocess.Popen[bytes], capture: _Capture, grace: float) -> None:
    """SIGTERM the group, keep draining for up to ``grace`` seconds, then SIGKILL it."""
    _signal_group(process.pid, signal.SIGTERM)
    grace_deadline = time.monotonic() + grace
    while not _has_exited(process.pid):
        remaining = grace_deadline - time.monotonic()
        if remaining <= 0:
            _signal_group(process.pid, signal.SIGKILL)
            break
        capture.pump(min(remaining, _POLL))
    _contain_and_reap(process)


def _supervise(
    process: subprocess.Popen[bytes], capture: _Capture, deadline: float, grace: float,
) -> bool:
    """Run until the child is reaped and its pipes are done; return whether the deadline hit.

    The pipes are drained to EOF after the child's exit (its group is dead by then), but a holder
    that escaped the group cannot keep them open past the deadline: that ends the run as timed out.
    """
    while True:
        if process.returncode is None:
            if _has_exited(process.pid):
                _contain_and_reap(process)
            elif capture.stdout_exceeded:
                _terminate(process, capture, grace)
        if process.returncode is not None and (capture.stdout_exceeded or capture.done()):
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if process.returncode is None:
                _terminate(process, capture, grace)
            return True
        capture.pump(min(remaining, _POLL))


def run_contained(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
    max_stdout: int,
    stderr_capture: int,
    grace: float,
) -> ContainedResult:
    """Run ``argv`` contained and report how it ended; a launch failure raises `OSError`.

    ``env`` replaces the environment; stdin is ``/dev/null``; no other descriptor is inherited.
    """
    if not (math.isfinite(timeout) and timeout > 0):
        raise ValueError("timeout must be a finite positive number of seconds")
    if not (math.isfinite(grace) and grace >= 0):
        raise ValueError("grace must be a finite non-negative number of seconds")
    if max_stdout <= 0 or stderr_capture <= 0:
        raise ValueError("max_stdout and stderr_capture must be positive byte counts")
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        list(argv), cwd=cwd, env=dict(env), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, shell=False, close_fds=True, start_new_session=True,
    )
    assert process.stdout is not None and process.stderr is not None
    try:
        with selectors.DefaultSelector() as selector:
            capture = _Capture(
                selector, process.stdout, process.stderr, max_stdout, stderr_capture)
            timed_out = _supervise(process, capture, deadline, grace)
    except BaseException:
        if process.returncode is None:
            _signal_group(process.pid, signal.SIGKILL)
            _contain_and_reap(process)
        raise
    finally:
        process.stdout.close()
        process.stderr.close()
    status = process.returncode
    return ContainedResult(
        returncode=status if status >= 0 else None,
        signal=-status if status < 0 else None,
        timed_out=timed_out,
        stdout=bytes(capture.buffers["stdout"]),
        stdout_exceeded=capture.stdout_exceeded,
        stderr=bytes(capture.buffers["stderr"]),
        stderr_truncated=capture.stderr_truncated,
    )
