"""Run a subprocess with bounded captured output and a hard deadline.

`subprocess.run(capture_output=True)` buffers unbounded output in memory until the child exits,
and its timeout kills only the direct child. This runner reads both pipes incrementally, never
requests more than one byte past a stream's bound, and on overflow or timeout kills the child's
whole process group (it starts in a new session) before re-raising.
"""
from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_CHUNK = 64 * 1024
_read = os.read


class OutputLimitExceeded(RuntimeError):
    """A captured stream exceeded its byte bound; the process group was killed."""

    def __init__(self, stream: str) -> None:
        super().__init__(f"subprocess {stream} exceeded its byte bound")
        self.stream = stream


@dataclass(frozen=True)
class BoundedCompletion:
    returncode: int
    stdout: bytes
    stderr: bytes


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    process.wait()


def _collect(
    process: subprocess.Popen[bytes], deadline: float, timeout: float, limits: dict[str, int],
) -> dict[str, bytes]:
    assert process.stdout is not None and process.stderr is not None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
            for key, _events in selector.select(remaining):
                stream = key.data
                buffer = buffers[stream]
                chunk = _read(key.fd, min(_CHUNK, limits[stream] - len(buffer) + 1))
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffer.extend(chunk)
                if len(buffer) > limits[stream]:
                    raise OutputLimitExceeded(stream)
    return {stream: bytes(buffer) for stream, buffer in buffers.items()}


def run_bounded(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str],
    timeout: float,
    max_stdout: int,
    max_stderr: int,
) -> BoundedCompletion:
    """Run ``argv`` to completion; raise `OutputLimitExceeded`, `TimeoutExpired` or `OSError`."""
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        list(argv), cwd=cwd, env=dict(env), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True,
    )
    with process:
        try:
            output = _collect(
                process, deadline, timeout, {"stdout": max_stdout, "stderr": max_stderr})
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
            returncode = process.wait(timeout=remaining)
        except BaseException:
            _kill_group(process)
            raise
    return BoundedCompletion(returncode, output["stdout"], output["stderr"])
