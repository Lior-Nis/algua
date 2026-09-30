"""One frozen planner child: the pre-launch content check, its invocation directory and launch.

Story 1.3c contract §4–§6, the filesystem and process half of the dispatcher. The bundle must
carry a wire-v1 / boundary-v1 protocol stamp and the wire-v1 child entry module before anything is
launched. Each launch gets a private ``0700`` directory under the invocations root holding exactly
``request.json`` and ``bars.arrow``, fsynced, then sealed (files ``0444``, directory ``0555``). The
child runs through :func:`~algua.primitives.contained_process.run_contained` with the exact §5
argv, a replacement environment and the §6 bounds; afterwards the directory is removed on every
path, including an exception or interrupt, best effort — a directory that cannot be removed is
inert bounded input, as after a crash. A child whose interpreter cannot be started raises
:class:`LaunchFailure` (the tenant's ``frozen_launch_failed``); any other ``OSError`` — setting up
the invocation directory on the data volume, or the supervisor's own pipes and reaping — is the
supervisor's and propagates as systemic. How the child ended is the returned ``ContainedResult``,
which :func:`process_failure` maps to its §8 code and :func:`process_diagnostic` renders as the
bounded, sanitized diagnostic (§6, §8).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Final

from algua.live.frozen_wire import (
    BARS_FILE,
    BOOTSTRAP,
    CHILD_MODULE,
    EXIT_OK,
    EXIT_UNSUPPORTED,
    KILL_GRACE_SECONDS,
    MAX_DIAGNOSTIC_BYTES,
    MAX_REQUEST_BYTES,
    MAX_STDOUT_BYTES,
    REQUEST_FILE,
    STDERR_CAPTURE_BYTES,
    TIMEOUT_SECONDS,
    WireError,
)
from algua.live.frozen_wire_json import expect_wire, parse_canonical
from algua.live.planner_contract import BOUNDARY_VERSION
from algua.primitives.contained_process import ContainedResult

PROTOCOL_FILE: Final = "_algua/protocol.json"
CHILD_FILE: Final = CHILD_MODULE.replace(".", "/") + ".py"

type Runner = Callable[..., ContainedResult]


class LaunchFailure(Exception):
    """The child's interpreter could not be started in the bundle (§8 ``frozen_launch_failed``)."""


def unsupported_content(bundle_root: Path) -> str | None:
    """Why this supervisor cannot launch the bundle's child (§5), or ``None`` when it can."""
    try:
        with (bundle_root / PROTOCOL_FILE).open("rb") as handle:
            data = handle.read(MAX_REQUEST_BYTES + 1)  # the child's own bound on bundle JSON
    except OSError as exc:
        return f"{PROTOCOL_FILE} is unreadable: {exc.strerror}"
    if len(data) > MAX_REQUEST_BYTES:
        return f"{PROTOCOL_FILE} exceeds {MAX_REQUEST_BYTES} bytes"
    try:
        stamp = parse_canonical(data)
        if type(stamp) is not dict:
            raise WireError("bad_protocol", "the stamp is not an object")
        expect_wire(stamp.get("frozen_wire"))
        boundary = stamp.get("planner_boundary_version")
        if type(boundary) is not int or boundary != BOUNDARY_VERSION:
            raise WireError("unsupported_boundary_version", repr(boundary))
    except WireError as exc:
        return f"{PROTOCOL_FILE} is not frozen-planner wire 1, planner boundary 1: {exc}"
    try:
        present = (bundle_root / CHILD_FILE).is_file()
    except OSError as exc:  # e.g. EACCES on a directory: is_file() only swallows "not found"
        return f"{CHILD_FILE} is unreadable: {exc.strerror}"
    if not present:
        return f"the bundle holds no {CHILD_FILE}"
    return None


def _printable(value: str | bytes) -> str:
    text = value.decode("latin-1") if isinstance(value, bytes) else value
    return "".join(char for char in text if " " <= char <= "~")


def sanitize_text(value: str | bytes) -> str:
    """Printable ASCII only — control and non-ASCII characters removed — cut to 8 KiB (§6, §7)."""
    return _printable(value)[:MAX_DIAGNOSTIC_BYTES]


def _flag(value: object) -> str:
    return "none" if value is None else str(value).lower()


def process_diagnostic(result: ContainedResult) -> str:
    """Exit status, signal, truncation flags and the sanitized head of stderr, within 8 KiB."""
    head = _printable(result.stderr)

    def render(stderr_truncated: bool) -> str:
        return (
            f"exit_status={_flag(result.returncode)} signal={_flag(result.signal)} "
            f"timed_out={_flag(result.timed_out)} stdout_exceeded={_flag(result.stdout_exceeded)} "
            f"stderr_truncated={_flag(stderr_truncated)} stderr={head}"
        )

    text = render(result.stderr_truncated)
    if len(text) > MAX_DIAGNOSTIC_BYTES:  # the head is cut to fit: the diagnostic says so
        text = render(True)
    return text[:MAX_DIAGNOSTIC_BYTES]


def process_failure(result: ContainedResult) -> str | None:
    """The §8 code for how the child ended, or ``None`` when its stdout is the result."""
    if result.timed_out:
        return "frozen_timeout"
    if result.stdout_exceeded:
        return "frozen_output_exceeded"
    if result.signal is None and result.returncode == EXIT_OK:
        return None
    if result.signal is None and result.returncode == EXIT_UNSUPPORTED:
        return "frozen_content_unsupported"
    return "frozen_exit_abnormal"


def child_argv(interpreter: Path, bundle_root: Path, invocation_dir: Path) -> list[str]:
    return [str(interpreter), "-I", "-B", "-c", BOOTSTRAP, str(bundle_root), str(invocation_dir)]


def child_env(environment_root: Path) -> dict[str, str]:
    """The child's whole environment: a replacement, never a filtered copy of the supervisor's."""
    return {
        "PATH": str(environment_root / "bin"),
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
    }


def _write_durable(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _remove(directory: Path) -> None:
    try:
        directory.chmod(0o700)  # the sealed 0555 directory refuses unlinking its entries
    except OSError:
        pass
    shutil.rmtree(directory, ignore_errors=True)


def launch_child(
    *,
    interpreter: Path,
    bundle_root: Path,
    environment_root: Path,
    request_json: bytes,
    bars_arrow: bytes,
    invocations_root: Path,
    run: Runner,
) -> ContainedResult:
    """Seal one invocation directory, run the child over it, and remove the directory."""
    invocations_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="invocation-", dir=invocations_root))  # 0700
    try:
        for name, data in ((REQUEST_FILE, request_json), (BARS_FILE, bars_arrow)):
            _write_durable(directory / name, data)
            (directory / name).chmod(0o444)
        directory.chmod(0o555)
        try:
            return run(
                child_argv(interpreter, bundle_root, directory),
                cwd=bundle_root,
                env=child_env(environment_root),
                timeout=TIMEOUT_SECONDS,
                max_stdout=MAX_STDOUT_BYTES,
                stderr_capture=STDERR_CAPTURE_BYTES,
                grace=KILL_GRACE_SECONDS,
            )
        except OSError as exc:
            # `subprocess.Popen` re-raises a failed exec (or chdir into the bundle) in the child as
            # the child's errno with that path as `filename`; out of descriptors, fork or reaping
            # failures carry none, and are the supervisor's.
            started_at = (str(interpreter), str(bundle_root))
            if exc.filename is None or str(exc.filename) not in started_at:
                raise
            raise LaunchFailure(str(exc)) from exc
    finally:
        _remove(directory)
