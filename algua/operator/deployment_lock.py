"""Operator-lock serialization for lifecycle changes that retire deployments."""
from __future__ import annotations

import os
import socket
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from algua.contracts.lifecycle import Stage, TransitionError
from algua.operator.schedule import OperatorLockHeld, operator_run_lock


def _operator_lock_path() -> Path:
    out = subprocess.run(  # noqa: S603,S607 — fixed argv, no shell
        ["git", "rev-parse", "--absolute-git-dir"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip()) / "operator.lock"


@contextmanager
def deployment_retirement_lock(from_stage: Stage, target: Stage) -> Iterator[None]:
    """Serialize retirement with the paper operator's broker/tick critical section."""
    required = (from_stage is Stage.PAPER and target is Stage.CANDIDATE) or target is Stage.RETIRED
    if not required:
        yield
        return
    try:
        with operator_run_lock(
            _operator_lock_path(), job="registry-transition", host=socket.gethostname(),
            pid=os.getpid(),
        ):
            yield
    except OperatorLockHeld as exc:
        raise TransitionError(
            "operator.lock is held; deployment retirement cannot interleave with a paper tick"
        ) from exc
