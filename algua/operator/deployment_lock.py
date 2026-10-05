"""Operator-lock serialization for retirement and every paper-lane exit."""
from __future__ import annotations

import os
import socket
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from algua.contracts.lifecycle import Stage, TransitionError
from algua.operator.schedule import OperatorLockHeld, operator_run_lock

#: The stages ``paper run-all`` ticks and the paper account reconcile counts.
_PAPER_LANE = frozenset({Stage.PAPER, Stage.FORWARD_TESTED})


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
def operator_transition_lock(from_stage: Stage, target: Stage) -> Iterator[None]:
    """Serialize retirement and every paper-lane exit with the paper operator's broker/tick
    critical section (Story 2.2 §2.5): without it, a timer-driven cycle whose pre-tick stage check
    passed could submit for the strategy after its exit commits (#685's orphan).

    The lock is ``operator.lock`` of the checkout whose code runs, taken non-blocking: a held lock
    refuses the transition at once (never waits, never deadlocks a nested acquire), before any exit
    drain is selected. It excludes the paper timer only when run from the operator's checkout. The
    edges locked before Story 2.2 (``paper -> candidate`` and every ``-> retired``) keep their
    message byte for byte; ``paper -> dormant`` and ``forward_tested -> live`` get their own."""
    retirement = target is Stage.RETIRED or (
        from_stage is Stage.PAPER and target is Stage.CANDIDATE)
    lane_exit = from_stage in _PAPER_LANE and target not in _PAPER_LANE
    if not (retirement or lane_exit):
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
            if retirement else
            "operator.lock is held; a paper-lane exit cannot interleave with a paper tick"
        ) from exc
