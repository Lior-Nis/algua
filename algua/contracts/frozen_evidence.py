"""One frozen planner attempt as permanent evidence (Story 1.3d).

The live frozen port builds a :class:`FrozenAttempt` once the supervisor has fully judged an
attempt; the registry persists it as one append-only ``frozen_invocations`` row. Keeping the value
here, pure and stdlib-only, lets both sides share it without the live layer importing the
registry. Identities beyond ``deployment_id`` follow from immutable registry rows and are never
copied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: Result kinds a successful attempt may carry (the Story 1.3c result vocabulary, minus refusals).
ATTEMPT_RESULT_KINDS = frozenset({
    "early_no_decision", "snapshot_required", "risk_failure", "late_no_decision", "decision",
})


@dataclass(frozen=True)
class FrozenAttempt:
    """Every column of one ``frozen_invocations`` row except its id.

    Exactly one of ``result_sha256`` (a success) and ``failure_code`` (the stable Story 1.3c tenant
    failure the supervisor raised) is set. ``request_json`` / ``request_sha256`` / ``bars_sha256``
    are ``None`` only for an attempt refused before its request could be encoded. A phase ``"b"``
    attempt names the successful phase ``"a"`` attempt of the same tick.
    """

    deployment_id: int
    request_id: str
    phase: Literal["a", "b"]
    phase_a_invocation_id: int | None
    snapshot_id: str
    bars_start: str | None
    bars_end: str | None
    request_json: str | None
    request_sha256: str | None
    bars_sha256: str | None
    phase_a_binding: str | None
    result_kind: str | None
    result_sha256: str | None
    failure_code: str | None
    returncode: int | None
    signal: int | None
    timed_out: bool
    stdout_exceeded: bool
    stderr_truncated: bool
    diagnostic: str | None
    started_at: str
    ended_at: str

    def __post_init__(self) -> None:
        if (self.result_sha256 is None) == (self.failure_code is None):
            raise ValueError("an attempt is either a success or a failure")
        if (self.result_sha256 is None) != (self.result_kind is None):
            raise ValueError("a successful attempt names its result kind")
        if self.result_kind is not None and self.result_kind not in ATTEMPT_RESULT_KINDS:
            raise ValueError(f"unknown attempt result kind {self.result_kind!r}")
        if (self.phase == "b") != (self.phase_a_invocation_id is not None):
            raise ValueError("a phase b attempt, and only it, names its phase a attempt")
        if (self.request_json is None) != (self.request_sha256 is None):
            raise ValueError("request bytes and their digest are recorded together")
        if self.diagnostic is not None and self.failure_code is None:
            raise ValueError("only a failed attempt carries a diagnostic")
