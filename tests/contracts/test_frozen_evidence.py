from __future__ import annotations

from dataclasses import replace

import pytest

from algua.contracts.frozen_evidence import FrozenAttempt

SUCCESS_A = FrozenAttempt(
    deployment_id=1, request_id="0" * 32, phase="a", phase_a_invocation_id=None,
    snapshot_id="snap", bars_start="2026-09-01T00:00:00+00:00",
    bars_end="2026-09-30T00:00:00+00:00", request_json="{}", request_sha256="a" * 64,
    bars_sha256="b" * 64, phase_a_binding="c" * 64, result_kind="snapshot_required",
    result_sha256="d" * 64, failure_code=None, returncode=0, signal=None, timed_out=False,
    stdout_exceeded=False, stderr_truncated=False, diagnostic=None,
    started_at="2026-09-30T10:00:00+00:00", ended_at="2026-09-30T10:00:01+00:00",
)


def test_a_well_formed_success_and_failure_are_accepted():
    failure = replace(SUCCESS_A, result_kind=None, result_sha256=None,
                      failure_code="frozen_timeout", returncode=None, signal=15, timed_out=True,
                      diagnostic="exit_status=None signal=15")
    assert failure.failure_code == "frozen_timeout"
    success_b = replace(SUCCESS_A, phase="b", phase_a_invocation_id=7, result_kind="decision")
    assert success_b.phase == "b"


@pytest.mark.parametrize("change", [
    {"failure_code": "frozen_timeout"},                          # both success and failure
    {"result_sha256": None, "result_kind": None},                # neither
    {"result_kind": None},                                       # success without a kind
    {"result_kind": "planner_rejected"},                         # a refusal is not a success kind
    {"result_kind": "decision"},                                 # a phase a child never decides
    {"phase": "b", "phase_a_invocation_id": 3},                  # a phase b child never snapshots
    {"phase": "b"},                                              # phase b without its phase a
    {"phase_a_invocation_id": 3},                                # phase a naming a phase a
    {"request_sha256": None},                                    # bytes without their digest
    {"diagnostic": "x"},                                         # diagnostic on a success
])
def test_inconsistent_attempts_are_refused(change):
    with pytest.raises(ValueError):
        replace(SUCCESS_A, **change)
