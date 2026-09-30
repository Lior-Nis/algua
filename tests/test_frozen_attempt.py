"""Story 1.3d contract §1, §3: every judged frozen attempt leaves exactly one evidence record.

`FrozenPlanner` records through its injected `record` once the supervisor has fully judged an
attempt. These tests drive the port against a fake child runner, collect the `FrozenAttempt`s it
builds, and hash the sealed invocation files and the child's stdout independently: what counts as
an attempt (every entry into the invocation step, pre-launch refusals included), what does not (a
phase the supervisor settles itself, a systemic exception), when a row is written (after the §7
cross-checks) and what it carries (digests of the exact bytes, the ids `record` returned).
"""

from __future__ import annotations

import errno
import hashlib
import os
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from algua.contracts.frozen_evidence import FrozenAttempt
from algua.live import frozen_invocation
from algua.live.frozen_dispatch import FrozenTenantFailure
from algua.live.frozen_wire import MAX_DIAGNOSTIC_BYTES, STDERR_CAPTURE_BYTES
from algua.live.frozen_wire_result import PlannerRejected, encode_result
from algua.live.planner import InProcessPlanner
from algua.live.planner_contract import (
    Decision,
    EarlyNoDecision,
    LateNoDecision,
    PlannerRiskFailure,
    SnapshotRequired,
    VenueBeliefRequired,
)
from tests._frozen_harness import (
    DEPLOYMENT_ID,
    NOW,
    REQUEST_ID,
    early_input,
    fixture_bars,
    in_process,
    late_input,
    overlaid,
)
from tests.test_frozen_dispatch import (
    BARS_END,
    BARS_START,
    LATE_BREACHES,
    PRINTABLE,
    SNAPSHOT_ID,
    FakeRun,
    Recorder,
    _bundle,
    _consistent,
    _early_breach,
    _ended,
    _exec_fails,
    _expected_decision,
    _forged,
    _in_process,
    _late_case,
    _ok,
    _pending,
    _target,
    _warm,
    frozen_port,
)


@pytest.fixture
def world(tmp_path):
    """A fake bundle and a factory for one tick's port, its fake runner and its recorder."""
    bundle = _bundle(tmp_path)
    environment = tmp_path / "environment"

    def planner(*answers: Any, target=None, record=None):
        run, recorder = FakeRun(*answers), record or Recorder()
        port = frozen_port(target or _target(bundle, environment), tmp_path / "invocations",
                           run=run, record=recorder)
        return port, run, recorder

    return SimpleNamespace(bundle=bundle, environment=environment, planner=planner)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decision_tick():
    """``(early, late, reference)`` for a clean tick that plans (Phase B is a decision)."""
    strategy = in_process()
    early = early_input(strategy)
    return early, late_input(strategy, early, "disabled"), _in_process()


def _assert_tick(attempt: FrozenAttempt, phase: str) -> None:
    """The columns every attempt of the tick shares, and its UTC start/end times."""
    assert (attempt.deployment_id, attempt.request_id, attempt.phase) == (
        DEPLOYMENT_ID, REQUEST_ID, phase)
    assert (attempt.snapshot_id, attempt.bars_start, attempt.bars_end) == (
        SNAPSHOT_ID, BARS_START, BARS_END)
    started = datetime.fromisoformat(attempt.started_at)
    ended = datetime.fromisoformat(attempt.ended_at)
    assert started.utcoffset() == ended.utcoffset() == timedelta(0)
    assert started <= ended


def _assert_sent(attempt: FrozenAttempt, call: dict[str, Any]) -> None:
    """The request text and both digests are of the exact bytes the child was given."""
    request, bars = call["files"]["request.json"][1], call["files"]["bars.arrow"][1]
    assert attempt.request_json == request.decode("utf-8")
    assert (attempt.request_sha256, attempt.bars_sha256) == (_sha(request), _sha(bars))


def _assert_unsent(attempt: FrozenAttempt) -> None:
    assert (attempt.request_json, attempt.request_sha256, attempt.bars_sha256) == (None, None, None)


def _assert_exit(attempt: FrozenAttempt, ended: Any) -> None:
    """Exit metadata is the child's, or none at all when no child ran."""
    expected = (None, None, False, False, False) if ended is None else (
        ended.returncode, ended.signal, ended.timed_out, ended.stdout_exceeded,
        ended.stderr_truncated)
    assert (attempt.returncode, attempt.signal, attempt.timed_out, attempt.stdout_exceeded,
            attempt.stderr_truncated) == expected


def _assert_success(attempt: FrozenAttempt, kind: str, stdout: bytes) -> None:
    assert (attempt.result_kind, attempt.result_sha256) == (kind, _sha(stdout))
    assert (attempt.failure_code, attempt.diagnostic) == (None, None)


def _assert_failure(attempt: FrozenAttempt, raised: FrozenTenantFailure) -> None:
    assert (attempt.result_kind, attempt.result_sha256) == (None, None)
    assert (attempt.failure_code, attempt.diagnostic) == (raised.code, raised.diagnostic)
    assert attempt.diagnostic is not None
    assert len(attempt.diagnostic) <= MAX_DIAGNOSTIC_BYTES and set(attempt.diagnostic) <= PRINTABLE


def _raises(call) -> FrozenTenantFailure:
    with pytest.raises(FrozenTenantFailure) as raised:
        call()
    return raised.value


# --- a normal tick ------------------------------------------------------------------------------


def test_a_decision_tick_records_its_phase_a_then_its_phase_b_attempt(world):
    early, late, reference = _decision_tick()
    snapshot, decision = reference.phase_a(early), reference.phase_b(late)
    assert isinstance(snapshot, SnapshotRequired) and isinstance(decision, Decision)
    stdout_a, stdout_b = encode_result("a", REQUEST_ID, snapshot), encode_result("b", REQUEST_ID,
                                                                                 decision)
    recorded_at_phase_b_launch: list[int] = []

    def phase_b_child(argv):
        recorded_at_phase_b_launch.append(len(recorder.attempts))
        return _ended(0, stdout=stdout_b)

    port, run, recorder = world.planner(_ended(0, stdout=stdout_a), phase_b_child)

    assert port.phase_a(early) == snapshot
    assert len(recorder.attempts) == 1  # Phase A is recorded when phase_a returns
    pd.testing.assert_frame_equal(port.closed_bars(early), reference.closed_bars(early))
    assert port.phase_b(_pending(late)) == VenueBeliefRequired()  # settled without a child
    assert port.final_invocation_id is None
    assert port.phase_b(late) == decision

    first, second = recorder.attempts
    assert recorded_at_phase_b_launch == [1]
    _assert_tick(first, "a")
    _assert_sent(first, run.calls[0])
    _assert_exit(first, _ended(0, stdout=stdout_a))
    _assert_success(first, "snapshot_required", stdout_a)
    assert first.phase_a_invocation_id is None
    assert first.phase_a_binding == snapshot.phase_a_binding  # the binding Phase A produced
    _assert_tick(second, "b")
    _assert_sent(second, run.calls[1])
    _assert_exit(second, _ended(0, stdout=stdout_b))
    _assert_success(second, "decision", stdout_b)
    assert second.phase_a_invocation_id == recorder.ids[0]  # the id record returned for A
    assert second.phase_a_binding == late.phase_a_binding == snapshot.phase_a_binding  # received
    assert port.final_invocation_id == recorder.ids[1]


def test_an_early_no_decision_is_one_successful_phase_a_attempt_without_a_final(world):
    empty = early_input(in_process(), empty=True)
    expected = _in_process().phase_a(empty)
    assert isinstance(expected, EarlyNoDecision)
    stdout = encode_result("a", REQUEST_ID, expected)
    port, run, recorder = world.planner(_ended(0, stdout=stdout))

    assert port.phase_a(empty) == expected

    (attempt,) = recorder.attempts
    _assert_tick(attempt, "a")
    _assert_sent(attempt, run.calls[0])
    _assert_success(attempt, "early_no_decision", stdout)
    assert (attempt.phase_a_binding, attempt.phase_a_invocation_id) == (None, None)
    assert port.final_invocation_id is None


def _warming_late_no_decision() -> tuple[Any, ...]:
    """A warming snapshot and its late no-decision: ``(early, late, execution, snap, answer)``."""
    warm = _warm()
    early = early_input(warm)
    reference = InProcessPlanner(overlaid(warm))
    first = reference.phase_a(early)
    assert isinstance(first, SnapshotRequired) and first.warming
    late = late_input(warm, early, "disabled")
    answer = reference.phase_b(late)
    assert isinstance(answer, LateNoDecision)
    return early, late, overlaid(warm).execution, first, answer


@pytest.mark.parametrize(
    "case,kind",
    [("decision", "decision"), ("weight_breach", "decision"),
     ("child_breach", "risk_failure"), ("late_no_decision", "late_no_decision")],
)
def test_every_successful_phase_b_attempt_is_recorded_as_the_final_invocation(world, case, kind):
    # A decision the supervisor's weight-rule rerun turns into a breach is still a successful
    # attempt: it decoded and passed every §7 cross-check; the breach is the kill-switch path's.
    early, late, reference = _decision_tick()
    target, snapshot = None, reference.phase_a(early)
    if case == "late_no_decision":
        early, late, execution, snapshot, child = _warming_late_no_decision()
        target = _target(world.bundle, world.environment, execution=execution)
    else:
        child = {"decision": _expected_decision(),
                 "weight_breach": _consistent({"AAA": 0.6, "BBB": 0.6}),
                 "child_breach": PlannerRiskFailure("long_only", "a negative weight", False),
                 }[case]
    stdout = encode_result("b", REQUEST_ID, child)
    port, _, recorder = world.planner(_ok("a", snapshot), _ended(0, stdout=stdout),
                                      target=target)
    port.phase_a(early)

    answer = port.phase_b(late)

    if case == "weight_breach":
        assert isinstance(answer, PlannerRiskFailure) and answer.kind == "gross_exposure"
    _, second = recorder.attempts
    _assert_success(second, kind, stdout)
    assert second.phase_a_invocation_id == recorder.ids[0]
    assert port.final_invocation_id == recorder.ids[1]


@pytest.mark.parametrize("case", ["no_newline", "unsanitized_detail"])
def test_the_result_digest_is_of_the_accepted_stdout_bytes_never_a_re_encoding(world, case):
    early, late, reference = _decision_tick()
    if case == "no_newline":  # the strict decoder accepts the document without its newline
        stdout = encode_result("b", REQUEST_ID, reference.phase_b(late))[:-1]
    else:  # the supervisor sanitizes a child's breach detail before using it
        stdout = encode_result("b", REQUEST_ID,
                               PlannerRiskFailure("gross_exposure", "\x1b[31mred\x00", False))
    port, _, recorder = world.planner(_ok("a", reference.phase_a(early)), _ended(0, stdout=stdout))
    port.phase_a(early)

    answer = port.phase_b(late)

    _, second = recorder.attempts
    assert second.result_sha256 == _sha(stdout)
    assert second.result_sha256 != _sha(encode_result("b", REQUEST_ID, answer))


# --- failed attempts ----------------------------------------------------------------------------


def _forged_binding() -> SnapshotRequired:
    snapshot = _in_process().phase_a(early_input(in_process()))
    assert isinstance(snapshot, SnapshotRequired)
    return replace(snapshot, phase_a_binding="f" * 64)


#: case -> how the Phase A child answers (or fails to start); each is one failed attempt.
PHASE_A_FAILURES = {
    "launch_failed": lambda: _exec_fails(FileNotFoundError, errno.ENOENT),
    "timeout": lambda: _ended(None, 9, timed_out=True, stderr=b"slow"),
    "output_exceeded": lambda: _ended(None, 15, stdout=b"{", stdout_exceeded=True),
    "signal": lambda: _ended(None, 11, stderr=b"Fatal Python error"),
    "exit_1_truncated": lambda: _ended(1, stderr=b"\x1bz" * (STDERR_CAPTURE_BYTES // 2),
                                       stderr_truncated=True),
    "exit_3": lambda: _ended(3, stderr=b"frozen_child: refused: nope\n"),
    "garbage": lambda: _ended(0, stdout=b"not json"),
    "child_rejected": lambda: _ok("a", PlannerRejected("invalid_now", "bad\x00 now")),
    "forged_binding": lambda: _ok("a", _forged_binding()),  # decodes; fails the cross-check
    "false_breach": lambda: _ok("a", PlannerRiskFailure("stale_marks", "forged", True)),
}
PHASE_A_CODES = {
    "launch_failed": "frozen_launch_failed", "timeout": "frozen_timeout",
    "output_exceeded": "frozen_output_exceeded", "signal": "frozen_exit_abnormal",
    "exit_1_truncated": "frozen_exit_abnormal", "exit_3": "frozen_content_unsupported",
    "garbage": "frozen_result_invalid", "child_rejected": "frozen_planner_rejected",
    "forged_binding": "frozen_result_invalid", "false_breach": "frozen_result_invalid",
}


@pytest.mark.parametrize("case", sorted(PHASE_A_FAILURES))
def test_a_failed_phase_a_attempt_is_one_failure_row_with_its_code_and_diagnostic(world, case):
    answer = PHASE_A_FAILURES[case]()
    port, run, recorder = world.planner(answer)

    raised = _raises(lambda: port.phase_a(early_input(in_process())))

    assert raised.code == PHASE_A_CODES[case]
    (attempt,) = recorder.attempts  # recorded once, after judgement: never also as a success
    _assert_tick(attempt, "a")
    _assert_failure(attempt, raised)
    _assert_sent(attempt, run.calls[0])
    _assert_exit(attempt, None if callable(answer) else answer)
    assert (attempt.phase_a_binding, attempt.phase_a_invocation_id) == (None, None)
    assert port.final_invocation_id is None
    if case == "exit_1_truncated":
        assert len(attempt.diagnostic or "") == MAX_DIAGNOSTIC_BYTES


#: case -> how the Phase B child answers; each is one failed attempt after a successful Phase A.
PHASE_B_FAILURES = {
    "exit_1": (lambda: _ended(1, stderr=b"Traceback"), "frozen_exit_abnormal"),
    "timeout": (lambda: _ended(None, 9, timed_out=True), "frozen_timeout"),
    "child_rejected": (lambda: _ok("b", PlannerRejected("invalid_captured_state", "no")),
                       "frozen_planner_rejected"),
    "forged_decision": (lambda: _ok("b", _forged("intents_vs_weights")), "frozen_result_invalid"),
    "forged_state": (lambda: _ok("b", _forged("peak_equity")), "frozen_result_invalid"),
    "false_breach": (lambda: _ok("b", PlannerRiskFailure("reconcile", "forged", False)),
                     "frozen_result_invalid"),
}


@pytest.mark.parametrize("case", sorted(PHASE_B_FAILURES))
def test_a_failed_phase_b_attempt_is_recorded_and_leaves_no_final_invocation(world, case):
    answer, code = PHASE_B_FAILURES[case]
    early, late, reference = _decision_tick()
    ended = answer()
    port, run, recorder = world.planner(_ok("a", reference.phase_a(early)), ended)
    port.phase_a(early)

    raised = _raises(lambda: port.phase_b(late))

    assert raised.code == code
    first, second = recorder.attempts
    assert first.result_kind == "snapshot_required"
    _assert_tick(second, "b")
    _assert_failure(second, raised)
    _assert_sent(second, run.calls[1])
    _assert_exit(second, ended)
    assert second.phase_a_invocation_id == recorder.ids[0]
    assert second.phase_a_binding == late.phase_a_binding  # the binding it received
    assert port.final_invocation_id is None


@pytest.mark.parametrize("case", ["content_unsupported", "request_too_large", "unencodable_input"])
def test_an_attempt_refused_before_launch_is_recorded_without_request_bytes(world, tmp_path,
                                                                          case):
    early = early_input(in_process())
    target, code = None, {"content_unsupported": "frozen_content_unsupported",
                          "request_too_large": "frozen_request_too_large",
                          "unencodable_input": "frozen_planner_rejected"}[case]
    if case == "content_unsupported":  # the bundle carries no protocol stamp
        target = _target(_bundle(tmp_path / "other", protocol=None), world.environment)
    elif case == "request_too_large":  # flat names: past the §6 request bound
        early = replace(early, early_positions={
            f"S{index:05d}{'X' * 40}": 0.0 for index in range(9_000)})
    else:  # valid for the planner, but not a wire-v1 value
        early = replace(early, now=pd.Timestamp(NOW) + pd.Timedelta(nanoseconds=1))
    port, run, recorder = world.planner(target=target)

    raised = _raises(lambda: port.phase_a(early))

    assert raised.code == code and run.calls == []
    (attempt,) = recorder.attempts
    _assert_tick(attempt, "a")
    _assert_failure(attempt, raised)
    _assert_unsent(attempt)
    _assert_exit(attempt, None)


# --- not attempts: what the supervisor settles itself, and systemic faults ----------------------


@pytest.mark.parametrize("case", ["stale_marks", "no_mark", "gate_universe", "request_id"])
def test_a_phase_a_the_supervisor_settles_without_a_child_records_nothing(world, case):
    early = early_input(in_process())
    target = None
    if case in ("stale_marks", "no_mark"):  # the supervisor's own breach
        early = _early_breach(case)
    elif case == "gate_universe":  # the supervisor refuses its own input
        target = _target(world.bundle, world.environment, gate=("AAA",))
    else:
        early = replace(early, request_id="xyz")
    port, run, recorder = world.planner(target=target)

    if case in ("stale_marks", "no_mark"):
        assert isinstance(port.phase_a(early), PlannerRiskFailure)
    else:
        assert _raises(lambda: port.phase_a(early)).code == "frozen_planner_rejected"

    assert run.calls == [] and recorder.attempts == []


@pytest.mark.parametrize("case", ["pending_belief", *LATE_BREACHES, "binding_refused"])
def test_a_phase_b_the_supervisor_settles_without_a_child_records_only_phase_a(world, case):
    if case in LATE_BREACHES:
        early, late = _late_case(case)
    else:
        early, late, _ = _decision_tick()
    reference = _in_process()
    port, run, recorder = world.planner(_ok("a", reference.phase_a(early)),
                                        _ok("b", _expected_decision()))
    port.phase_a(early)

    if case == "pending_belief":
        assert port.phase_b(_pending(late)) == VenueBeliefRequired()
    elif case == "binding_refused":  # the supervisor's own run of the planner's binding check
        refused = replace(late, phase_a_binding="0" * 64)
        assert _raises(lambda: port.phase_b(refused)).code == "frozen_planner_rejected"
    else:
        assert port.phase_b(late) == reference.phase_b(late)

    assert len(run.calls) == 1
    assert [attempt.result_kind for attempt in recorder.attempts] == ["snapshot_required"]
    assert port.final_invocation_id is None


def test_a_closed_bars_rejection_after_phase_a_records_only_the_phase_a_success(world):
    early, _, reference = _decision_tick()
    port, _, recorder = world.planner(_ok("a", reference.phase_a(early)))
    port.phase_a(early)
    older = fixture_bars()
    shifted = replace(early, raw_bars=older[older.index < datetime(2023, 1, 5, tzinfo=UTC)])

    assert _raises(lambda: port.closed_bars(shifted)).code == "frozen_result_invalid"

    (attempt,) = recorder.attempts
    assert (attempt.phase, attempt.result_kind) == ("a", "snapshot_required")
    assert port.final_invocation_id is None


SYSTEMIC = {
    "interrupt": KeyboardInterrupt,
    "supervisor_bug": lambda: RuntimeError("supervisor bug"),
    "out_of_descriptors": lambda: OSError(errno.EMFILE, "Too many open files"),
}


@pytest.mark.parametrize("phase", ["a", "b"])
@pytest.mark.parametrize("fault", sorted(SYSTEMIC))
def test_a_systemic_exception_during_an_attempt_records_nothing(world, fault, phase):
    early, late, reference = _decision_tick()
    raised = SYSTEMIC[fault]()
    answers = [raised] if phase == "a" else [_ok("a", reference.phase_a(early)), raised]
    port, _, recorder = world.planner(*answers)
    if phase == "b":
        port.phase_a(early)

    with pytest.raises(type(raised)):
        if phase == "a":
            port.phase_a(early)
        else:
            port.phase_b(late)

    assert [attempt.phase for attempt in recorder.attempts] == ([] if phase == "a" else ["a"])
    assert port.final_invocation_id is None


def test_an_invocation_directory_fault_is_systemic_and_records_nothing(world, monkeypatch):
    def fault(*args, **kwargs):
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(frozen_invocation.tempfile, "mkdtemp", fault)
    port, run, recorder = world.planner()

    with pytest.raises(OSError):
        port.phase_a(early_input(in_process()))

    assert run.calls == [] and recorder.attempts == []


class _FailingRecorder(Recorder):
    def __call__(self, attempt: FrozenAttempt) -> int:
        super().__call__(attempt)
        raise sqlite3.OperationalError("database is locked")


@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_a_recorder_error_is_systemic_and_replaces_the_tenant_outcome(world, outcome):
    # A SQLite error while recording would affect every tenant (Story 1.3d §3): it propagates in
    # place of the attempt's own result or tenant failure, and the port keeps no id.
    early, _, reference = _decision_tick()
    answer = _ok("a", reference.phase_a(early)) if outcome == "success" else _ended(1)
    port, _, recorder = world.planner(answer, record=_FailingRecorder())

    with pytest.raises(sqlite3.OperationalError):
        port.phase_a(early)

    assert len(recorder.attempts) == 1
    with pytest.raises(RuntimeError, match="Phase A"):  # no snapshot was kept for Phase B
        port.phase_b(late_input(in_process(), early, "disabled"))
