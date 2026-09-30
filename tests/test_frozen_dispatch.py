"""Story 1.3c contract §3–§8: the supervisor side of the frozen planner (`frozen_dispatch`).

The fast tests drive `FrozenPlanner` with a fake `run` that answers crafted `ContainedResult`s and
inspects the sealed invocation directory while the "child" runs. The end-to-end tests launch the
real child from a real bundle through the shared harness and require the frozen port to be
indistinguishable from the in-process planner: equal canonical results per phase, and an equal
effect trace for a whole `run_tick`.
"""

from __future__ import annotations

import errno
import os
import stat
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from algua.contracts.types import OrderIntent, Side
from algua.execution.alpaca_broker import TickSnapshot
from algua.live import frozen_invocation
from algua.live.frozen_dispatch import (
    FROZEN_FAILURE_CODES,
    FrozenPlanner,
    FrozenTarget,
    FrozenTenantFailure,
)
from algua.live.frozen_invocation import sanitize_text
from algua.live.frozen_wire import (
    BOOTSTRAP,
    KILL_GRACE_SECONDS,
    MAX_DIAGNOSTIC_BYTES,
    MAX_STDOUT_BYTES,
    STDERR_CAPTURE_BYTES,
    TIMEOUT_SECONDS,
    WireIdentity,
    encode_request,
)
from algua.live.frozen_wire_result import PlannerRejected, encode_result
from algua.live.live_loop import PlannerContext, TickHooks, run_tick
from algua.live.planner import InProcessPlanner
from algua.live.planner_binding import bars_digest, phase_a_binding
from algua.live.planner_contract import (
    Decision,
    EarlyNoDecision,
    LateNoDecision,
    PlannerRiskFailure,
    PlannerState,
    SnapshotRequired,
    VenueBeliefEnabled,
    VenueBeliefPending,
    VenueBeliefRequired,
)
from algua.primitives.contained_process import ContainedResult
from algua.risk.limits import DARK_FEED_KINDS, RISK_BREACH_KINDS, RiskBreach
from algua.strategies.base import config_hash
from tests._frozen_harness import (
    ARTIFACT_ID,
    CALENDAR,
    DEPLOYMENT_ID,
    ENV_DIGEST,
    GATE,
    MANIFEST,
    NOW,
    REQUEST_ID,
    STAMP,
    STRATEGY,
    digest,
    early_input,
    fixture_bars,
    frozen_store,
    in_process,
    late_input,
    overlaid,
    protocol_bytes,
    recorded_json,
)

IDENTITY = WireIdentity(STRATEGY, DEPLOYMENT_ID, ARTIFACT_ID, MANIFEST, digest("good"), ENV_DIGEST)
DAY = timedelta(days=1)
PRINTABLE = {chr(code) for code in range(0x20, 0x7F)}


def _strategy():
    return overlaid(in_process())


def _in_process() -> InProcessPlanner:
    return InProcessPlanner(_strategy())


def _pending(late):
    return replace(late, captured=replace(late.captured, venue_belief=VenueBeliefPending()))


#: The rules only a strategy's weights can break (`validate_decision_weights`); every other breach
#: kind is a strategy-free wall the supervisor re-derives itself (§7).
DECISION_KINDS = frozenset(
    {"gross_exposure", "long_only", "max_weight_per_symbol", "non_finite_weight",
     "out_of_universe"})
STRATEGY_FREE_KINDS = RISK_BREACH_KINDS - DECISION_KINDS
LATE_BREACHES = ("drawdown", "gross_exposure_realized", "non_positive_equity", "reconcile",
                 "stale_marks")
PRE_BELIEF = ("drawdown", "non_positive_equity")  # what Phase B checks before it asks for a belief


def _aged(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """``symbol``'s marks three weeks old: a dark feed for that name."""
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(
        [ts - 21 * DAY if sym == symbol else ts
         for ts, sym in zip(frame.index, frame.symbol, strict=True)], name="timestamp")
    return frame


def _late_case(case: str):
    """``(early, late)`` for a clean Phase B, or one breaking the named strategy-free wall."""
    strategy = in_process()
    early = early_input(strategy)
    if case == "stale_marks":  # a gate-universe name's marks are stale, the held book's fresh
        early = replace(early, raw_bars=_aged(early.raw_bars, "AAA"))
    late = late_input(strategy, early, "drawdown" if case == "drawdown" else "disabled")
    captured = late.captured
    if case == "reconcile":
        captured = replace(captured, venue_belief=VenueBeliefEnabled({"OLD": 3.0}))
    elif case == "gross_exposure_realized":  # OLD worth 200 on an equity of 100
        captured = replace(captured, market_values={"OLD": 200.0})
    elif case == "non_positive_equity":
        captured = replace(captured, sizing_equity=0.0)
    return early, replace(late, captured=captured)


# --- a fake bundle and a fake contained runner --------------------------------------------------


def _bundle(root: Path, *, protocol: bytes | None = STAMP, child: bool = True) -> Path:
    bundle = root / "bundle"
    (bundle / "_algua").mkdir(parents=True)
    if protocol is not None:
        (bundle / "_algua/protocol.json").write_bytes(protocol)
    if child:
        (bundle / "algua/live").mkdir(parents=True)
        (bundle / "algua/live/frozen_child.py").write_text("")
    return bundle


def _target(bundle: Path, environment: Path, *, gate: tuple[str, ...] = GATE,
            execution=None) -> FrozenTarget:
    return FrozenTarget(
        identity=IDENTITY,
        bundle_root=bundle,
        environment_root=environment,
        interpreter=environment / "bin/python",
        execution=_strategy().execution if execution is None else execution,
        gate_universe=gate,
    )


def _ok(phase: str, result: object, request_id: str = REQUEST_ID) -> ContainedResult:
    return ContainedResult(0, None, False, encode_result(phase, request_id, result), False, b"",
                           False)


def _ended(returncode: int | None = None, signal: int | None = None, *, timed_out: bool = False,
           stdout: bytes = b"", stdout_exceeded: bool = False, stderr: bytes = b"",
           stderr_truncated: bool = False) -> ContainedResult:
    return ContainedResult(returncode, signal, timed_out, stdout, stdout_exceeded, stderr,
                           stderr_truncated)


class FakeRun:
    """Answers each launch from a script and records what the child would have seen."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, argv, **kwargs):
        invocation = Path(argv[-1])
        self.calls.append({
            "argv": list(argv), **kwargs,
            "invocation": invocation,
            "dir_mode": stat.S_IMODE(invocation.stat().st_mode),
            "files": {path.name: (stat.S_IMODE(path.stat().st_mode), path.read_bytes())
                      for path in invocation.iterdir()},
        })
        answer = self.answers.pop(0)
        if callable(answer):  # an exception built from the argv, as Popen builds an exec failure
            answer = answer(argv)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _exec_fails(exc_type: type[OSError], code: int):
    """A launch whose exec fails, reported exactly as `subprocess.Popen` reports it: the child's
    errno and ``filename`` set to the executable (argv[0])."""
    return lambda argv: exc_type(code, os.strerror(code), argv[0])


@pytest.fixture
def world(tmp_path):
    """A fake bundle, an environment path, an invocations root and a planner factory."""
    bundle = _bundle(tmp_path)
    environment = tmp_path / "environment"
    invocations = tmp_path / "invocations"

    def planner(*answers: Any, target: FrozenTarget | None = None):
        run = FakeRun(*answers)
        port = FrozenPlanner(target or _target(bundle, environment),
                             invocations_root=invocations, run=run)
        return port, run

    return SimpleNamespace(bundle=bundle, environment=environment, invocations=invocations,
                           planner=planner)


def _leftovers(invocations: Path) -> list[str]:
    return sorted(path.name for path in invocations.iterdir()) if invocations.exists() else []


def _failure(call, code: str) -> FrozenTenantFailure:
    with pytest.raises(FrozenTenantFailure) as raised:
        call()
    assert raised.value.code == code, raised.value.diagnostic
    assert raised.value.deployment_id == DEPLOYMENT_ID
    return raised.value


# --- the failure type ---------------------------------------------------------------------------


def test_the_failure_vocabulary_is_the_ten_contract_codes():
    assert isinstance(FROZEN_FAILURE_CODES, frozenset)
    assert FROZEN_FAILURE_CODES == {
        "frozen_content_unavailable", "frozen_content_unsupported", "frozen_request_too_large",
        "frozen_launch_failed", "frozen_timeout", "frozen_exit_abnormal",
        "frozen_output_exceeded", "frozen_result_invalid", "frozen_planner_rejected",
        "frozen_live_unsupported",
    }


def test_a_failure_carries_its_code_deployment_and_a_sanitized_bounded_diagnostic():
    failure = FrozenTenantFailure(
        "frozen_timeout", 7, "a\x00b\nc\x1b[31md\x7fé " + "x" * (2 * MAX_DIAGNOSTIC_BYTES))

    assert (failure.code, failure.deployment_id) == ("frozen_timeout", 7)
    assert failure.diagnostic.startswith("abc[31md" + "x")
    assert len(failure.diagnostic.encode()) == MAX_DIAGNOSTIC_BYTES
    assert set(failure.diagnostic) <= PRINTABLE
    assert "frozen_timeout" in str(failure)
    with pytest.raises(ValueError, match="unknown frozen failure code"):
        FrozenTenantFailure("frozen_mystery", 7, "")


def test_sanitize_text_keeps_printable_ascii_only_and_bounds_it():
    assert sanitize_text("marks stale — refusing\x00\n") == "marks stale  refusing"
    assert sanitize_text(b"\xff\xc3\xa9\x1b[0mok\x7f\r\n") == "[0mok"
    assert sanitize_text("x" * (MAX_DIAGNOSTIC_BYTES + 5)) == "x" * MAX_DIAGNOSTIC_BYTES


# --- launch: the exact command, the sealed directory, cleanup -----------------------------------


def test_a_phase_launches_the_exact_contract_command_over_a_sealed_directory(world, monkeypatch):
    created: list[int] = []
    mkdtemp = frozen_invocation.tempfile.mkdtemp

    def recording_mkdtemp(*args, **kwargs):
        path = mkdtemp(*args, **kwargs)
        created.append(stat.S_IMODE(Path(path).stat().st_mode))
        return path

    monkeypatch.setattr(frozen_invocation.tempfile, "mkdtemp", recording_mkdtemp)
    synced: list[str] = []
    fsync = frozen_invocation.os.fsync

    def recording_fsync(fd):
        synced.append(Path(f"/proc/self/fd/{fd}").resolve().name)
        fsync(fd)

    monkeypatch.setattr(frozen_invocation.os, "fsync", recording_fsync)
    early = early_input(in_process())
    expected = _in_process().phase_a(early)
    port, run = world.planner(_ok("a", expected))

    assert port.phase_a(early) == expected

    (call,) = run.calls
    invocation = call["invocation"]
    assert invocation.parent == world.invocations
    assert call["argv"] == [str(world.environment / "bin/python"), "-I", "-B", "-c", BOOTSTRAP,
                            str(world.bundle), str(invocation)]
    assert call["cwd"] == world.bundle
    assert call["env"] == {"PATH": str(world.environment / "bin"), "HOME": "/nonexistent",
                           "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC"}
    assert (call["timeout"], call["max_stdout"], call["stderr_capture"], call["grace"]) == (
        TIMEOUT_SECONDS, MAX_STDOUT_BYTES, STDERR_CAPTURE_BYTES, KILL_GRACE_SECONDS)
    request, bars = encode_request("a", early, None, IDENTITY)
    assert call["dir_mode"] == 0o555
    assert call["files"] == {"request.json": (0o444, request), "bars.arrow": (0o444, bars)}
    assert created == [0o700]
    assert sorted(synced) == ["bars.arrow", "request.json"]
    assert _leftovers(world.invocations) == []


def test_phase_b_sends_the_late_request(world):
    strategy = in_process()
    early = early_input(strategy)
    late = late_input(strategy, early, "enabled")
    planner = _in_process()
    port, run = world.planner(_ok("a", planner.phase_a(early)), _ok("b", planner.phase_b(late)))

    port.phase_a(early)
    assert port.phase_b(late) == planner.phase_b(late)

    request, bars = encode_request("b", early, late, IDENTITY)
    assert run.calls[1]["files"] == {"request.json": (0o444, request), "bars.arrow": (0o444, bars)}


@pytest.mark.parametrize(
    "answer,error",
    [
        (_ok("a", PlannerRejected("invalid_now", "bad")), FrozenTenantFailure),
        (_ended(1), FrozenTenantFailure),
        (_ended(None, 9, timed_out=True), FrozenTenantFailure),
        (_ended(0, stdout=b"not json"), FrozenTenantFailure),
        (_exec_fails(FileNotFoundError, errno.ENOENT), FrozenTenantFailure),
        (RuntimeError("supervisor bug"), RuntimeError),
        (KeyboardInterrupt(), KeyboardInterrupt),
    ],
    ids=["rejected", "exit_1", "timeout", "garbage", "launch_oserror", "runtime_error",
         "interrupt"],
)
def test_the_invocation_directory_is_removed_on_every_path(world, answer, error):
    port, run = world.planner(answer)

    with pytest.raises(error):
        port.phase_a(early_input(in_process()))

    assert len(run.calls) == 1 and run.calls[0]["dir_mode"] == 0o555
    assert _leftovers(world.invocations) == []


def test_the_invocation_directory_is_removed_after_success(world):
    early = early_input(in_process())
    port, run = world.planner(_ok("a", _in_process().phase_a(early)))

    port.phase_a(early)

    assert not run.calls[0]["invocation"].exists()
    assert _leftovers(world.invocations) == []


# --- whose fault: a supervisor-side OSError is systemic, only an exec failure is the tenant's ----


@pytest.mark.parametrize("where", ["root", "mkdtemp", "write"])
@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EROFS, errno.EIO])
def test_an_invocation_directory_fault_is_systemic_not_the_tenants(world, monkeypatch, where,
                                                                  code):
    # data_dir full, read-only or failing: every tenant would hit it, so it must not be isolated
    # as this tenant's `frozen_launch_failed` (§8: only the interpreter failing to start is).
    early = early_input(in_process())
    port, run = world.planner(_ok("a", _in_process().phase_a(early)))

    def fault(*args, **kwargs):
        raise OSError(code, os.strerror(code))

    if where == "root":
        monkeypatch.setattr(frozen_invocation.Path, "mkdir", fault)
    elif where == "mkdtemp":
        monkeypatch.setattr(frozen_invocation.tempfile, "mkdtemp", fault)
    else:
        monkeypatch.setattr(frozen_invocation.os, "fsync", fault)

    with pytest.raises(OSError) as raised:
        port.phase_a(early)

    assert raised.value.errno == code
    assert run.calls == []
    monkeypatch.undo()
    assert _leftovers(world.invocations) == []


@pytest.mark.parametrize(
    "fault",
    [OSError(errno.EMFILE, "Too many open files"), ChildProcessError(errno.ECHILD, "No child")],
    ids=["emfile", "echild"],
)
def test_a_supervisor_side_fault_inside_the_launch_is_systemic(world, fault):
    # Out of descriptors while creating the pipes, or a lost child while reaping: the supervisor's
    # own state, not an interpreter that could not start.
    port, run = world.planner(fault)

    with pytest.raises(OSError) as raised:
        port.phase_a(early_input(in_process()))

    assert raised.value is fault and len(run.calls) == 1
    assert _leftovers(world.invocations) == []


@pytest.mark.parametrize("case", ["missing", "not_executable"])
def test_a_real_interpreter_that_cannot_start_is_the_tenants_launch_failure(tmp_path, case):
    bundle = _bundle(tmp_path)
    environment = tmp_path / "environment"
    if case == "not_executable":
        (environment / "bin").mkdir(parents=True)
        (environment / "bin/python").write_text("")  # mode 0644: exec fails with EACCES
    port = FrozenPlanner(_target(bundle, environment), invocations_root=tmp_path / "inv")

    failure = _failure(lambda: port.phase_a(early_input(in_process())), "frozen_launch_failed")

    assert str(environment / "bin/python") in failure.diagnostic
    assert _leftovers(tmp_path / "inv") == []


def test_an_unreadable_bundle_entry_is_unsupported_content_not_a_raw_error(tmp_path):
    bundle = _bundle(tmp_path)
    live = bundle / "algua/live"
    live.chmod(0)  # the entry module can no longer be stat'ed: EACCES, not "missing"
    try:
        if os.access(live / "frozen_child.py", os.F_OK):
            pytest.skip("running with privileges that ignore directory permissions")
        run = FakeRun()
        port = FrozenPlanner(_target(bundle, tmp_path / "env"),
                             invocations_root=tmp_path / "inv", run=run)

        failure = _failure(lambda: port.phase_a(early_input(in_process())),
                           "frozen_content_unsupported")

        assert "frozen_child.py" in failure.diagnostic and run.calls == []
    finally:
        live.chmod(0o755)


# --- process outcomes -> §8 codes ---------------------------------------------------------------


@pytest.mark.parametrize(
    "answer,code,fragments",
    [
        (_ended(None, 9, timed_out=True), "frozen_timeout", ["timed_out=true", "signal=9"]),
        (_ended(None, 9, timed_out=True, stdout_exceeded=True), "frozen_timeout",
         ["timed_out=true", "stdout_exceeded=true"]),
        (_ended(None, 15, stdout_exceeded=True), "frozen_output_exceeded",
         ["stdout_exceeded=true", "signal=15"]),
        (_ended(None, 11, stderr=b"Fatal Python error"), "frozen_exit_abnormal",
         ["signal=11", "Fatal Python error"]),
        (_ended(3, stderr=b"frozen_child: refused: nope\n"), "frozen_content_unsupported",
         ["exit_status=3", "frozen_child: refused: nope"]),
        (_ended(2, stderr=b"frozen_child: bad request"), "frozen_exit_abnormal",
         ["exit_status=2"]),
        (_ended(1, stderr=b"Traceback"), "frozen_exit_abnormal", ["exit_status=1", "Traceback"]),
        (_ended(0, stdout=b"{}"), "frozen_result_invalid", ["missing_key"]),
        (_ended(0, stdout=b""), "frozen_result_invalid", []),
        (_exec_fails(PermissionError, errno.EACCES), "frozen_launch_failed",
         ["Permission denied"]),
    ],
    ids=["timeout", "timeout_and_overflow", "output_exceeded", "signal", "exit_3", "exit_2",
         "exit_1", "bad_result", "no_result", "launch"],
)
def test_process_outcomes_map_to_the_contract_codes(world, answer, code, fragments):
    port, _ = world.planner(answer)

    failure = _failure(lambda: port.phase_a(early_input(in_process())), code)

    for fragment in fragments:
        assert fragment in failure.diagnostic


def test_a_result_echoing_another_request_is_invalid(world):
    early = early_input(in_process())
    port, _ = world.planner(_ok("a", _in_process().phase_a(early), request_id="f" * 32))

    assert "echo_mismatch" in _failure(lambda: port.phase_a(early),
                                       "frozen_result_invalid").diagnostic


def test_the_process_diagnostic_is_printable_ascii_bounded_and_flags_truncation(world):
    stderr = (b"line one\n\x00\x1b[2Jline\ttwo\xc3\xa9\r\n" + b"z" * STDERR_CAPTURE_BYTES)
    port, _ = world.planner(_ended(1, stderr=stderr[:STDERR_CAPTURE_BYTES], stderr_truncated=True))

    failure = _failure(lambda: port.phase_a(early_input(in_process())), "frozen_exit_abnormal")

    assert set(failure.diagnostic) <= PRINTABLE
    assert len(failure.diagnostic) <= MAX_DIAGNOSTIC_BYTES
    assert "stderr_truncated=true" in failure.diagnostic
    assert "line one[2Jlinetwo" in failure.diagnostic
    assert failure.diagnostic.endswith("z")


def test_a_stderr_head_cut_to_fit_the_diagnostic_is_flagged(world):
    port, _ = world.planner(_ended(1, stderr=b"w" * (MAX_DIAGNOSTIC_BYTES + 1)))

    failure = _failure(lambda: port.phase_a(early_input(in_process())), "frozen_exit_abnormal")

    assert "stderr_truncated=true" in failure.diagnostic
    assert len(failure.diagnostic) == MAX_DIAGNOSTIC_BYTES


def test_a_short_stderr_is_not_flagged_truncated(world):
    port, _ = world.planner(_ended(1, stderr=b"boom"))

    failure = _failure(lambda: port.phase_a(early_input(in_process())), "frozen_exit_abnormal")

    assert "stderr_truncated=false" in failure.diagnostic and failure.diagnostic.endswith("boom")


# --- refusals before any launch -----------------------------------------------------------------


@pytest.mark.parametrize(
    "protocol,child",
    [
        (None, True),
        (b"not json", True),
        (b"[]", True),
        (protocol_bytes(frozen_wire={"name": "frozen-planner", "version": 2}), True),
        (protocol_bytes(frozen_wire={"name": "other-planner", "version": 1}), True),
        (protocol_bytes(planner_boundary_version=2), True),
        (protocol_bytes(planner_boundary_version=True), True),
        (STAMP, False),
    ],
    ids=["no_protocol", "not_json", "not_object", "wire_v2", "wire_name", "boundary_v2",
         "boundary_bool", "no_child"],
)
def test_unsupported_content_is_refused_before_any_launch(tmp_path, protocol, child):
    bundle = _bundle(tmp_path, protocol=protocol, child=child)
    run = FakeRun()
    port = FrozenPlanner(_target(bundle, tmp_path / "env"), invocations_root=tmp_path / "inv",
                         run=run)

    _failure(lambda: port.phase_a(early_input(in_process())), "frozen_content_unsupported")

    assert run.calls == []


def test_the_content_check_runs_once_per_port(world):
    strategy = in_process()
    early = early_input(strategy)
    late = late_input(strategy, early, "disabled")
    planner = _in_process()
    port, run = world.planner(_ok("a", planner.phase_a(early)), _ok("b", planner.phase_b(late)))
    port.phase_a(early)
    (world.bundle / "_algua/protocol.json").unlink()

    port.phase_b(late)

    assert len(run.calls) == 2


@pytest.mark.parametrize("case", ["collection", "bytes"])
def test_an_oversized_request_is_refused_before_launch(world, case):
    early = early_input(in_process())
    # Flat names: held names without bars would be a dark feed the supervisor reports first.
    if case == "collection":  # a JSON collection over the §6 element bound
        positions = {f"S{index:05d}": 0.0 for index in range(10_001)}
    else:  # under the element bound, over the 256 KiB request bound
        positions = {f"S{index:05d}{'X' * 40}": 0.0 for index in range(9_000)}
    port, run = world.planner()

    _failure(lambda: port.phase_a(replace(early, early_positions=positions)),
             "frozen_request_too_large")

    assert run.calls == []


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        # the planner's own validation, which the supervisor runs before its verdict
        ({"request_id": "xyz"}, "invalid_request_id: request_id must be 32 lowercase hex chars"),
        ({"max_drawdown": -0.1}, "invalid_max_drawdown: max_drawdown must be finite in [0, 1]"),
        # valid for the planner, but not a wire-v1 value
        ({"now": pd.Timestamp(NOW) + pd.Timedelta(nanoseconds=1)}, "bad_timestamp"),
    ],
    ids=["request_id", "max_drawdown", "nanosecond_now"],
)
def test_a_supervisor_input_the_planner_or_wire_refuses_is_rejected_before_launch(
        world, change, diagnostic):
    port, run = world.planner()

    failure = _failure(lambda: port.phase_a(replace(early_input(in_process()), **change)),
                       "frozen_planner_rejected")

    assert diagnostic in failure.diagnostic and run.calls == []


def test_an_early_input_for_another_gate_universe_is_refused_before_launch(world):
    port, run = world.planner(target=_target(world.bundle, world.environment, gate=("AAA",)))

    _failure(lambda: port.phase_a(early_input(in_process())), "frozen_planner_rejected")

    assert run.calls == []


# --- result mapping -----------------------------------------------------------------------------


def test_a_planner_refusal_is_a_tenant_failure(world):
    port, _ = world.planner(_ok("a", PlannerRejected("invalid_now", "bad\x00 now\n")))

    failure = _failure(lambda: port.phase_a(early_input(in_process())), "frozen_planner_rejected")

    assert failure.diagnostic == "invalid_now: bad now"


@pytest.mark.parametrize("kind", sorted(DECISION_KINDS))
def test_a_childs_decision_level_breach_keeps_breach_semantics_with_a_sanitized_detail(world,
                                                                                      kind):
    # Only the strategy's weights can break these rules, so the supervisor cannot re-derive them;
    # the child's report keeps `RiskBreach` semantics, its detail sanitized (§7).
    detail = "\x1b[31mred\x00\n" + "y" * (2 * MAX_DIAGNOSTIC_BYTES)
    port, late = _decision_world(world, PlannerRiskFailure(kind, detail, True))

    result = port.phase_b(late)

    assert isinstance(result, PlannerRiskFailure)
    assert (result.kind, result.is_dark_feed) == (kind, False)  # dark-feed is the supervisor's
    assert result.detail.startswith("[31mredyyy") and len(result.detail) == MAX_DIAGNOSTIC_BYTES
    assert set(result.detail) <= PRINTABLE


def test_the_decision_level_breach_kinds_are_exactly_the_weight_rules():
    from algua.contracts.types import ExecutionContract
    from algua.risk.limits import DECISION_BREACH_KINDS, validate_decision_weights

    contract = ExecutionContract(rebalance_frequency="1d", max_weight_per_symbol=0.5)
    raised = set()
    for weights in ({"AAA": float("nan")}, {"ZZZ": 0.1}, {"AAA": -0.1}, {"AAA": 0.6},
                    {"AAA": 0.5, "BBB": 0.5, "CCC": 0.5}):
        with pytest.raises(RiskBreach) as breach:
            validate_decision_weights(pd.Series(weights), contract, "s", ["AAA", "BBB", "CCC"])
        raised.add(breach.value.kind)

    # Append-only per wire version: every kind the validator raises today must be accepted, and a
    # retired kind stays accepted so older bundles still decode (Story 1.3c §7).
    assert raised <= DECISION_BREACH_KINDS and DECISION_KINDS <= DECISION_BREACH_KINDS
    assert DECISION_BREACH_KINDS < RISK_BREACH_KINDS
    assert not DECISION_BREACH_KINDS & DARK_FEED_KINDS


def test_an_unknown_risk_kind_is_invalid_even_past_the_codec(world, monkeypatch):
    from algua.live import frozen_dispatch

    early = early_input(in_process())
    port, _ = world.planner(_ok("a", _in_process().phase_a(early)))
    monkeypatch.setattr(frozen_dispatch, "decode_result",
                        lambda *a, **k: PlannerRiskFailure("made_up_kind", "x", False))

    _failure(lambda: port.phase_a(early), "frozen_result_invalid")


@pytest.mark.parametrize("case", ["clean", *LATE_BREACHES])
def test_a_pending_venue_belief_is_answered_as_in_process_without_a_child(world, case):
    # Before the belief the planner checks equity and drawdown; the supervisor runs that same
    # pre-belief verdict itself, so a breach there is reported one call early exactly as in
    # process, and every other case asks for the belief.
    early, late = _late_case(case)
    reference = _in_process()
    port, run = world.planner(_ok("a", reference.phase_a(early)))
    port.phase_a(early)

    answer = port.phase_b(_pending(late))

    assert answer == reference.phase_b(_pending(late))
    assert isinstance(answer, PlannerRiskFailure if case in PRE_BELIEF else VenueBeliefRequired)
    assert len(run.calls) == 1


# --- closed bars and Phase A's decision time ----------------------------------------------------


def test_closed_bars_are_the_in_process_frame(world):
    early = early_input(in_process())
    port, run = world.planner(_ok("a", _in_process().phase_a(early)))
    port.phase_a(early)

    pd.testing.assert_frame_equal(port.closed_bars(early), _in_process().closed_bars(early))
    assert len(run.calls) == 1


def test_closed_bars_with_another_decision_time_are_invalid(world):
    early = early_input(in_process())
    port, _ = world.planner(_ok("a", _in_process().phase_a(early)))
    port.phase_a(early)
    older = fixture_bars()
    shifted = replace(early, raw_bars=older[older.index < datetime(2023, 1, 5, tzinfo=UTC)])

    _failure(lambda: port.closed_bars(shifted), "frozen_result_invalid")


def test_a_snapshot_at_another_decision_time_is_invalid(world):
    early = early_input(in_process())
    first = _in_process().phase_a(early)
    assert isinstance(first, SnapshotRequired)
    port, _ = world.planner(_ok("a", replace(first, decision_ts=first.decision_ts - DAY)))

    _failure(lambda: port.phase_a(early), "frozen_result_invalid")


@pytest.mark.parametrize(
    "change",
    [
        {"reason": "warming"},
        {"decision_ts": NOW - 3 * DAY},
        {"peak_equity": 1e9},
        {"equity": 5.0},
        {"positions_before": (("OLD", 3.0),)},
    ],
    ids=["reason", "decision_ts", "peak_equity", "equity", "positions_before"],
)
def test_an_early_no_decision_the_planner_would_not_produce_is_invalid(world, change):
    empty = early_input(in_process(), empty=True)
    expected = _in_process().phase_a(empty)
    assert isinstance(expected, EarlyNoDecision) and expected.reason == "no_bars"
    if "reason" in change:
        forged = replace(expected, reason=change["reason"])
    else:
        forged = replace(expected, state=replace(expected.state, **change))
    port, _ = world.planner(_ok("a", forged))

    _failure(lambda: port.phase_a(empty), "frozen_result_invalid")


def test_an_early_no_decision_the_planner_would_produce_is_returned(world):
    empty = early_input(in_process(), empty=True)
    expected = _in_process().phase_a(empty)
    port, _ = world.planner(_ok("a", expected))

    assert port.phase_a(empty) == expected


# --- decision validation ------------------------------------------------------------------------


def _decision_world(world, decision: object):
    strategy = in_process()
    early = early_input(strategy)
    late = late_input(strategy, early, "disabled")
    port, run = world.planner(_ok("a", _in_process().phase_a(early)), _ok("b", decision))
    port.phase_a(early)
    return port, late


def _expected_decision() -> Decision:
    strategy = in_process()
    early = early_input(strategy)
    result = _in_process().phase_b(late_input(strategy, early, "disabled"))
    assert isinstance(result, Decision) and len(result.ordered_intents) == 2
    return result


def _with(decision: Decision, *, weights=None, intents=None, **state: Any) -> Decision:
    new_state = replace(decision.state, **state)
    if weights is not None:
        new_state = replace(new_state, target_weights=weights)
    return Decision(new_state, decision.ordered_intents if intents is None else tuple(intents))


def _consistent(weights: dict[str, float]) -> Decision:
    """A decision whose intents are exactly `build_intents` of `weights` (OLD is held at 0.1)."""
    good = _expected_decision()
    ts = good.state.decision_ts
    current = {"OLD": 0.1}
    intents = [
        OrderIntent(symbol, Side.BUY if weights.get(symbol, 0.0) > current.get(symbol, 0.0)
                    else Side.SELL, weights.get(symbol, 0.0), ts)
        for symbol in sorted(set(weights) | set(current))
        if abs(weights.get(symbol, 0.0) - current.get(symbol, 0.0)) > 1e-9
    ]
    return _with(good, weights=tuple(sorted(weights.items())), intents=intents)


def _forged(case: str) -> Decision:
    good = _expected_decision()
    ts = good.state.decision_ts
    buy, sell = good.ordered_intents
    if case == "intent_ts":
        return _with(good, intents=[replace(buy, decision_ts=ts - DAY), sell])
    if case == "state_ts":
        return _with(good, decision_ts=ts - DAY)
    if case == "duplicate_symbol":
        return _with(good, intents=[buy, buy, sell])
    if case == "outside_symbol":  # consistent intents for an unheld name outside the gate universe
        return _consistent({"AAA": 0.0, "BBB": 0.5, "ZZZ": 0.3})
    if case == "intents_vs_weights":
        return _with(good, intents=[replace(buy, target_weight=0.5), sell])
    if case == "intent_order":
        return _with(good, intents=[sell, buy])
    if case == "missing_intent":
        return _with(good, intents=[buy])
    if case == "peak_equity":
        return _with(good, peak_equity=1e9)
    if case == "positions_before":
        return _with(good, positions_before=(("OLD", 3.0),))
    assert case == "realized_gross"
    return _with(good, realized_gross=0.0)


#: forged decision -> the check that must refuse it (by its diagnostic).
FORGED = {
    "intent_ts": "decision timestamp",
    "state_ts": "decision timestamp",
    "duplicate_symbol": "repeat a symbol",
    "outside_symbol": "outside the gate universe and holdings: ['ZZZ']",
    "intents_vs_weights": "build_intents",
    "intent_order": "build_intents",
    "missing_intent": "build_intents",
    "peak_equity": "decision state",
    "positions_before": "decision state",
    "realized_gross": "decision state",
}


@pytest.mark.parametrize("case", sorted(FORGED))
def test_a_decision_the_supervisor_cannot_reproduce_is_invalid(world, case):
    port, late = _decision_world(world, _forged(case))

    failure = _failure(lambda: port.phase_b(late), "frozen_result_invalid")

    assert FORGED[case] in failure.diagnostic


def test_a_late_no_decision_after_a_ready_phase_a_is_invalid(world):
    decision = _expected_decision()
    port, late = _decision_world(world, LateNoDecision("warming", replace(decision.state,
                                                                          target_weights=())))

    _failure(lambda: port.phase_b(late), "frozen_result_invalid")


def test_a_reproducible_decision_is_returned(world):
    port, late = _decision_world(world, _expected_decision())

    assert port.phase_b(late) == _expected_decision()


@pytest.mark.parametrize(
    "weights,kind,detail",
    [
        ({"AAA": 0.0, "BBB": 1.5}, "max_weight_per_symbol",
         "single-name weight(s) for ['BBB'] exceed max_weight_per_symbol 1.0000"),
        ({"AAA": 0.6, "BBB": 0.6}, "gross_exposure",
         "gross exposure 1.2000 exceeds max_gross_exposure 1.0000"),
        ({"AAA": 0.0, "BBB": 0.5, "OLD": 0.2}, "out_of_universe",
         f"strategy '{STRATEGY}' returned nonzero target weight(s) for out-of-universe symbol(s) "
         "['OLD'] (allowed: ['AAA', 'BBB'])"),
        ({"AAA": -0.2, "BBB": 0.5}, "long_only",
         f"long-only: strategy '{STRATEGY}' returned negative target weight(s) for ['AAA']"),
    ],
)
def test_a_decision_breaking_the_weight_rules_is_a_breach(world, weights, kind, detail):
    port, late = _decision_world(world, _consistent(weights))

    assert port.phase_b(late) == PlannerRiskFailure(kind, detail, False)


def _float32_at_cap():
    """The fixture strategy handing back float32 weights sitting exactly at its 0.1 cap."""
    base = in_process()
    execution = replace(base.execution, max_weight_per_symbol=0.1)
    return replace(
        base, config=base.config.model_copy(update={"execution": execution}),
        construct_fn=lambda scores, view, params: pd.Series({"AAA": 0.1}, dtype="float32"))


def test_float32_weights_at_the_cap_get_one_verdict_from_child_and_supervisor(world):
    # The child validates what `plan` validated and emitted; the supervisor re-validates the
    # emitted weights as float64. Both must see the same data, or a float32 weight at the cap that
    # the child passes would trip and flatten the tenant in the supervisor.
    strategy = _float32_at_cap()
    early = early_input(strategy)
    late = late_input(strategy, early, "disabled")
    child = InProcessPlanner(overlaid(strategy))  # exactly the planner the child runs
    port, _ = world.planner(
        _ok("a", child.phase_a(early)), _ok("b", child.phase_b(late)),
        target=_target(world.bundle, world.environment, execution=overlaid(strategy).execution))
    port.phase_a(early)

    assert port.phase_b(late) == child.phase_b(late)


# --- the supervisor's own strategy-free verdict is authoritative (§7) ---------------------------


def _early_breach(case: str):
    early = early_input(in_process())
    if case == "stale_marks":  # the held name's marks are three weeks old
        return replace(early, raw_bars=_aged(early.raw_bars, "OLD"))
    if case == "no_mark":  # a held name without any bar: a dark feed
        return replace(early, early_positions={"OLD": 2.0, "ZZZ": 1.0})
    assert case == "unvaluable_marks"
    frame = early.raw_bars.copy()
    frame.loc[(frame.symbol == "OLD") & (frame.index == frame.index.max()), "close"] = float("nan")
    return replace(early, raw_bars=frame)


@pytest.mark.parametrize("case", ["stale_marks", "no_mark", "unvaluable_marks"])
def test_a_phase_a_breach_the_supervisor_finds_is_returned_whatever_the_child_says(world, case):
    early = _early_breach(case)
    expected = _in_process().phase_a(early)
    assert isinstance(expected, PlannerRiskFailure) and expected.is_dark_feed
    clean = _in_process().phase_a(early_input(in_process()))
    port, run = world.planner(_ok("a", clean))  # a child that claims all is well

    assert port.phase_a(early) == expected
    assert run.calls == []  # the verdict needs no strategy code, so no child is launched


def _forged_early(case: str) -> object:
    early = early_input(in_process())
    good = _in_process().phase_a(early)
    assert isinstance(good, SnapshotRequired) and not good.warming
    ts = good.decision_ts
    if case == "warming_no_decision":  # exactly the state a warming tick would carry
        return EarlyNoDecision("warming", PlannerState(ts, (), (("OLD", 2.0),), 0.0, None, True,
                                                       0.0))
    if case == "warming_snapshot":  # a consistent binding: only the warming flag is forged
        return SnapshotRequired(ts, True, phase_a_binding(early, decision_ts=ts, warming=True))
    if case == "binding":
        return replace(good, phase_a_binding="f" * 64)
    return PlannerRiskFailure(case, "forged", case in DARK_FEED_KINDS)


@pytest.mark.parametrize(
    "case", ["warming_no_decision", "warming_snapshot", "binding", *sorted(STRATEGY_FREE_KINDS)])
def test_a_phase_a_answer_other_than_the_supervisors_verdict_is_invalid(world, case):
    # A ready strategy with fresh marks: the child may not report it warming, bind another
    # outcome, or claim a breach -- a false `stale_marks` would halt the whole account.
    port, run = world.planner(_ok("a", _forged_early(case)))

    _failure(lambda: port.phase_a(early_input(in_process())), "frozen_result_invalid")

    assert len(run.calls) == 1


def _warm():
    """The fixture strategy with a warm-up longer than the fixture's three closed sessions."""
    base = in_process()
    execution = replace(base.execution, warmup_bars=5)
    return replace(base, config=base.config.model_copy(update={"execution": execution}))


def _warm_world(world, *answers: Any, held: bool = True):
    warm = _warm()
    early = early_input(warm)
    if not held:
        early = replace(early, early_positions={})
    port, run = world.planner(*answers, target=_target(world.bundle, world.environment,
                                                       execution=overlaid(warm).execution))
    return port, run, early, InProcessPlanner(overlaid(warm))


def test_a_flat_warming_tick_is_the_supervisors_no_decision(world):
    _, _, early, reference = _warm_world(world, held=False)
    expected = reference.phase_a(early)
    assert isinstance(expected, EarlyNoDecision) and expected.reason == "warming"
    port, _, _, _ = _warm_world(world, _ok("a", expected), held=False)
    assert port.phase_a(early) == expected

    snapshot = SnapshotRequired(expected.state.decision_ts, True, phase_a_binding(
        early, decision_ts=expected.state.decision_ts, warming=True))
    port, _, _, _ = _warm_world(world, _ok("a", snapshot), held=False)
    _failure(lambda: port.phase_a(early), "frozen_result_invalid")


def _warm_phase_b(world, answer: object):
    _, _, early, reference = _warm_world(world)
    first = reference.phase_a(early)
    assert isinstance(first, SnapshotRequired) and first.warming
    port, run, _, _ = _warm_world(world, _ok("a", first), _ok("b", answer))
    port.phase_a(early)
    late = late_input(_warm(), early, "disabled")
    return port, late, reference.phase_b(late)


def test_a_warming_late_no_decision_the_planner_would_produce_is_returned(world):
    _, _, expected = _warm_phase_b(world, _expected_decision())
    assert isinstance(expected, LateNoDecision)
    port, late, _ = _warm_phase_b(world, expected)

    assert port.phase_b(late) == expected


@pytest.mark.parametrize("answer", ["decision", "state", "breach"])
def test_a_warming_phase_b_answered_otherwise_is_invalid(world, answer):
    _, _, expected = _warm_phase_b(world, _expected_decision())
    assert isinstance(expected, LateNoDecision)
    forged = {
        "decision": _expected_decision(),
        "state": replace(expected, state=replace(expected.state, peak_equity=1e9)),
        "breach": PlannerRiskFailure("gross_exposure", "forged", False),  # no weights to break
    }[answer]
    port, late, _ = _warm_phase_b(world, forged)

    _failure(lambda: port.phase_b(late), "frozen_result_invalid")


@pytest.mark.parametrize("case", LATE_BREACHES)
def test_a_phase_b_breach_the_supervisor_finds_is_returned_whatever_the_child_says(world, case):
    early, late = _late_case(case)
    reference = _in_process()
    expected = reference.phase_b(late)
    assert isinstance(expected, PlannerRiskFailure) and expected.kind == case
    port, run = world.planner(_ok("a", reference.phase_a(early)), _ok("b", _expected_decision()))
    port.phase_a(early)

    assert port.phase_b(late) == expected
    assert len(run.calls) == 1  # no Phase B child


@pytest.mark.parametrize("kind", sorted(STRATEGY_FREE_KINDS))
def test_a_phase_b_breach_the_supervisor_does_not_find_is_invalid_never_a_halt(world, kind):
    port, late = _decision_world(world, PlannerRiskFailure(kind, "forged", kind in DARK_FEED_KINDS))

    _failure(lambda: port.phase_b(late), "frozen_result_invalid")


@pytest.mark.parametrize(
    "change,code",
    [
        ({"captured": {"drawdown_equity": float("nan")}}, "invalid_captured_state"),
        ({"captured": {"market_values": {}}}, "invalid_captured_state"),
        ({"phase_a_binding": "0" * 64}, "phase_a_binding_mismatch"),
        ({"captured": {"request_id": "f" * 32}}, "request_id_mismatch"),
    ],
    ids=["non_finite", "symbol_sets", "binding", "request_id"],
)
def test_a_late_input_the_planner_refuses_is_rejected_before_a_child(world, change, code):
    strategy = in_process()
    early = early_input(strategy)
    late = late_input(strategy, early, "disabled")
    late = replace(late, captured=replace(late.captured, **change.get("captured", {})),
                   **{key: value for key, value in change.items() if key != "captured"})
    expected = _in_process().phase_b(late)
    assert not isinstance(expected, PlannerRiskFailure) and expected.code == code
    port, run = world.planner(_ok("a", _in_process().phase_a(early)),
                              _ok("b", _expected_decision()))
    port.phase_a(early)

    failure = _failure(lambda: port.phase_b(late), "frozen_planner_rejected")

    assert failure.diagnostic == f"{expected.code}: {expected.detail}"
    assert len(run.calls) == 1


def test_phase_b_before_a_phase_a_snapshot_is_a_supervisor_error(world):
    strategy = in_process()
    early = early_input(strategy)
    port, run = world.planner()

    with pytest.raises(RuntimeError, match="Phase A"):
        port.phase_b(late_input(strategy, early, "disabled"))
    assert run.calls == []


# --- end to end: the real child from a real bundle ----------------------------------------------


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory):
    with frozen_store(tmp_path_factory.mktemp("store")) as built:
        yield built


def _real_target(store) -> FrozenTarget:
    return FrozenTarget(
        identity=IDENTITY,
        bundle_root=store.bundle("good"),
        environment_root=store.environment,
        interpreter=store.python,
        execution=_strategy().execution,
        gate_universe=GATE,
    )


def _canonical(phase: str, result: object) -> bytes:
    return encode_result(phase, REQUEST_ID, result)


@pytest.mark.parametrize("case", ["no_bars", "warming", "decision", "drawdown"])
def test_the_frozen_port_equals_the_in_process_planner(store, tmp_path, case):
    strategy = in_process()
    early = early_input(strategy, empty=case == "no_bars")
    if case == "warming":  # bars, but none for the gate universe, and nothing held
        frame = early.raw_bars
        early = replace(early, raw_bars=frame[~frame.symbol.isin(GATE)], early_positions={})
    reference = _in_process()
    port = FrozenPlanner(_real_target(store), invocations_root=tmp_path / "invocations")

    first = port.phase_a(early)

    assert _canonical("a", first) == _canonical("a", reference.phase_a(early))
    if case in ("no_bars", "warming"):
        assert isinstance(first, EarlyNoDecision) and first.reason == case
        return
    assert isinstance(first, SnapshotRequired)
    pd.testing.assert_frame_equal(port.closed_bars(early), reference.closed_bars(early))
    late = late_input(strategy, early, "enabled" if case == "decision" else "drawdown")
    pending = port.phase_b(_pending(late))  # the drawdown breach is reported here, as in process
    assert pending == reference.phase_b(_pending(late))
    assert isinstance(pending, VenueBeliefRequired if case == "decision" else PlannerRiskFailure)
    second = port.phase_b(late)
    assert _canonical("b", second) == _canonical("b", reference.phase_b(late))
    assert isinstance(second, Decision if case == "decision" else PlannerRiskFailure)
    assert _leftovers(tmp_path / "invocations") == []


# --- end to end: run_tick's effect trace --------------------------------------------------------


CONTEXT = PlannerContext(DEPLOYMENT_ID, ARTIFACT_ID, MANIFEST, config_hash(in_process()),
                         recorded_json(in_process()), CALENDAR)


def _tick(port_for, case: str) -> tuple[list[Any], Any]:
    """One paper-shaped tick over recording fakes; returns (effect trace, result or breach)."""
    events: list[Any] = []
    frame = fixture_bars(empty=case == "no_bars")
    if case == "stale_marks":  # OLD's marks are three weeks old: a dark feed for a held name
        frame = _aged(frame, "OLD")
    held = {} if case in ("no_bars", "warming") else {"OLD": 2.0}
    snap = TickSnapshot(equity=100.0, market_values={"AAA": 0.0, "BBB": 0.0, "OLD": 10.0},
                        qtys={"AAA": 0.0, "BBB": 0.0, "OLD": 2.0})
    belief = {"OLD": 3.0} if case == "reconcile" else {"OLD": 2.0}

    def get_bars(symbols, start, end, timeframe):
        events.append(("fetch", tuple(symbols), timeframe))
        if case == "warming":  # closed bars, but none for the gate universe: a flat book warms up
            return frame[frame.symbol == "CCC"]
        return frame[frame.symbol.isin(symbols)]

    def live_snapshot(closed):
        events.append(("live_snapshot", bars_digest(closed)))
        return snap, 100.0

    def submit_sized(intent, snapshot, coid, reserve=None):
        events.append(("submit", intent, snapshot, coid))
        return "noop" if intent.target_weight == 0.0 else f"order-{intent.symbol}"

    strategy = _strategy()
    broker = SimpleNamespace(
        get_positions=lambda: events.append("positions") or dict(held),
        snapshot=lambda universe: events.append(("snapshot", tuple(universe))) or snap,
        cancel_open_orders=lambda: events.append("cancel"),
        submit_sized=submit_sized,
    )
    hooks = TickHooks(
        client_order_id_for=lambda name, ts, symbol: f"{name}|{ts.isoformat()}|{symbol}",
        on_submitted=lambda order: events.append(("submitted", order)),
        should_halt=lambda: events.append("halt") or False,
        venue_belief=lambda: events.append("belief") or dict(belief),
        peak_equity=200.0 if case == "drawdown" else 100.0,
        live_snapshot=live_snapshot if case == "decision" else None,
        before_submit=lambda intent, coid: events.append(("before", intent, coid)),
        on_noop=lambda intent, coid: events.append(("noop", intent, coid)),
        planner_context=CONTEXT,
        planner=port_for(strategy),
    )
    try:
        outcome: Any = run_tick(strategy, broker, SimpleNamespace(get_bars=get_bars),
                                NOW - 10 * DAY, NOW, now=NOW, hooks=hooks, max_drawdown=0.1)
    except RiskBreach as breach:
        outcome = (breach.kind, breach.detail)
    return events, outcome


@pytest.mark.parametrize(
    "case", ["decision", "no_bars", "warming", "stale_marks", "reconcile", "drawdown"])
def test_run_tick_through_the_frozen_port_has_the_in_process_effect_trace(store, tmp_path, case):
    in_process_trace, in_process_outcome = _tick(lambda strategy: None, case)
    frozen_trace, frozen_outcome = _tick(
        lambda strategy: FrozenPlanner(_real_target(store), invocations_root=tmp_path / "inv"),
        case)

    # The same outcome -- a breach with identical kind AND text (planner breach texts are plain
    # ASCII, so the §7 sanitizer leaves them untouched) -- through the identical effect trace:
    # the supervisor reports the drawdown breach before reading the belief, as in process.
    assert frozen_outcome == in_process_outcome
    assert frozen_trace == in_process_trace
    if case in ("stale_marks", "reconcile", "drawdown"):
        assert in_process_outcome[0] == case
    elif case == "decision":
        assert frozen_outcome.submitted and frozen_outcome.target_weights
        assert [event[0] for event in frozen_trace if isinstance(event, tuple)].count(
            "submitted") == 1
        assert any(event[0] == "live_snapshot" for event in frozen_trace
                   if isinstance(event, tuple))
    else:  # an early no-decision: no snapshot, no belief, no submit
        assert frozen_outcome.submitted == [] and frozen_outcome.target_weights == {}
        assert frozen_trace == ["positions", ("fetch", GATE, "1d")]
    assert _leftovers(tmp_path / "inv") == []
