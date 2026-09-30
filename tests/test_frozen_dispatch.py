"""Story 1.3c contract §3–§8: the supervisor side of the frozen planner (`frozen_dispatch`).

The fast tests drive `FrozenPlanner` with a fake `run` that answers crafted `ContainedResult`s and
inspects the sealed invocation directory while the "child" runs. The end-to-end tests launch the
real child from a real bundle through the shared harness and require the frozen port to be
indistinguishable from the in-process planner: equal canonical results per phase, and an equal
effect trace for a whole `run_tick`.
"""

from __future__ import annotations

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
from algua.live.planner_binding import bars_digest
from algua.live.planner_contract import (
    Decision,
    EarlyNoDecision,
    LateNoDecision,
    PlannerRiskFailure,
    SnapshotRequired,
    VenueBeliefPending,
    VenueBeliefRequired,
)
from algua.primitives.contained_process import ContainedResult
from algua.risk.limits import RiskBreach
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


def _target(bundle: Path, environment: Path, *, gate: tuple[str, ...] = GATE) -> FrozenTarget:
    return FrozenTarget(
        identity=IDENTITY,
        bundle_root=bundle,
        environment_root=environment,
        interpreter=environment / "bin/python",
        execution=_strategy().execution,
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
        if isinstance(answer, BaseException):
            raise answer
        return answer


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
        (FileNotFoundError(2, "no interpreter"), FrozenTenantFailure),
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
        (PermissionError(13, "denied"), "frozen_launch_failed", ["denied"]),
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
    if case == "collection":  # a JSON collection over the §6 element bound
        positions = {f"S{index:05d}": 1.0 for index in range(10_001)}
    else:  # under the element bound, over the 256 KiB request bound
        positions = {f"S{index:05d}{'X' * 40}": 1.0 for index in range(9_000)}
    port, run = world.planner()

    _failure(lambda: port.phase_a(replace(early, early_positions=positions)),
             "frozen_request_too_large")

    assert run.calls == []


def test_an_unencodable_supervisor_input_is_planner_rejected_before_launch(world):
    port, run = world.planner()

    failure = _failure(lambda: port.phase_a(replace(early_input(in_process()), request_id="xyz")),
                       "frozen_planner_rejected")

    assert "bad_hex" in failure.diagnostic and run.calls == []


def test_an_early_input_for_another_gate_universe_is_refused_before_launch(world):
    port, run = world.planner(target=_target(world.bundle, world.environment, gate=("AAA",)))

    _failure(lambda: port.phase_a(early_input(in_process())), "frozen_planner_rejected")

    assert run.calls == []


# --- result mapping -----------------------------------------------------------------------------


def test_a_planner_refusal_is_a_tenant_failure(world):
    port, _ = world.planner(_ok("a", PlannerRejected("invalid_now", "bad\x00 now\n")))

    failure = _failure(lambda: port.phase_a(early_input(in_process())), "frozen_planner_rejected")

    assert failure.diagnostic == "invalid_now: bad now"


@pytest.mark.parametrize("kind,dark", [("stale_marks", True), ("drawdown", False)])
def test_a_child_risk_failure_keeps_breach_semantics_with_a_sanitized_detail(world, kind, dark):
    detail = "\x1b[31mred\x00\n" + "y" * (2 * MAX_DIAGNOSTIC_BYTES)
    port, _ = world.planner(_ok("a", PlannerRiskFailure(kind, detail, not dark)))

    result = port.phase_a(early_input(in_process()))

    assert isinstance(result, PlannerRiskFailure)
    assert (result.kind, result.is_dark_feed) == (kind, dark)
    assert result.detail.startswith("[31mredyyy") and len(result.detail) == MAX_DIAGNOSTIC_BYTES
    assert set(result.detail) <= PRINTABLE


def test_an_unknown_risk_kind_is_invalid_even_past_the_codec(world, monkeypatch):
    from algua.live import frozen_dispatch

    early = early_input(in_process())
    port, _ = world.planner(_ok("a", _in_process().phase_a(early)))
    monkeypatch.setattr(frozen_dispatch, "decode_result",
                        lambda *a, **k: PlannerRiskFailure("made_up_kind", "x", False))

    _failure(lambda: port.phase_a(early), "frozen_result_invalid")


def test_a_pending_venue_belief_is_answered_without_launching_a_child(world):
    strategy = in_process()
    early = early_input(strategy)
    late = late_input(strategy, early, "drawdown")  # in process, this breach precedes the belief
    port, run = world.planner(_ok("a", _in_process().phase_a(early)))
    port.phase_a(early)

    assert port.phase_b(_pending(late)) == VenueBeliefRequired()
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


def _decision_world(world, decision: object, *, snapshot: SnapshotRequired | None = None):
    strategy = in_process()
    early = early_input(strategy)
    late = late_input(strategy, early, "disabled")
    first = _in_process().phase_a(early) if snapshot is None else snapshot
    port, run = world.planner(_ok("a", first), _ok("b", decision))
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


def test_a_decision_on_captured_values_the_planner_refuses_is_invalid(world):
    port, late = _decision_world(world, _expected_decision())
    broke = replace(late, captured=replace(late.captured, sizing_equity=0.0))

    assert "admit no late result" in _failure(lambda: port.phase_b(broke),
                                              "frozen_result_invalid").diagnostic


def test_a_decision_after_a_warming_phase_a_is_invalid(world):
    first = _in_process().phase_a(early_input(in_process()))
    assert isinstance(first, SnapshotRequired) and not first.warming
    port, late = _decision_world(world, _expected_decision(),
                                 snapshot=replace(first, warming=True))

    _failure(lambda: port.phase_b(late), "frozen_result_invalid")


def test_a_late_no_decision_after_a_ready_phase_a_is_invalid(world):
    decision = _expected_decision()
    port, late = _decision_world(world, LateNoDecision("warming", replace(decision.state,
                                                                          target_weights=())))

    _failure(lambda: port.phase_b(late), "frozen_result_invalid")


def test_a_late_no_decision_with_a_state_it_cannot_have_is_invalid(world):
    first = _in_process().phase_a(early_input(in_process()))
    decision = _expected_decision()
    state = replace(decision.state, target_weights=(), peak_equity=1e9)
    port, late = _decision_world(world, LateNoDecision("warming", state),
                                 snapshot=replace(first, warming=True))

    _failure(lambda: port.phase_b(late), "frozen_result_invalid")


def test_a_late_no_decision_the_planner_would_produce_is_returned(world):
    first = _in_process().phase_a(early_input(in_process()))
    state = replace(_expected_decision().state, target_weights=())
    port, late = _decision_world(world, LateNoDecision("warming", state),
                                 snapshot=replace(first, warming=True))

    assert port.phase_b(late) == LateNoDecision("warming", state)


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
    assert port.phase_b(_pending(late)) == VenueBeliefRequired()
    second = port.phase_b(late)
    assert _canonical("b", second) == _canonical("b", reference.phase_b(late))
    assert isinstance(second, Decision if case == "decision" else PlannerRiskFailure)
    assert list((tmp_path / "invocations").iterdir()) == []


# --- end to end: run_tick's effect trace --------------------------------------------------------


CONTEXT = PlannerContext(DEPLOYMENT_ID, ARTIFACT_ID, MANIFEST, config_hash(in_process()),
                         recorded_json(in_process()), CALENDAR)


def _tick(port_for, case: str) -> tuple[list[Any], Any]:
    """One paper-shaped tick over recording fakes; returns (effect trace, result or breach)."""
    events: list[Any] = []
    frame = fixture_bars()
    if case == "stale_marks":  # OLD's marks are three weeks old: a dark feed for a held name
        frame.index = pd.DatetimeIndex(
            [ts - timedelta(days=21) if symbol == "OLD" else ts
             for ts, symbol in zip(frame.index, frame.symbol, strict=True)], name="timestamp")
    snap = TickSnapshot(equity=100.0, market_values={"AAA": 0.0, "BBB": 0.0, "OLD": 10.0},
                        qtys={"AAA": 0.0, "BBB": 0.0, "OLD": 2.0})
    belief = {"OLD": 3.0} if case == "reconcile" else {"OLD": 2.0}

    def get_bars(symbols, start, end, timeframe):
        events.append(("fetch", tuple(symbols), timeframe))
        return frame[frame.symbol.isin(symbols)]

    def live_snapshot(closed):
        events.append(("live_snapshot", bars_digest(closed)))
        return snap, 100.0

    def submit_sized(intent, snapshot, coid, reserve=None):
        events.append(("submit", intent, snapshot, coid))
        return "noop" if intent.target_weight == 0.0 else f"order-{intent.symbol}"

    strategy = _strategy()
    broker = SimpleNamespace(
        get_positions=lambda: events.append("positions") or {"OLD": 2.0},
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


@pytest.mark.parametrize("case", ["decision", "stale_marks", "reconcile", "drawdown"])
def test_run_tick_through_the_frozen_port_has_the_in_process_effect_trace(store, tmp_path, case):
    in_process_trace, in_process_outcome = _tick(lambda strategy: None, case)
    frozen_trace, frozen_outcome = _tick(
        lambda strategy: FrozenPlanner(_real_target(store), invocations_root=tmp_path / "inv"),
        case)

    if case == "decision":
        assert frozen_outcome == in_process_outcome
        assert frozen_outcome.submitted and frozen_outcome.target_weights
    else:  # a breach: its kind, then the in-process detail as §7 sanitizes it
        kind, detail = in_process_outcome
        assert kind == case
        assert frozen_outcome == (kind, sanitize_text(detail))
    if case == "drawdown":
        # Contract §3: the frozen port answers the pending belief without a child, so the
        # supervisor reads the venue belief once before Phase B reports the drawdown breach that
        # the in-process planner reports one call earlier. The read is the only difference.
        assert in_process_outcome[0] == "drawdown"
        assert frozen_trace == [*in_process_trace, "belief"]
    else:
        assert frozen_trace == in_process_trace
    if case == "decision":
        assert [event[0] for event in frozen_trace if isinstance(event, tuple)].count(
            "submitted") == 1
        assert any(event[0] == "live_snapshot" for event in frozen_trace
                   if isinstance(event, tuple))
    assert list((tmp_path / "inv").iterdir()) == []
