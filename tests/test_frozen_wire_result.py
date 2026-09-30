"""Story 1.3c contract §7: the frozen planner's result codec (one JSON document on stdout).

`encode_result` is what the child writes; `decode_result` is the supervisor's strict reader. Every
result kind round-trips to the existing planner value (planner refusals to `PlannerRejected`), and
every structural deviation is refused with a machine reason. Semantic allowlists and cross-checks
against Phase A belong to the dispatcher, not here.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
import pytest

from algua.contracts.canonical import canonical_json
from algua.contracts.types import OrderIntent, Side
from algua.live import frozen_wire
from algua.live.frozen_wire import MAX_JSON_COLLECTION, MAX_JSON_DEPTH, WireError
from algua.live.frozen_wire_result import PlannerRejected, decode_result, encode_result
from algua.live.planner import phase_a, phase_b
from algua.live.planner_contract import (
    Decision,
    EarlyNoDecision,
    LateNoDecision,
    PhaseBindingFailure,
    PlannerInputFailure,
    PlannerRiskFailure,
    PlannerState,
    SnapshotRequired,
    VenueBeliefRequired,
)
from tests.test_frozen_wire import (
    DELETE,
    REQUEST_ID,
    edit,
    make_captured,
    make_early,
    make_late,
    make_strategy,
)

TS = pd.Timestamp("2023-01-04", tz="UTC")
STATE = PlannerState(TS, (("AAA", 0.5),), (("OLD", 2.0),), 100.0, 100.0, True, 0.2)
EMPTY = PlannerState(None, (), (), 0.0, None, True, 0.0)
INTENTS = (OrderIntent("AAA", Side.BUY, 0.5, TS), OrderIntent("OLD", Side.SELL, 0.0, TS))

ROUND_TRIPS = [
    ("a", EarlyNoDecision("no_bars", EMPTY), EarlyNoDecision("no_bars", EMPTY)),
    ("a", EarlyNoDecision("warming", STATE), EarlyNoDecision("warming", STATE)),
    ("a", SnapshotRequired(TS, True, "f" * 64), SnapshotRequired(TS, True, "f" * 64)),
    ("a", SnapshotRequired(None, False, "0" * 64), SnapshotRequired(None, False, "0" * 64)),
    ("a", PlannerRiskFailure("stale_marks", "stale AAA", True),
     PlannerRiskFailure("stale_marks", "stale AAA", True)),
    ("b", PlannerRiskFailure("drawdown", "déjà 12%", False),
     PlannerRiskFailure("drawdown", "déjà 12%", False)),
    ("a", PlannerInputFailure("invalid_now", "now must be UTC"),
     PlannerRejected("invalid_now", "now must be UTC")),
    ("b", PhaseBindingFailure("phase_a_binding_mismatch", "x"),
     PlannerRejected("phase_a_binding_mismatch", "x")),
    ("b", PlannerRejected("invalid_captured_state", ""),
     PlannerRejected("invalid_captured_state", "")),
    ("b", LateNoDecision("warming", STATE), LateNoDecision("warming", STATE)),
    ("b", Decision(STATE, INTENTS), Decision(STATE, INTENTS)),
    ("b", Decision(EMPTY, ()), Decision(EMPTY, ())),
]


@pytest.mark.parametrize(("phase", "result", "expected"), ROUND_TRIPS)
def test_every_result_kind_round_trips(phase, result, expected):
    stdout = encode_result(phase, REQUEST_ID, result)
    assert stdout.endswith(b"\n") and stdout.count(b"\n") == 1
    assert stdout[:-1] == canonical_json(json.loads(stdout)).encode("utf-8")
    decoded = decode_result(stdout, phase=phase, request_id=REQUEST_ID)
    assert decoded == expected
    assert encode_result(phase, REQUEST_ID, decoded) == stdout
    assert decode_result(stdout[:-1], phase=phase, request_id=REQUEST_ID) == expected


def test_result_envelope_and_state_shape():
    root = json.loads(encode_result("b", REQUEST_ID, Decision(STATE, INTENTS)))
    assert root == {
        "wire": {"name": "frozen-planner", "version": 1},
        "phase": "b",
        "request_id": REQUEST_ID,
        "result": {
            "kind": "decision",
            "state": {
                "decision_ts": "2023-01-04T00:00:00.000000+00:00",
                "target_weights": [["AAA", (0.5).hex()]],
                "positions_before": [["OLD", (2.0).hex()]],
                "equity": (100.0).hex(),
                "peak_equity": (100.0).hex(),
                "reconcile_ok": True,
                "realized_gross": (0.2).hex(),
            },
            "intents": [
                {"symbol": "AAA", "side": "buy", "target_weight": (0.5).hex(),
                 "decision_ts": "2023-01-04T00:00:00.000000+00:00"},
                {"symbol": "OLD", "side": "sell", "target_weight": (0.0).hex(),
                 "decision_ts": "2023-01-04T00:00:00.000000+00:00"},
            ],
        },
    }
    risk = json.loads(encode_result("a", REQUEST_ID, PlannerRiskFailure("reconcile", "d", False)))
    assert risk["result"] == {"kind": "risk_failure", "risk_kind": "reconcile", "detail": "d"}


def test_real_planner_results_round_trip():
    strategy = make_strategy()
    early = make_early(strategy)
    first = phase_a(strategy, early)
    assert isinstance(first, SnapshotRequired)
    assert decode_result(encode_result("a", REQUEST_ID, first), phase="a",
                         request_id=REQUEST_ID) == first
    second = phase_b(strategy, make_late(strategy, early, make_captured()))
    assert isinstance(second, Decision)
    assert decode_result(encode_result("b", REQUEST_ID, second), phase="b",
                         request_id=REQUEST_ID) == second


def test_state_mappings_are_sorted_and_special_floats_cross_losslessly():
    state = PlannerState(
        TS, (("ZZZ", -0.0), ("AAA", math.nan)), (("B", math.inf), ("A", -math.inf)),
        5e-324, None, False, 0,
    )
    stdout = encode_result("a", REQUEST_ID, EarlyNoDecision("warming", state))
    decoded = decode_result(stdout, phase="a", request_id=REQUEST_ID)
    assert isinstance(decoded, EarlyNoDecision)
    got = decoded.state
    assert [s for s, _ in got.target_weights] == ["AAA", "ZZZ"]
    assert math.isnan(got.target_weights[0][1])
    assert math.copysign(1.0, got.target_weights[1][1]) == -1.0
    assert got.positions_before == (("A", -math.inf), ("B", math.inf))
    assert got.equity == 5e-324 and got.realized_gross == 0.0 and got.reconcile_ok is False
    assert encode_result("a", REQUEST_ID, decoded) == stdout


@pytest.mark.parametrize(
    ("phase", "result", "reason"),
    [
        ("b", VenueBeliefRequired(), "not_encodable"),
        ("a", object(), "not_encodable"),
        ("a", Decision(STATE, INTENTS), "forbidden_kind"),
        ("a", LateNoDecision("warming", STATE), "forbidden_kind"),
        ("b", SnapshotRequired(TS, False, "f" * 64), "forbidden_kind"),
        ("b", EarlyNoDecision("no_bars", EMPTY), "forbidden_kind"),
        ("c", PlannerRejected("x", "y"), "bad_phase"),
        ("a", EarlyNoDecision("no_bars", PlannerState(None, (("A", 1.0), ("A", 2.0)), (), 0.0,
                                                      None, True, 0.0)), "bad_mapping"),
        ("a", EarlyNoDecision("no_bars", PlannerState(pd.Timestamp("2023-01-04"), (), (), 0.0,
                                                      None, True, 0.0)), "bad_timestamp"),
        ("b", Decision(STATE, (OrderIntent("AAA", "hold", 0.5, TS),)), "bad_value"),
    ],
)
def test_encode_refuses_what_the_wire_cannot_carry(phase, result, reason):
    with pytest.raises(WireError) as caught:
        encode_result(phase, REQUEST_ID, result)
    assert caught.value.reason == reason


def _stdout(phase: str = "b") -> bytes:
    result = Decision(STATE, INTENTS) if phase == "b" else SnapshotRequired(TS, False, "f" * 64)
    return encode_result(phase, REQUEST_ID, result)


def _redo(stdout: bytes, path: tuple[Any, ...], value: Any) -> bytes:
    return edit(stdout[:-1], path, value) + b"\n"


@pytest.mark.parametrize(
    ("phase", "path", "value", "reason"),
    [
        ("b", ("request_id",), "f" * 32, "echo_mismatch"),
        ("b", ("phase",), "a", "echo_mismatch"),
        ("b", ("wire", "version"), 2, "bad_wire"),
        ("b", ("wire", "version"), True, "bad_wire"),
        ("b", ("result", "kind"), "snapshot_required", "forbidden_kind"),
        ("a", ("result", "kind"), "decision", "forbidden_kind"),
        ("b", ("result", "kind"), "explode", "unknown_kind"),
        ("b", ("result",), [], "bad_type"),
        ("b", ("result", "extra"), 1, "unknown_key"),
        ("b", ("result", "state", "extra"), 1, "unknown_key"),
        ("b", ("result", "intents", 0, "extra"), 1, "unknown_key"),
        ("b", ("request_id",), DELETE, "missing_key"),
        ("b", ("result", "state", "equity"), DELETE, "missing_key"),
        ("b", ("result", "state", "equity"), True, "bad_float"),
        ("b", ("result", "state", "equity"), 100.0, "bad_float"),
        ("b", ("result", "intents", 0, "target_weight"), False, "bad_float"),
        ("b", ("result", "state", "reconcile_ok"), 1, "bad_type"),
        ("a", ("result", "warming"), 0, "bad_type"),
        ("b", ("result", "intents", 0, "side"), "hold", "bad_value"),
        ("b", ("result", "intents"), {}, "bad_type"),
        ("b", ("result", "state", "decision_ts"), "2023-01-04T00:00:00.000000", "bad_timestamp"),
        ("b", ("result", "state", "target_weights"), [["Z", "0x0.0p+0"], ["A", "0x0.0p+0"]],
         "mapping_order"),
        ("a", ("result", "phase_a_binding"), "F" * 64, "bad_hex"),
        ("b", ("result", "state", "target_weights"),
         [[f"S{i:05d}", "0x0.0p+0"] for i in range(MAX_JSON_COLLECTION + 1)], "json_collection"),
    ],
)
def test_decode_refuses_structural_deviations(phase, path, value, reason):
    with pytest.raises(WireError) as caught:
        decode_result(_redo(_stdout(phase), path, value), phase=phase, request_id=REQUEST_ID)
    assert caught.value.reason == reason


def _nested_list(depth: int) -> Any:
    value: Any = []
    for _ in range(depth):
        value = [value]
    return value


def _body(result: dict[str, Any], phase: str) -> bytes:
    envelope = {"wire": {"name": "frozen-planner", "version": 1}, "phase": phase,
                "request_id": REQUEST_ID, "result": result}
    return canonical_json(envelope).encode("utf-8") + b"\n"


@pytest.mark.parametrize(
    ("phase", "result", "reason"),
    [
        ("a", {"kind": "early_no_decision", "reason": "stale", "state": {}}, "bad_value"),
        ("b", {"kind": "late_no_decision", "reason": "no_bars", "state": {}}, "bad_value"),
        ("a", {"kind": "risk_failure", "risk_kind": 5, "detail": "d"}, "bad_type"),
        ("b", {"kind": "planner_rejected", "code": "c", "detail": None}, "bad_type"),
        ("b", {"kind": "planner_rejected", "code": "c", "detail": _nested_list(MAX_JSON_DEPTH)},
         "json_depth"),
    ],
)
def test_decode_refuses_malformed_result_bodies(phase, result, reason):
    with pytest.raises(WireError) as caught:
        decode_result(_body(result, phase), phase=phase, request_id=REQUEST_ID)
    assert caught.value.reason == reason


def _raw_cases() -> list[tuple[str, Any, str]]:
    def whitespace(data: bytes) -> bytes:
        return json.dumps(json.loads(data), sort_keys=True, indent=1).encode()

    return [
        ("duplicate_key", lambda d: b'{"phase":"b",' + d[1:], "duplicate_key"),
        ("trailing_data", lambda d: d + b"{}", "invalid_json"),
        ("two_newlines", lambda d: d + b"\n", "not_canonical"),
        ("leading_space", lambda d: b" " + d, "not_canonical"),
        ("whitespace", whitespace, "not_canonical"),
        ("nan_token", lambda d: d.replace(f'"{(100.0).hex()}"'.encode(), b"NaN", 1), "non_finite"),
        ("invalid_utf8", lambda d: d.replace(b'"AAA"', b'"A\xffA"'), "invalid_utf8"),
        ("empty", lambda d: b"", "invalid_json"),
        ("oversize", lambda d: b" " * (frozen_wire.MAX_STDOUT_BYTES + 1), "output_too_large"),
    ]


@pytest.mark.parametrize(("case", "mutate", "reason"), _raw_cases(), ids=lambda v: str(v)[:20])
def test_decode_refuses_non_canonical_stdout(case, mutate, reason):
    stdout = _stdout("b")
    mutated = mutate(stdout)
    assert mutated != stdout, case
    with pytest.raises(WireError) as caught:
        decode_result(mutated, phase="b", request_id=REQUEST_ID)
    assert caught.value.reason == reason
